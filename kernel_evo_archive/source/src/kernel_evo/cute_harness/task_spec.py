"""Compact, source-derived routing facts for Python CuTe DSL authoring."""

from __future__ import annotations

import ast
import re
from typing import Any, Iterable


_OPERATION_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("attention", ("attention", "scaled_dot_product_attention", "softmax")),
    ("convolution", ("conv1d", "conv2d", "conv3d", "convolution")),
    ("gemm", ("gemm", "matmul", "bmm", "einsum", "linear", "mm")),
    ("reduction", ("sum", "mean", "amax", "amin", "max", "min", "argmax", "reduce")),
)


def infer_operation(source: str) -> str:
    """Infer the dominant operation family without executing user source."""
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(source or "")).lower()
    for operation, needles in _OPERATION_PATTERNS:
        if any(re.search(rf"\b{re.escape(needle)}\b", text) for needle in needles):
            return operation
    return "elementwise"


def _qualified_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _qualified_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    return ""


def _literal(value: ast.AST | None) -> Any:
    if value is None:
        return None
    try:
        parsed = ast.literal_eval(value)
    except (ValueError, TypeError):
        return None
    if isinstance(parsed, (bool, int, float, str)):
        return parsed
    if isinstance(parsed, (list, tuple)) and len(parsed) <= 8:
        if all(isinstance(item, (bool, int, float, str)) for item in parsed):
            return list(parsed)
    return None


def _forward_function(tree: ast.Module) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    candidates: list[tuple[int, ast.FunctionDef | ast.AsyncFunctionDef]] = []
    class_priority = {"Model": 0, "_RefModel": 1, "ModelNew": 2}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)) and item.name == "forward":
                candidates.append((class_priority.get(node.name, 3), item))
    return min(candidates, default=(99, None), key=lambda item: item[0])[1]


def _unique(values: Iterable[str], *, limit: int) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
        if len(result) >= limit:
            break
    return result


def extract_task_spec(
    source: str,
    *,
    operation: str = "",
    precision: str = "bf16",
    runtime_precision: str = "",
    arch: str = "",
) -> dict[str, Any]:
    """Return bounded task facts used to route an author to the right harness material.

    This intentionally does not pretend to recover a complete tensor ABI from arbitrary
    Python. Unknown facts stay unknown instead of being guessed.
    """
    text = str(source or "")
    inferred = str(operation or infer_operation(text)).lower()
    spec: dict[str, Any] = {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "operation": inferred,
        "precision": str(precision or "bf16").lower(),
        "runtime_precision": str(runtime_precision or precision or "bf16").lower(),
        "arch": str(arch or ""),
        "inputs": [],
        "source_operations": [],
        "constants": [],
        "signals": [],
        "unknowns": [],
    }
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        spec["unknowns"].append(f"source AST unavailable: {exc.msg}")
        return spec

    forward = _forward_function(tree)
    if forward is None:
        spec["unknowns"].append("forward signature not found")
    else:
        positional = [*forward.args.posonlyargs, *forward.args.args]
        spec["inputs"] = [arg.arg for arg in positional if arg.arg != "self"][:12]
        if forward.args.vararg:
            spec["signals"].append("variadic forward ABI")

    calls: list[str] = []
    constants: list[dict[str, Any]] = []
    walk_root: ast.AST = forward or tree
    for node in ast.walk(walk_root):
        if isinstance(node, ast.Call):
            name = _qualified_name(node.func)
            if name:
                calls.append(name)
            for keyword in node.keywords:
                literal = _literal(keyword.value)
                if keyword.arg in {"dim", "axis", "groups", "stride", "padding", "dilation"} and literal is not None:
                    constants.append({"name": keyword.arg, "value": literal, "line": node.lineno})
        elif isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in {0, 1, -1}:
                constants.append({"value": node.value, "line": node.lineno})

    spec["source_operations"] = _unique(calls, limit=12)
    seen_constants: list[dict[str, Any]] = []
    for item in constants:
        compact = {key: value for key, value in item.items() if key != "line"}
        if compact not in [{key: value for key, value in old.items() if key != "line"} for old in seen_constants]:
            seen_constants.append(item)
        if len(seen_constants) >= 8:
            break
    spec["constants"] = seen_constants

    lowered_calls = " ".join(calls).lower()
    if any(name in lowered_calls for name in ("reshape", "view", "permute", "transpose")):
        spec["signals"].append("layout-transforming source ops")
    if "contiguous" in lowered_calls:
        spec["signals"].append("source requests contiguous materialization")
    if inferred in {"attention", "reduction"}:
        spec["signals"].append("numerically sensitive reduction")
    if inferred in {"gemm", "attention", "convolution"}:
        spec["signals"].append("tensor-core candidate")
    if spec["precision"] == "fp8":
        spec["signals"].append("quantization and conversion cost are part of the contract")

    spec["signals"] = _unique(spec["signals"], limit=8)
    if not spec["inputs"]:
        spec["unknowns"].append("input names")
    spec["unknowns"].extend(("shape domain", "stride/alignment guarantees"))
    return spec


def compact_task_spec(spec: dict[str, Any]) -> list[str]:
    """Render only facts that help an author choose what to read next."""
    lines = [
        f"- operation: `{spec.get('operation', 'unknown')}`; target: "
        f"`{spec.get('precision', 'unknown')}` on `{spec.get('arch') or 'auto'}`",
    ]
    inputs = spec.get("inputs", [])
    if inputs:
        lines.append("- forward inputs observed: " + ", ".join(f"`{value}`" for value in inputs))
    operations = spec.get("source_operations", [])
    if operations:
        lines.append("- source operation signals: " + ", ".join(f"`{value}`" for value in operations[:8]))
    signals = spec.get("signals", [])
    if signals:
        lines.append("- routing signals: " + "; ".join(str(value) for value in signals[:5]))
    unknowns = spec.get("unknowns", [])
    if unknowns:
        lines.append(
            "- do not assume: " + ", ".join(str(value) for value in unknowns[:4]) + "; preserve the existing fallback"
        )
    return lines
