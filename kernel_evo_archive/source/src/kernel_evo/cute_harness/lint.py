"""Fast, non-executing checks for common Python CuTe DSL candidate mistakes."""

from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class LintIssue:
    code: str
    severity: str
    message: str
    action: str
    line: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _decorator_name(node: ast.AST) -> str:
    if isinstance(node, ast.Call):
        return _decorator_name(node.func)
    if isinstance(node, ast.Attribute):
        prefix = _decorator_name(node.value)
        return f"{prefix}.{node.attr}" if prefix else node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _call_name(node: ast.Call) -> str:
    return _decorator_name(node.func)


def candidate_kernel_delta(candidate_source: str, baseline_source: str) -> dict[str, Any]:
    """Describe candidate-local CuTe kernel bodies added or changed from a parent."""

    candidate = _kernel_fingerprints(candidate_source)
    baseline = _kernel_fingerprints(baseline_source)
    added = sorted(set(candidate).difference(baseline))
    modified = sorted(
        name for name in set(candidate).intersection(baseline) if candidate[name] != baseline[name]
    )
    return {
        "candidate_kernels": sorted(candidate),
        "baseline_kernels": sorted(baseline),
        "added": added,
        "modified": modified,
        "changed": [*added, *modified],
    }


def _kernel_fingerprints(source: str) -> dict[str, str]:
    try:
        tree = ast.parse(str(source or ""))
    except SyntaxError:
        return {}
    fingerprints: dict[str, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not any(_decorator_name(item).endswith("cute.kernel") for item in node.decorator_list):
            continue
        normalized = ast.dump(node, annotate_fields=True, include_attributes=False)
        fingerprints[node.name] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return fingerprints


def _forward_nodes(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    result: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name not in {"ModelNew", "_CuteSeedMixin"}:
            continue
        result.extend(
            item
            for item in node.body
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
            and item.name in {"forward", "_cute_apply"}
        )
    return result


def _guarded_compile_calls(function: ast.AST) -> set[int]:
    guarded: set[int] = set()
    for node in ast.walk(function):
        if not isinstance(node, ast.If) or "None" not in ast.unparse(node.test):
            continue
        for child in node.body:
            for item in ast.walk(child):
                if isinstance(item, ast.Call) and _call_name(item) in {"cute.compile", "cutlass.cute.compile"}:
                    guarded.add(id(item))
    return guarded


def _issue(
    code: str,
    severity: str,
    message: str,
    action: str,
    node: ast.AST | None = None,
) -> LintIssue:
    return LintIssue(code, severity, message, action, getattr(node, "lineno", None))


def lint_cute_source(
    source: str,
    *,
    precision: str = "bf16",
    arch: str = "sm_90a",
    operation: str = "",
    codegen_contract: str = "",
) -> dict[str, Any]:
    """Return bounded compliance/performance findings without compiling the candidate.

    Findings are evidence, not promotion decisions. KernelEvo's evaluator remains the
    correctness and performance authority.
    """
    text = str(source or "")
    issues: list[LintIssue] = []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return {
            "schema_version": 1,
            "dialect": "cute_dsl_python",
            "parseable": False,
            "issues": [
                _issue(
                    "python-syntax",
                    "error",
                    exc.msg,
                    "Repair Python syntax before invoking CuTe compilation.",
                    exc,
                ).to_dict()
            ],
            "counts": {"error": 1, "warning": 0, "info": 0},
        }

    functions = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    kernel_functions = {
        node.name: node
        for node in functions
        if any(_decorator_name(item).endswith("cute.kernel") for item in node.decorator_list)
    }
    jit_functions = {
        node.name: node
        for node in functions
        if any(_decorator_name(item).endswith("cute.jit") for item in node.decorator_list)
    }
    calls = [(node, _call_name(node)) for node in ast.walk(tree) if isinstance(node, ast.Call)]
    call_names = [name for _, name in calls]
    compiled_targets: dict[str, ast.AST] = {}
    compiled_aliases: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Call) or _call_name(value) not in {"cute.compile", "cutlass.cute.compile"}:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            name = _decorator_name(target)
            if name:
                short_name = name.rsplit(".", 1)[-1]
                compiled_targets[short_name] = node
                compiled_aliases.setdefault(short_name, set())
    # Follow the common compile-local -> cached-instance-attribute handoff.
    # This is intentionally bounded dataflow, but avoids declaring a proven
    # cached executor dead merely because its call uses ``self.<name>``.
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = node.value
        if not isinstance(value, ast.Name) or value.id not in compiled_targets:
            continue
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            alias = _decorator_name(target)
            if alias:
                compiled_aliases[value.id].add(alias.rsplit(".", 1)[-1])

    if not re.search(r"(^|\n)\s*import\s+cutlass(?:\s|$|,)", text):
        issues.append(
            _issue(
                "missing-cutlass-import",
                "error",
                "The candidate does not import `cutlass`.",
                "Import `cutlass` and use its exact DSL scalar types.",
            )
        )
    if "cutlass.cute" not in text:
        issues.append(
            _issue(
                "missing-cute-import",
                "error",
                "The candidate does not import the Python CuTe DSL.",
                "Use `import cutlass.cute as cute`; do not substitute CuTe C++ or the legacy API.",
            )
        )
    if not kernel_functions:
        issues.append(
            _issue(
                "missing-kernel",
                "error",
                "No `@cute.kernel` function was found.",
                "Define a Python CuTe DSL kernel that contributes to the returned output.",
            )
        )
    if not jit_functions:
        issues.append(
            _issue(
                "missing-jit-launch",
                "error",
                "No `@cute.jit` launch function was found.",
                "Launch the kernel from a host JIT function with explicit grid and block shapes.",
            )
        )
    if not any(name.endswith("cute.compile") or name == "cute.compile" for name in call_names):
        issues.append(
            _issue(
                "missing-compile",
                "error",
                "No `cute.compile(...)` call was found.",
                "Compile once for a complete static cache key and reuse the executor.",
            )
        )
    for target, node in compiled_targets.items():
        names = {target, *compiled_aliases.get(target, set())}
        uses = [
            name
            for name in call_names
            if any(name == value or name.endswith(f".{value}") for value in names)
        ]
        if not uses:
            issues.append(
                _issue(
                    "compiled-executor-unused",
                    "error",
                    f"Compiled executor `{target}` is assigned but never called.",
                    "Call the cached executor from the forward path and return data produced by it.",
                    node,
                )
            )
    if "from_dlpack" not in text:
        issues.append(
            _issue(
                "missing-dlpack",
                "error",
                "No Torch-to-CuTe DLPack conversion was found.",
                "Convert detached tensors with `from_dlpack(..., assumed_align=...)`.",
            )
        )

    for function in _forward_nodes(tree):
        guarded_compile_calls = _guarded_compile_calls(function)
        for node in ast.walk(function):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name.endswith("cute.compile") or name == "cute.compile":
                if id(node) in guarded_compile_calls:
                    continue
                issues.append(
                    _issue(
                        "compile-in-forward",
                        "warning",
                        "`cute.compile(...)` executes in the forward path.",
                        "Cache the compiled executor outside the hot path and key every static specialization.",
                        node,
                    )
                )
            if name.endswith(".contiguous"):
                issues.append(
                    _issue(
                        "forward-contiguous",
                        "warning",
                        "Forward performs a potentially materializing `.contiguous()` conversion.",
                        "Remove it, prepack immutable data, or include its cost in end-to-end timing.",
                        node,
                    )
                )

    for name, node in kernel_functions.items():
        call_pattern = re.compile(rf"\b{re.escape(name)}\s*\(")
        occurrences = len(call_pattern.findall(text))
        if occurrences <= 1:
            issues.append(
                _issue(
                    "unlaunched-kernel",
                    "error",
                    f"Kernel `{name}` appears to be defined but never called by a launch function.",
                    "Call it from `@cute.jit` and launch the resulting invocation.",
                    node,
                )
            )

    lowered_precision = str(precision or "bf16").lower()
    lowered_operation = str(operation or "").lower()
    contract = str(codegen_contract or "").strip().lower()
    matrix_like = lowered_operation in {"gemm", "attention", "convolution"}
    has_versioned_mma_helper = "make_trivial_tiled_mma" in text
    if lowered_precision == "fp8" and matrix_like and not (
        "MmaF8Op" in text or has_versioned_mma_helper
    ):
        issues.append(
            _issue(
                "fp8-without-wgmma",
                "warning",
                "The FP8 matrix candidate does not reference `MmaF8Op`.",
                "Use genuine FP8 WGMMA and require WGMMA in generated code; casts alone are not acceleration.",
            )
        )
    if contract == "hopper_wgmma":
        contract_markers = (
            ("MmaF8Op", "make_trivial_tiled_mma", "contract-without-wgmma-source", "a WGMMA atom"),
            ("make_tiled_tma_atom", "TmaLoad", "contract-without-tma-source", "a TMA load atom"),
            ("PipelineTmaAsync", "mbarrier", "contract-without-mbarrier-source", "a TMA/mbarrier pipeline"),
        )
        for primary, alternate, code, description in contract_markers:
            if primary not in text and alternate not in text:
                issues.append(
                    _issue(
                        code,
                        "error",
                        f"The `hopper_wgmma` contract requires source evidence for {description}.",
                        "Wire the required mechanism into the compiled production executor, "
                        "then verify the emitted artifact during evaluation.",
                    )
                )
    elif contract == "vector" and not any(
        marker in text
        for marker in ("num_bits_per_copy", "UniversalCopy", "vector_size", "copy_bits")
    ):
        issues.append(
            _issue(
                "contract-without-vector-source",
                "error",
                "The `vector` contract requires an explicit wide-copy mechanism in source.",
                "Use an explicit vector copy width and verify emitted LDG/STG width during evaluation.",
            )
        )
    if lowered_precision == "bf16" and matrix_like and not (
        "MmaF16BF16Op" in text or has_versioned_mma_helper
    ):
        issues.append(
            _issue(
                "bf16-without-wgmma",
                "info",
                "The BF16 matrix candidate does not reference the Hopper BF16 WGMMA atom.",
                "Use the WGMMA reference only when replacing the matrix mainloop is the current hypothesis.",
            )
        )
    hardcoded_sm90 = re.search(
        r"(?:CUTE_DSL_ARCH[^\n]{0,48}['\"]sm_90['\"]|gpu-architecture[^\n]{0,24}sm_90(?:\b|['\"]))",
        text,
        flags=re.IGNORECASE,
    )
    if str(arch or "") == "sm_90a" and hardcoded_sm90:
        issues.append(
            _issue(
                "arch-feature-mismatch",
                "error",
                "Source hard-codes `sm_90`, which omits Hopper architecture-specific features.",
                "Use the evaluator-provided `sm_90a` target for TMA/WGMMA.",
            )
        )
    if kernel_functions and "elem_less" not in text and "predicate" not in text.lower():
        issues.append(
            _issue(
                "tail-proof-not-visible",
                "info",
                "No obvious boundary predicate was found.",
                "If shapes can be ragged, prove a residue path with identity coordinates and `cute.elem_less`.",
            )
        )
    if (
        "make_copy_atom" in text
        and ("copy_bits" in text or "vector_size" in text)
        and "num_bits_per_copy" not in text
    ):
        issues.append(
            _issue(
                "implicit-copy-width",
                "warning",
                "The TV layout suggests a wide copy, but the copy atom does not state `num_bits_per_copy`.",
                "Prove alignment, request the bit width explicitly, and verify the emitted LDG/STG width.",
            )
        )
    if "mark_layout_dynamic()" in text and matrix_like:
        issues.append(
            _issue(
                "broad-dynamic-layout",
                "info",
                "Matrix operands use the broad default dynamic-layout marking.",
                "Preserve the intended leading mode and cache all remaining static layout/tile properties.",
            )
        )

    order = {"error": 0, "warning": 1, "info": 2}
    issues.sort(key=lambda item: (order.get(item.severity, 9), item.line or 0, item.code))
    counts = {severity: sum(item.severity == severity for item in issues) for severity in order}
    return {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "parseable": True,
        "kernels": sorted(kernel_functions),
        "jit_functions": sorted(jit_functions),
        "issues": [item.to_dict() for item in issues],
        "counts": counts,
    }


def compact_lint_findings(report: dict[str, Any], *, limit: int = 4) -> list[str]:
    """Return only actionable findings suitable for a small author packet."""
    findings: list[str] = []
    for issue in report.get("issues", []):
        if not isinstance(issue, dict):
            continue
        if issue.get("severity") not in {"error", "warning"}:
            continue
        where = f" line {issue['line']}" if issue.get("line") else ""
        findings.append(
            f"- `{issue.get('code', 'finding')}`{where}: {issue.get('message', '')} "
            f"Next: {issue.get('action', '')}"
        )
        if len(findings) >= max(0, int(limit)):
            break
    return findings
