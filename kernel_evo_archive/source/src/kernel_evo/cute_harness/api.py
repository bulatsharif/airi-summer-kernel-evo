"""Exact installed-symbol lookup for the Python CuTe DSL."""

from __future__ import annotations

import importlib
import importlib.metadata
import inspect
import re
from pathlib import Path
from typing import Any

from kernel_evo.cute_harness.paths import HARNESS_ROOT


_SYMBOL_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$")
_ALIASES = {
    "cute": "cutlass.cute",
    "cutlass": "cutlass",
    "pipeline": "cutlass.pipeline",
    "warpgroup": "cutlass.cute.nvgpu.warpgroup",
    "cpasync": "cutlass.cute.nvgpu.cpasync",
    "sm90_utils": "cutlass.utils.hopper_helpers",
}


def _resolve_qualified(symbol: str) -> tuple[str, Any]:
    if not _SYMBOL_RE.fullmatch(symbol):
        raise ValueError(f"Invalid Python symbol: {symbol!r}")
    parts = symbol.split(".")
    module_name = _ALIASES.get(parts[0], parts[0])
    consumed = 1
    if parts[0] == "cutlass":
        module_name = "cutlass"
    module = importlib.import_module(module_name)
    value: Any = module
    for part in parts[consumed:]:
        value = getattr(value, part)
    canonical_prefix = module_name if parts[0] not in _ALIASES else module_name
    canonical = canonical_prefix + ("." + ".".join(parts[consumed:]) if len(parts) > consumed else "")
    return canonical, value


def resolve_symbol(symbol: str) -> tuple[str, Any]:
    if "." in symbol or symbol.split(".")[0] in _ALIASES:
        return _resolve_qualified(symbol)
    matches: list[tuple[str, Any]] = []
    for alias in ("cute", "warpgroup", "cpasync", "pipeline", "cutlass"):
        try:
            matches.append(_resolve_qualified(f"{alias}.{symbol}"))
        except (AttributeError, ImportError):
            continue
    unique = {name: value for name, value in matches}
    if not unique:
        raise AttributeError(f"Python CuTe DSL symbol not found: {symbol}")
    if len(unique) > 1:
        raise ValueError(f"Ambiguous symbol {symbol!r}; use one of: {', '.join(sorted(unique))}")
    return next(iter(unique.items()))


def _source_roots(value: Any) -> tuple[Path, ...]:
    roots: list[Path] = []
    source = inspect.getsourcefile(value)
    if source:
        path = Path(source).resolve()
        package_root = next((parent for parent in path.parents if parent.name == "cutlass"), path.parent)
        roots.append(package_root)
    roots.append(HARNESS_ROOT / "examples")
    return tuple(dict.fromkeys(roots))


def _local_usages(value: Any, canonical: str, *, limit: int) -> list[dict[str, Any]]:
    needle_candidates = {canonical, canonical.rsplit(".", 1)[-1]}
    usages: list[dict[str, Any]] = []
    for root in _source_roots(value):
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except OSError:
                continue
            for number, line in enumerate(lines, 1):
                has_symbol = any(needle in line for needle in needle_candidates)
                if has_symbol and not line.lstrip().startswith(("def ", "class ")):
                    usages.append({"path": str(path), "line": number, "text": line.strip()[:300]})
                    if len(usages) >= limit:
                        return usages
    return usages


def lookup_api(symbol: str, *, max_usages: int = 3, max_doc_chars: int = 4_000) -> dict[str, Any]:
    canonical, value = resolve_symbol(symbol)
    try:
        signature = str(inspect.signature(value))
    except (TypeError, ValueError):
        signature = ""
    source_file = inspect.getsourcefile(value) or ""
    try:
        _, source_line = inspect.getsourcelines(value)
    except (OSError, TypeError):
        source_line = None
    doc = inspect.getdoc(value) or ""
    deprecated = getattr(value, "__deprecated__", "")
    try:
        version = importlib.metadata.version("nvidia-cutlass-dsl")
    except importlib.metadata.PackageNotFoundError:
        version = "not-installed"
    return {
        "dialect": "cute_dsl_python",
        "cutlass_version": version,
        "requested_symbol": symbol,
        "canonical_symbol": canonical,
        "kind": type(value).__name__,
        "signature": signature,
        "source": {"path": source_file, "line": source_line},
        "docstring": doc[:max_doc_chars],
        "deprecated": str(deprecated or ""),
        "local_usages": _local_usages(value, canonical, limit=max(0, max_usages)),
    }
