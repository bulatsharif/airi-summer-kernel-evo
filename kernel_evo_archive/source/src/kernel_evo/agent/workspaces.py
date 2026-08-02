"""Filesystem isolation and compact context helpers for island workspaces."""

from __future__ import annotations

import ast
from pathlib import Path

from kernel_evo.agent.errors import ConfigurationError


SUPPORTED_BASELINE_SUFFIXES = (".py", ".cu", ".cpp", ".cc", ".cuh", ".triton")


def resolve_baseline(path_value: str | Path, *, candidate_name: str = "") -> Path:
    path = Path(path_value).expanduser().resolve()
    if path.is_file():
        return path
    if not path.exists():
        raise ConfigurationError(f"Baseline not found: {path}")
    if not path.is_dir():
        raise ConfigurationError(f"Baseline must be a file or directory: {path}")

    if candidate_name:
        named_candidate = path / candidate_name
        if named_candidate.is_file():
            return named_candidate.resolve()
        raise ConfigurationError(
            f"candidate_name `{candidate_name}` was not found in baseline directory: {path}"
        )

    preferred = ("kernel.py", "candidate.py", "seed.py", "kernel.cu")
    for filename in preferred:
        candidate = path / filename
        if candidate.is_file():
            return candidate.resolve()
    candidates = sorted(
        candidate
        for candidate in path.iterdir()
        if candidate.is_file() and candidate.suffix.lower() in SUPPORTED_BASELINE_SUFFIXES
    )
    if len(candidates) == 1:
        return candidates[0].resolve()
    if not candidates:
        raise ConfigurationError(f"No candidate source file found in baseline directory: {path}")
    raise ConfigurationError(
        f"Baseline directory is ambiguous ({len(candidates)} source files); set baseline to one file."
    )


def summarize_tests(tests_value: str) -> str:
    if not tests_value:
        return (
            "KernelEvo owns correctness and performance evaluation. "
            "The author must preserve the baseline interface and must not run the full benchmark.\n"
        )
    path = Path(tests_value).expanduser().resolve()
    if not path.exists():
        return f"Configured tests path is currently unavailable: `{path}`.\n"
    if path.is_file():
        return _file_summary(path)

    files = sorted(
        candidate.relative_to(path)
        for candidate in path.rglob("*")
        if candidate.is_file()
        and "__pycache__" not in candidate.parts
        and candidate.suffix.lower() not in {".pyc", ".pyo"}
    )
    visible = files[:20]
    lines = [f"Tests are owned by KernelEvo at `{path}`.", "Representative files:"]
    lines.extend(f"- `{item}`" for item in visible)
    if len(files) > len(visible):
        lines.append(f"- … and {len(files) - len(visible)} more files")
    test_names: list[str] = []
    for relative in files[:40]:
        candidate = path / relative
        if candidate.suffix != ".py":
            continue
        try:
            tree = ast.parse(candidate.read_text(encoding="utf-8", errors="replace"))
        except (OSError, SyntaxError):
            continue
        test_names.extend(
            node.name
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name.startswith("test_")
        )
    if test_names:
        lines.append("Covered contracts:")
        lines.extend(f"- `{name}`" for name in test_names[:24])
    lines.append(
        "Do not edit tests or run the full benchmark. Before submission, run Python syntax "
        "compilation and `kernel-evo cute lint` on the candidate, passing the idea's explicit "
        "`--contract` when one is configured. When TASK.md provides a bounded compile/execute "
        "check, run that exact command as well; it is not a benchmark."
    )
    return "\n".join(lines) + "\n"


def backend_rules(backend: str, custom_rules_file: str = "") -> str:
    common = [
        "- Preserve `ModelNew` and the baseline call interface.",
        "- Implement exactly one bounded optimization idea.",
        "- Keep the candidate self-contained; do not edit tests, baseline, state, or other islands.",
        "- Do not run the full correctness/performance harness. KernelEvo evaluates after the barrier.",
        "- Do not replace requested-precision compute with Python-side float32 promotion.",
    ]
    specific = {
        "triton": [
            "- Launch at least one real `@triton.jit` kernel whose output contributes to the result.",
            "- Mask non-divisible boundaries and keep pointer arithmetic shape-safe.",
        ],
        "cuda_inline": [
            "- Compile and launch a real CUDA kernel; dead or unused inline code is invalid.",
            "- Keep device, dtype, stream, and boundary handling consistent with the baseline.",
        ],
        "cute": [
            "- Use only the Python CuTe DSL (`nvidia-cutlass-dsl` / `cutlass.cute`), "
            "never CuTe C++ or the legacy CUTLASS Python API.",
            "- Use `@cute.kernel`, launch from `@cute.jit`, and cache `cute.compile(...)` output.",
            "- Use tiled copy atoms/fragments and identity-tensor predication; avoid scalar runtime indexing.",
            "- Detach tensors before `from_dlpack` and use `mark_layout_dynamic()` where shape reuse matters.",
            "- Before submission, run Python syntax compilation and `kernel-evo cute lint`, "
            "passing the idea's explicit `--contract` when one is configured. If TASK.md provides "
            "a bounded compile/execute check, run that exact command too; do not run correctness, "
            "profiling, or benchmark modes.",
        ],
    }
    lines = ["# Authoring rules", "", *common, *specific.get(backend, [])]
    if custom_rules_file:
        custom_path = Path(custom_rules_file).expanduser().resolve()
        if not custom_path.is_file():
            raise ConfigurationError(f"Rules file not found: {custom_path}")
        custom = custom_path.read_text(encoding="utf-8")
        if len(custom) > 12_000:
            custom = custom[:12_000].rstrip() + "\n\n[custom rules truncated by KernelEvo]\n"
        lines.extend(["", "# Project-specific rules", "", custom.rstrip()])
    return "\n".join(lines).rstrip() + "\n"


def _file_summary(path: Path) -> str:
    if path.suffix.lower() in {".md", ".txt"}:
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > 4_000:
            text = text[:4_000].rstrip() + "\n\n[summary truncated by KernelEvo]\n"
        return f"Tests are owned by KernelEvo at `{path}`.\n\n{text.rstrip()}\n"
    return (
        f"Tests are owned by KernelEvo at `{path}`. Preserve the baseline interface; "
        "do not edit or run the full benchmark during authoring.\n"
    )
