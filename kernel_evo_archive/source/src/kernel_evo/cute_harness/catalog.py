"""Version/architecture aware retrieval for compact CuTe DSL author context."""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from kernel_evo.cute_harness.capabilities import probe_capabilities, resolve_target_arch
from kernel_evo.cute_harness.correctness import build_correctness_contract
from kernel_evo.cute_harness.experiments import compact_experiment_lessons, query_experiments
from kernel_evo.cute_harness.lint import compact_lint_findings, lint_cute_source
from kernel_evo.cute_harness.paths import resource_path
from kernel_evo.cute_harness.task_spec import compact_task_spec, extract_task_spec, infer_operation


@dataclass(frozen=True, slots=True)
class CatalogEntry:
    id: str
    title: str
    kind: str
    path: Path
    dialect: str
    versions: tuple[str, ...]
    arches: tuple[str, ...]
    precisions: tuple[str, ...]
    operations: tuple[str, ...]
    concepts: tuple[str, ...]
    files: tuple[Path, ...]
    deep_files: tuple[Path, ...]
    why: str = ""
    use_when: tuple[str, ...] = ()
    symbols: tuple[str, ...] = ()
    always: bool = False
    priority: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "path": str(self.path),
            "dialect": self.dialect,
            "versions": list(self.versions),
            "arches": list(self.arches),
            "precisions": list(self.precisions),
            "operations": list(self.operations),
            "concepts": list(self.concepts),
            "files": [str(path) for path in self.files],
            "deep_files": [str(path) for path in self.deep_files],
            "why": self.why,
            "use_when": list(self.use_when),
            "symbols": list(self.symbols),
        }


@dataclass(frozen=True, slots=True)
class AgentContextBundle:
    text: str
    readable_files: tuple[Path, ...]
    entries: tuple[CatalogEntry, ...]
    metadata: dict[str, Any]


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


@lru_cache(maxsize=1)
def load_catalog() -> tuple[CatalogEntry, ...]:
    payload = yaml.safe_load(resource_path("catalog.yaml").read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping) or payload.get("dialect") != "cute_dsl_python":
        raise ValueError("CuTe harness catalog must declare dialect: cute_dsl_python")
    entries: list[CatalogEntry] = []
    for raw in payload.get("entries", []):
        if not isinstance(raw, Mapping):
            continue
        primary = resource_path(str(raw["path"]))
        files = tuple(resource_path(str(item)) for item in raw.get("files", []))
        deep_files = tuple(resource_path(str(item)) for item in raw.get("deep_files", []))
        entries.append(
            CatalogEntry(
                id=str(raw["id"]),
                title=str(raw.get("title", raw["id"])),
                kind=str(raw.get("kind", "semantic")),
                path=primary,
                dialect=str(raw.get("dialect", payload["dialect"])),
                versions=_strings(raw.get("versions", ())),
                arches=_strings(raw.get("arches", ("any",))),
                precisions=_strings(raw.get("precisions", ("any",))),
                operations=_strings(raw.get("operations", ("any",))),
                concepts=_strings(raw.get("concepts", ())),
                files=files,
                deep_files=deep_files,
                why=str(raw.get("why", "")).strip(),
                use_when=tuple(str(item).strip() for item in raw.get("use_when", ()) if str(item).strip()),
                symbols=tuple(str(item).strip() for item in raw.get("symbols", ()) if str(item).strip()),
                always=bool(raw.get("always", False)),
                priority=int(raw.get("priority", 0)),
            )
        )
    return tuple(entries)


def _matches(value: str, allowed: tuple[str, ...]) -> bool:
    return not allowed or "any" in allowed or value in allowed


def _version_matches(version: str, allowed: tuple[str, ...]) -> bool:
    return not allowed or any(version == item or version.startswith(f"{item}.") for item in allowed)


def _query_tokens(values: Iterable[str]) -> set[str]:
    return {
        token
        for value in values
        for token in re.findall(r"[a-z0-9_]+", str(value).lower())
        if len(token) > 1
    }


def search_catalog(
    *,
    precision: str = "bf16",
    arch: str = "sm_90a",
    operation: str = "any",
    concepts: Sequence[str] = (),
    query: str = "",
    version: str = "",
    limit: int = 8,
) -> list[CatalogEntry]:
    if not version:
        try:
            version = importlib.metadata.version("nvidia-cutlass-dsl")
        except importlib.metadata.PackageNotFoundError:
            version = ""
    precision = str(precision or "bf16").lower()
    arch = str(arch or "").lower()
    operation = str(operation or "any").lower()
    wanted = _query_tokens((*concepts, query, operation, precision))
    ranked: list[tuple[int, str, CatalogEntry]] = []
    for entry in load_catalog():
        if entry.dialect != "cute_dsl_python":
            continue
        if version not in {"", "not-installed"} and not _version_matches(version, entry.versions):
            continue
        if arch and not _matches(arch, entry.arches):
            continue
        if not _matches(precision, entry.precisions):
            continue
        if operation != "any" and not _matches(operation, entry.operations):
            continue
        score = entry.priority + (1000 if entry.always else 0)
        score += 80 if precision in entry.precisions else 0
        score += 40 if arch in entry.arches else 0
        score += 120 if operation in entry.operations else 0
        score += 25 * len(wanted.intersection(_query_tokens(entry.concepts)))
        ranked.append((score, entry.id, entry))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [entry for _, _, entry in ranked[: max(1, int(limit))]]


def _requested_concepts(idea: Mapping[str, str], source: str, precision: str) -> tuple[str, ...]:
    text = " ".join((str(idea.get("id", "")), str(idea.get("summary", "")), source[:20_000])).lower()
    concepts = {"layout", "predication", "compile-cache"}
    mapping = {
        "copy": "tiled-copy",
        "vector": "vectorization",
        "layout": "layout",
        "tile": "tiling",
        "gemm": "wgmma",
        "matmul": "wgmma",
        "einsum": "wgmma",
        "pipeline": "pipeline",
        "tma": "tma",
        "cluster": "cluster",
        "occupancy": "occupancy",
        "register": "register-pressure",
        "shared": "smem",
        "fuse": "fusion",
    }
    concepts.update(value for needle, value in mapping.items() if needle in text)
    concepts.add("fp8" if precision == "fp8" else "bf16")
    return tuple(sorted(concepts))


def _selection_reason(
    entry: CatalogEntry,
    *,
    precision: str,
    arch: str,
    operation: str,
    concepts: Sequence[str],
) -> str:
    reasons: list[str] = []
    if entry.always:
        reasons.append("foundational route supplied for every compatible task")
    if precision in entry.precisions:
        reasons.append(f"matches {precision}")
    if arch in entry.arches:
        reasons.append(f"matches {arch}")
    if operation in entry.operations:
        reasons.append(f"matches {operation}")
    overlap = sorted(set(concepts).intersection(entry.concepts))
    if overlap:
        reasons.append("covers " + ", ".join(overlap[:4]))
    return "; ".join(reasons) or "closest compatible verified entry"


def _wants_deep_reference(entry: CatalogEntry, *, concepts: Sequence[str], operation: str) -> bool:
    if not entry.deep_files:
        return False
    deep_concepts = {
        "wgmma",
        "tma",
        "pipeline",
        "mbarrier",
        "cluster",
        "smem",
        "register-pressure",
        "fusion",
        "tiled-copy",
        "vectorization",
        "predication",
        "specialization",
    }
    matching = set(concepts).intersection(entry.concepts)
    return operation in entry.operations and bool(deep_concepts.intersection(matching))


def build_agent_context(
    *,
    config: Mapping[str, Any],
    idea: Mapping[str, str],
    baseline_path: str | Path | None = None,
    task_source: str = "",
    experiment_database: str | Path | None = None,
) -> AgentContextBundle:
    precision = str(config.get("precision", "bf16") or "bf16").lower()
    device_text = str(config.get("device", "cuda:0"))
    try:
        device = int(device_text.rsplit(":", 1)[-1]) if ":" in device_text else 0
    except ValueError:
        device = 0
    arch = resolve_target_arch(
        explicit_arch=str(config.get("cute_arch", "")),
        arch_list=str(config.get("arch_list", "")),
        device=device,
    )
    report = probe_capabilities(
        device=device,
        explicit_arch=str(config.get("cute_arch", "")),
        arch_list=str(config.get("arch_list", "")),
    )
    source = str(task_source or "")
    if baseline_path:
        try:
            baseline_source = Path(baseline_path).read_text(encoding="utf-8", errors="replace")[:60_000]
            if not source:
                source = baseline_source
        except OSError:
            baseline_source = ""
    else:
        baseline_source = source
    operation = str(config.get("cute_operation", "") or infer_operation(source)).lower()
    concepts = _requested_concepts(idea, source, precision)
    limit = int(config.get("cute_context_cards", 7) or 7)
    entries = search_catalog(
        precision=precision,
        arch=arch or "sm_90a",
        operation=operation,
        concepts=concepts,
        query=f"{idea.get('id', '')} {idea.get('summary', '')}",
        version=str(report["packages"]["nvidia-cutlass-dsl"]),
        limit=limit,
    )
    readable: list[Path] = []
    deep_candidates: list[tuple[int, str, Path]] = []
    idea_tokens = _query_tokens((str(idea.get("id", "")), str(idea.get("summary", ""))))
    for entry in entries:
        for path in (entry.path, *entry.files):
            if path not in readable:
                readable.append(path)
        if _wants_deep_reference(entry, concepts=concepts, operation=operation):
            entry_tokens = _query_tokens((entry.title, entry.why, *entry.use_when, *entry.concepts))
            deep_score = 10 * len(idea_tokens.intersection(entry_tokens))
            deep_score += len(set(concepts).intersection(entry.concepts))
            for path in entry.deep_files:
                deep_candidates.append((deep_score, entry.id, path))
    deep_limit = max(0, int(config.get("cute_context_deep_files", 1) or 0))
    deep_candidates.sort(key=lambda item: (-item[0], item[1], str(item[2])))
    deep_readable = list(dict.fromkeys(path for _, _, path in deep_candidates))[:deep_limit]
    if str(idea.get("codegen_contract", "")) == "hopper_wgmma":
        verified_kernel = resource_path("examples/hopper_wgmma_gemm/kernel.py")
        if verified_kernel not in deep_readable:
            deep_readable.append(verified_kernel)
    readable.extend(path for path in deep_readable if path not in readable)

    features = report.get("features", {})
    enabled = ", ".join(sorted(name for name, value in features.items() if value)) or "none detected"
    issues = list(report.get("issues", []))
    if not entries:
        issues.append(
            "No verified catalog lane matches this installed DSL/version/architecture. "
            "Use exact API lookup and do not copy a 4.2 construction into a different release."
        )
    task_spec = extract_task_spec(
        source,
        operation=operation,
        precision=precision,
        runtime_precision=str(config.get("runtime_precision", "") or precision),
        arch=arch,
    )
    correctness_contract = build_correctness_contract(
        operation=operation,
        precision=precision,
    )
    lint_report = (
        lint_cute_source(
            baseline_source,
            precision=precision,
            arch=arch or "sm_90a",
            operation=operation,
        )
        if baseline_source.strip()
        else {"issues": [], "counts": {"error": 0, "warning": 0, "info": 0}}
    )
    lines = [
        "# Retrieved Python CuTe DSL harness context",
        "",
        "This bundle is restricted to `nvidia-cutlass-dsl` / `cutlass.cute` Python. "
        "Do not use CuTe C++ or the legacy CUTLASS Python operation API.",
        "",
        "## Exact contract",
        "",
        f"- installed DSL: `{report['packages']['nvidia-cutlass-dsl']}`",
        f"- target architecture: `{arch or '<not detected>'}`",
        f"- requested precision: `{precision}`; runtime precision: "
        f"`{config.get('runtime_precision', '') or precision}`",
        f"- inferred operation family: `{operation}`",
        f"- detected low-level features: {enabled}",
        "",
        "## Task route",
        "",
        *compact_task_spec(task_spec),
        "- evaluator coverage route: "
        + ", ".join(f"`{item['id']}`" for item in correctness_contract["cases"][:5]),
    ]
    if issues:
        lines.extend(("", "Warnings:", *[f"- {item}" for item in issues]))
    if precision == "fp8":
        lines.extend(
            (
                "",
                "## FP8 validity rule",
                "",
                "A float8 cast is not an FP8 acceleration. A genuine Hopper FP8 fast path uses "
                "`MmaF8Op`/WGMMA on `sm_90a`, normally with TMA-fed shared-memory operands and "
                "Float32 accumulation. Keep the public module ABI in the configured runtime precision "
                "and account for conversion/prepacking cost.",
            )
        )
    elif precision == "bf16":
        lines.extend(
            (
                "",
                "## BF16 default",
                "",
                "Use `MmaF16BF16Op(cutlass.BFloat16, cutlass.Float32, ...)` for GEMM-like work. "
                "Preserve BF16 storage/I/O, use Float32 accumulators, and fuse conversion/epilogue "
                "work before the final BF16 store.",
            )
        )
    lint_lines = compact_lint_findings(lint_report, limit=3)
    if lint_lines:
        lines.extend(("", "### Baseline signals", "", *lint_lines))
    lines.extend(
        (
            "",
            "## Navigation",
            "",
            "Read the first item that answers the current decision. Do not read every card or the full "
            "reference kernel by default.",
            "",
        )
    )
    for index, entry in enumerate(entries, 1):
        reason = _selection_reason(
            entry,
            precision=precision,
            arch=arch or "sm_90a",
            operation=operation,
            concepts=concepts,
        )
        when = entry.use_when[0] if entry.use_when else "the current hypothesis touches this concept"
        why = entry.why or entry.title
        lines.extend(
            (
                f"{index}. `{entry.id}` — **use when:** {when}",
                f"   **why now:** {reason}. **what it establishes:** {why}",
                f"   **read first:** `{entry.path}`",
            )
        )
        if entry.symbols:
            lines.append(
                "   **verify APIs before editing:** " + ", ".join(f"`{symbol}`" for symbol in entry.symbols[:5])
            )
    if deep_readable:
        lines.extend(("", "### Deep reference — open only for the matching construction", ""))
        for path in deep_readable:
            owner = next((entry for entry in entries if path in entry.deep_files), None)
            reason = owner.use_when[-1] if owner and owner.use_when else "copying the complete verified construction"
            reason = reason.rstrip(".")
            if reason:
                reason = reason[0].lower() + reason[1:]
            lines.append(f"- `{path}` — use only for {reason}; preserve its pipeline invariants as a unit.")

    lesson_limit = max(0, int(config.get("cute_context_lessons", 3) or 0))
    if experiment_database and lesson_limit:
        records = query_experiments(
            experiment_database,
            task=operation,
            tag=precision,
            limit=max(lesson_limit * 2, lesson_limit),
        )
        lesson_lines = compact_experiment_lessons(records, limit=lesson_limit)
        if lesson_lines:
            lines.extend(("", "## Prior local evidence", "", *lesson_lines))
    lines.extend(
        (
            "",
            "Use only the readable files listed in this island packet. KernelEvo owns correctness, "
            "sanitizers, benchmarking, profiling, and promotion after the authoring barrier.",
        )
    )
    max_chars = int(config.get("cute_context_max_chars", 10_000) or 10_000)
    text = "\n".join(lines).rstrip() + "\n"
    if len(text) > max_chars:
        suffix = "\n\n[Navigation stopped at the context budget; use `kernel-evo cute search` for another route.]\n"
        cutoff = max(0, max_chars - len(suffix))
        boundary = text.rfind("\n\n", 0, cutoff)
        text = text[: boundary if boundary > 0 else cutoff].rstrip() + suffix
    return AgentContextBundle(
        text=text,
        readable_files=tuple(readable),
        entries=tuple(entries),
        metadata={
            "dialect": "cute_dsl_python",
            "version": report["packages"]["nvidia-cutlass-dsl"],
            "arch": arch,
            "precision": precision,
            "operation": operation,
            "concepts": list(concepts),
            "task_spec": task_spec,
            "correctness_contract": correctness_contract,
            "baseline_lint": lint_report,
            "entry_reasons": {
                entry.id: _selection_reason(
                    entry,
                    precision=precision,
                    arch=arch or "sm_90a",
                    operation=operation,
                    concepts=concepts,
                )
                for entry in entries
            },
            "deep_files": [str(path) for path in deep_readable],
            "codegen_contracts": [
                str(path)
                for entry in entries
                for path in (*entry.files,)
                if path.name == "expected_codegen.yaml"
            ],
            "idea_id": str(idea.get("id", "")),
            "idea_codegen_contract": str(idea.get("codegen_contract", "")),
            "idea_requires_capability": str(idea.get("requires_capability", "")),
            "idea_produces_capability": str(idea.get("produces_capability", "")),
            "capability_fingerprint": report.get("fingerprint", ""),
        },
    )
