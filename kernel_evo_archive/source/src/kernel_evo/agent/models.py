"""Small public value objects for direct agent integration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping


class RunPhase(StrEnum):
    READY = "ready"
    AUTHORING = "authoring"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"
    COMPLETE = "complete"


class IslandStatus(StrEnum):
    AWAITING_AUTHOR = "awaiting_author"
    SUBMITTED = "submitted"
    EVALUATING = "evaluating"
    EVALUATED = "evaluated"


@dataclass(frozen=True, slots=True)
class AuthoringTask:
    run_id: str
    backend: str
    iteration: int
    island: int
    task_file: Path
    candidate_path: Path
    editable_files: tuple[Path, ...]
    readable_files: tuple[Path, ...]
    idea_id: str
    idea_summary: str
    # Set when documentation is delivered by prompt instead of as readable files.
    # The coordinator prepends this file's text to the authoring session prompt.
    prompt_context_file: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "role": "kernel-author",
            "backend": self.backend,
            "iteration": self.iteration,
            "island": self.island,
            "task_file": str(self.task_file),
            "candidate_path": str(self.candidate_path),
            "editable_files": [str(path) for path in self.editable_files],
            "readable_files": [str(path) for path in self.readable_files],
            "prompt_context_file": (
                str(self.prompt_context_file) if self.prompt_context_file else ""
            ),
            "idea": {"id": self.idea_id, "summary": self.idea_summary},
        }


@dataclass(slots=True)
class EvaluationContext:
    run_id: str
    iteration: int
    island: int
    run_dir: Path
    island_dir: Path
    candidate_path: Path
    baseline_path: Path
    problem_dir: Path | None
    config: Mapping[str, Any]
    run_config: Mapping[str, Any]


@dataclass(slots=True)
class EvaluationResult:
    compiled: bool = False
    correctness: bool = False
    valid: bool = False
    runtime_us: float | None = None
    ref_runtime_us: float | None = None
    speedup: float = 0.0
    fitness: float = 0.0
    status: str = "failed"
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_metrics(cls, metrics: Mapping[str, Any]) -> "EvaluationResult":
        compiled = _as_bool(metrics.get("compiled", False))
        correctness = _as_bool(metrics.get("correctness", False))
        valid = _as_bool(metrics.get("is_valid", metrics.get("valid", compiled and correctness)))
        runtime_us = _positive_float(metrics.get("runtime_us"))
        ref_runtime_us = _positive_float(metrics.get("ref_runtime_us"))
        speedup = _as_float(metrics.get("speedup", metrics.get("fitness", 0.0)))
        fitness = _as_float(metrics.get("fitness", speedup))
        known = {
            "compiled",
            "correctness",
            "is_valid",
            "valid",
            "runtime_us",
            "ref_runtime_us",
            "speedup",
            "fitness",
            "status",
            "error",
            "metadata",
        }
        extra = {str(key): value for key, value in metrics.items() if key not in known}
        metadata = metrics.get("metadata")
        if isinstance(metadata, Mapping):
            extra.update({str(key): value for key, value in metadata.items()})
        return cls(
            compiled=compiled,
            correctness=correctness,
            valid=valid,
            runtime_us=runtime_us,
            ref_runtime_us=ref_runtime_us,
            speedup=speedup,
            fitness=fitness,
            status=str(metrics.get("status", "passed" if valid else "failed")),
            error=str(metrics.get("error", "") or ""),
            metadata=extra,
        )

    @classmethod
    def failed(cls, error: BaseException | str) -> "EvaluationResult":
        return cls(status="failed", error=str(error))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "passed", "valid"}
    return bool(value)


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _positive_float(value: Any) -> float | None:
    parsed = _as_float(value)
    return parsed if parsed > 0 and parsed < 1_000_000_000.0 else None


def is_repairable_result(
    result: Any,
    repair_count: int,
    repair_limit: int,
) -> bool:
    if not isinstance(result, Mapping) or repair_count >= repair_limit:
        return False
    if bool(result.get("valid")):
        return False
    status = str(result.get("status", ""))
    return (
        not bool(result.get("compiled"))
        or not bool(result.get("correctness"))
        or status in {"invalid_compliance", "invalid_codegen", "invalid_graph"}
    )
