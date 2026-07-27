from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from .tasks import TaskSpec


EVALUATOR_MARKER = "# === CUTE_HARNESS_EVALUATOR_V1 ==="


@dataclass(frozen=True)
class EvaluationConfig:
    seed: int
    warmup: int = 2
    repeats: int = 5

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("evaluation seed must be non-negative")
        if self.warmup < 0:
            raise ValueError("benchmark warmup must be non-negative")
        if self.repeats < 1:
            raise ValueError("benchmark repeats must be positive")


def default_evaluation_config(task: TaskSpec) -> EvaluationConfig:
    problem = task.data.get("problem")
    if not isinstance(problem, dict) or not isinstance(problem.get("seed"), int):
        raise RuntimeError(f"{task.id}: problem.seed must be an integer")
    return EvaluationConfig(seed=int(problem["seed"]))


def _split_source(source: str, source_name: str) -> tuple[str, str]:
    before, marker, after = source.partition(EVALUATOR_MARKER)
    if not marker:
        raise RuntimeError(
            f"{source_name}: missing evaluator marker "
            f"{EVALUATOR_MARKER!r}"
        )
    if EVALUATOR_MARKER in after:
        raise RuntimeError(f"{source_name}: multiple evaluator markers")
    return before.rstrip() + "\n", after.lstrip()


def split_starter(task: TaskSpec) -> tuple[str, str]:
    source = task.starter_path.read_text(encoding="utf-8")
    return _split_source(source, f"{task.id}: starter")


def candidate_starter(task: TaskSpec) -> str:
    candidate, _ = split_starter(task)
    return candidate


def baseline_candidate(task: TaskSpec) -> str:
    source = task.baseline_path.read_text(encoding="utf-8")
    candidate, _ = _split_source(source, f"{task.id}: baseline")
    return candidate


def install_task_references(task: TaskSpec, workspace: Path) -> list[str]:
    references_dir = workspace / "references"
    installed: list[str] = []
    for reference_path in task.reference_paths:
        references_dir.mkdir(exist_ok=True)
        destination = references_dir / reference_path.name
        shutil.copy2(reference_path, destination)
        installed.append(destination.relative_to(workspace).as_posix())
    return installed


def assemble_submission(
    task: TaskSpec,
    candidate_path: Path,
    config: EvaluationConfig | None = None,
) -> str:
    candidate = candidate_path.read_text(encoding="utf-8")
    if EVALUATOR_MARKER in candidate:
        raise RuntimeError("candidate contains the reserved evaluator marker")
    evaluation = config or default_evaluation_config(task)
    _, evaluator = split_starter(task)
    return (
        candidate.rstrip()
        + "\n\n"
        + EVALUATOR_MARKER
        + "\n"
        + f"_CUTE_HARNESS_SEED = {evaluation.seed}\n"
        + f"_CUTE_HARNESS_WARMUP = {evaluation.warmup}\n"
        + f"_CUTE_HARNESS_REPEATS = {evaluation.repeats}\n\n"
        + evaluator
    )
