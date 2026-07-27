from __future__ import annotations

from pathlib import Path

from .tasks import TaskSpec


EVALUATOR_MARKER = "# === CUTE_HARNESS_EVALUATOR_V1 ==="


def split_starter(task: TaskSpec) -> tuple[str, str]:
    source = task.starter_path.read_text(encoding="utf-8")
    before, marker, after = source.partition(EVALUATOR_MARKER)
    if not marker:
        raise RuntimeError(
            f"{task.id}: starter is missing evaluator marker "
            f"{EVALUATOR_MARKER!r}"
        )
    if EVALUATOR_MARKER in after:
        raise RuntimeError(f"{task.id}: starter has multiple evaluator markers")
    return before.rstrip() + "\n", after.lstrip()


def candidate_starter(task: TaskSpec) -> str:
    candidate, _ = split_starter(task)
    return candidate


def assemble_submission(task: TaskSpec, candidate_path: Path) -> str:
    candidate = candidate_path.read_text(encoding="utf-8")
    if EVALUATOR_MARKER in candidate:
        raise RuntimeError("candidate contains the reserved evaluator marker")
    _, evaluator = split_starter(task)
    return (
        candidate.rstrip()
        + "\n\n"
        + EVALUATOR_MARKER
        + "\n"
        + evaluator
    )
