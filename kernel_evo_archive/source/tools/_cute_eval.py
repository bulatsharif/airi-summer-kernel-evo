"""One bounded B300 evaluation, callable by the author during its own turn.

Invoked through `tools/cute-eval`, which is the only form the agent is allowed
to run. Prints a compact verdict on stdout: the author never reads the
evaluation artifacts, so nothing here can expose the reference implementation,
whose directory stays denied at the read layer.

Each call is recorded under the island's `agent-evals/` so the runner can report
evaluations per turn and evaluations to first pass -- the metric this change
exists to make measurable.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

from kernel_evo.cute_harness.b300 import EvaluationConfig, evaluate, failure_reason, load_task
from kernel_evo.cute_harness.b300_policy import check_candidate

DEFAULT_BUDGET = 4


def island_dir(candidate: Path) -> Path:
    """The island owning this candidate: <island>/candidate/submission.py."""
    return candidate.resolve().parents[1]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: cute-eval <task_id> <candidate_path>", file=sys.stderr)
        return 2
    task_id, candidate_arg = argv
    candidate = Path(candidate_arg).expanduser()
    if not candidate.is_file():
        print(f"FAIL: candidate does not exist: {candidate}")
        return 1

    try:
        task = load_task(task_id)
    except (KeyError, ValueError, FileNotFoundError) as error:
        print(f"FAIL: unknown task {task_id}: {error}")
        return 1

    # A candidate the gate rejects would be refused by the evaluator anyway;
    # catching it here keeps a doomed submission off the shared device and does
    # not spend budget.
    report = check_candidate(candidate, task.policy)
    if not report.passed:
        print("REJECTED (no GPU time used, budget not spent):")
        for message in report.errors:
            print(f"  - {message}")
        print("Fix these and evaluate again.")
        return 1

    budget = int(os.environ.get("CUTE_AGENT_EVAL_BUDGET", DEFAULT_BUDGET))
    evals_root = island_dir(candidate) / "agent-evals"
    evals_root.mkdir(parents=True, exist_ok=True)
    used = len([p for p in evals_root.iterdir() if p.is_dir()])

    # Re-running an unchanged candidate returns the identical verdict and buys
    # nothing. One run spent 7 of 23 device evaluations that way, repeating
    # `validation failed: full_abs=8.928038` to the digit.
    digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
    previous = evals_root / f"eval-{used:02d}" / "candidate.py"
    if used and previous.is_file() and hashlib.sha256(previous.read_bytes()).hexdigest() == digest:
        print("UNCHANGED: this candidate is byte-identical to your last evaluation.")
        print("It will fail the same way. Change the code first; no budget was spent.")
        return 1
    if used >= budget:
        print(f"BUDGET EXHAUSTED: {used}/{budget} evaluations used this turn.")
        print("Leave your best candidate in place and finish the turn.")
        return 1

    output_dir = evals_root / f"eval-{used + 1:02d}"
    config = EvaluationConfig(
        seed=int(os.environ.get("CUTE_EVAL_SEED", 0)),
        warmup=int(os.environ.get("CUTE_EVAL_WARMUP", 5)),
        repeats=int(os.environ.get("CUTE_EVAL_REPEATS", 50)),
        timeout=float(os.environ.get("CUTE_EVAL_TIMEOUT", 900.0)),
    )
    try:
        record = evaluate(task, candidate, output_dir, config)
    except Exception as error:  # noqa: BLE001 - the agent must see any failure as text
        print(f"FAIL: evaluation error: {type(error).__name__}: {error}")
        return 1

    remaining = budget - (used + 1)
    if record.get("passed"):
        runtime = record.get("kernel_time_ms")
        print(f"PASS  kernel_time_ms={runtime}")
        print(f"({used + 1}/{budget} evaluations used, {remaining} left.)")
        print("The kernel is correct. Stop here unless you intend to make it faster.")
        return 0

    print(f"FAIL  {failure_reason(record)}")
    print(f"({used + 1}/{budget} evaluations used, {remaining} left.)")
    print("Fix the reported error in the candidate and evaluate again." if remaining else
          "No evaluations left. Leave your best candidate in place and finish the turn.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
