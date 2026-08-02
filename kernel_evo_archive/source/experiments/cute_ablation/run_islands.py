"""Multi-island agentic run: N authors per turn, barrier selection, bounded repairs.

`run_iter_matrix.py` drives one island. Everything needed for more already exists
in the controller -- `island_count`, per-island elites, archive migration through
`select_baseline_entry(migration_interval=...)`, per-island idea diversity, and
bounded repair through `reopen_island_for_repair` -- so this driver only has to
stop assuming island 0.

The loop per iteration:

  1. `prepare_iteration` returns one AuthoringTask per island. Author all of them
     concurrently; each island's parent is chosen from the archive by the
     scheduler, so islands diverge and periodically pull from a shared elite.
  2. `evaluate_iteration` grades every island at one barrier.
  3. Any island whose failure is *localized* (did not compile, failed
     correctness, or a compliance/codegen/graph rejection) gets up to
     `max_repairs_per_island` bounded repair turns -- re-authored against its own
     diagnostic, re-evaluated, without spending a new iteration.
  4. `advance_iteration` promotes elites and moves on.

This is deliberately not the removed GigaEvo path: no diff mutation, no handing
the model a working kernel. Every island authors from the skeleton, exactly as
the single-island study did, so results stay comparable to it.

Isolation note: islands share one run directory and sit at `iter_NNN/island_{i}`,
so a worktree cannot separate them the way it separates arms. The agent's read
permissions must deny sibling `island_*` paths; verify with a probe before
trusting any number, and see RUNBOOK.md section 1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import yaml  # noqa: E402

from run_iter_matrix import (  # noqa: E402
    REPO_ROOT,
    agent_config,
    author_candidate,
    check_protocol_matches,
    freeze_documentation,
    load_task,
    summarize,
)
from kernel_evo import KernelEvoAgent  # noqa: E402
from kernel_evo.agent.models import is_repairable_result  # noqa: E402

PRINT_LOCK = threading.Lock()


def say(message: str) -> None:
    with PRINT_LOCK:
        print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def island_dir(run_dir: Path, iteration: int, island: int) -> Path:
    return run_dir / f"iter_{iteration:03d}" / f"island_{island}"


def author_one(task: Any, model: str, timeout: int, label: str) -> dict[str, Any]:
    """Author a single island, reusing the cached usage if it already ran."""
    trace_dir = task.candidate_path.parents[1] / "agent"
    usage_path = trace_dir / "usage.json"
    if usage_path.exists():
        return json.loads(usage_path.read_text(encoding="utf-8"))
    started = time.monotonic()
    usage = author_candidate(task, trace_dir, model, timeout)
    say(f"  {label} authored in {(time.monotonic() - started) / 60:.1f}m "
        f"(wrote={usage.get('wrote_candidate')}, evals={usage.get('agent_evaluations', 0)})")
    return usage


def run(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    task_id = config["task"]
    settings = config["agent"]
    islands = int(settings["islands"])
    steps = int(settings["steps"])
    repair_limit = int(settings.get("max_repairs_per_island", 0) or 0)
    tier = config["tiers"][0]

    root.mkdir(parents=True, exist_ok=True)
    check_protocol_matches(root, config)
    (root / "study.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    task = load_task(task_id)
    freeze_documentation(root, task, tier)

    arm = root / "iter" / task_id / tier / "r01"
    controller = KernelEvoAgent(arm / "kernel_evo")
    run_id = "run"
    run_dir = controller.runs_dir / run_id
    if not (run_dir / "state.json").exists():
        controller.init_run(agent_config(config, tier, REPO_ROOT / "tasks" / "cute" / "tasks" / task_id),
                            run_id=run_id)

    samples: list[dict[str, Any]] = []
    while True:
        status = controller.status(run_id)
        if status["phase"] == "complete":
            break
        iteration = int(status["current_iteration"])
        say(f"turn {iteration}/{steps}: authoring {islands} islands")

        if status["phase"] in {"ready", "authoring"}:
            tasks = controller.prepare_iteration(run_id)
            with ThreadPoolExecutor(max_workers=islands) as pool:
                futures = {
                    pool.submit(author_one, item, config["model"],
                                int(settings["session_timeout_seconds"]),
                                f"turn{iteration} island{index}"): index
                    for index, item in enumerate(tasks)
                }
                usages = {futures[f]: f.result() for f in futures}
        else:
            usages = {}
            for index in range(islands):
                path = island_dir(run_dir, iteration, index) / "agent" / "usage.json"
                usages[index] = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

        started = time.monotonic()
        report = controller.evaluate_iteration(run_id)
        evaluation_seconds = time.monotonic() - started
        rows = report.get("islands", [])
        valid = sum(1 for r in rows if r.get("valid"))
        say(f"turn {iteration}: barrier -> {valid}/{len(rows)} islands valid "
            f"({evaluation_seconds / 60:.1f}m)")

        # Bounded repair: only for localized failures, only while under the limit.
        # `rows` from the barrier is authoritative and is refreshed after every
        # repair; the loop is hard-capped so a controller that keeps reporting the
        # same failure cannot spin. The island's usage.json must be cleared first
        # or author_one returns the previous session and nothing is re-authored --
        # that bug spawned 9 sessions for a 2-island smoke run.
        for index in range(len(rows)):
            for attempt in range(1, repair_limit + 1):
                row = rows[index] if index < len(rows) else {}
                if not isinstance(row, dict) or not is_repairable_result(row, attempt - 1, repair_limit):
                    break
                try:
                    controller.reopen_island_for_repair(run_id, iteration, index)
                except Exception as error:  # not eligible, or the phase moved on
                    say(f"  turn{iteration} island{index}: repair refused ({type(error).__name__})")
                    break
                say(f"  turn{iteration} island{index}: repair {attempt}/{repair_limit}")
                repair_tasks = controller.prepare_iteration(run_id)
                usage_path = (
                    repair_tasks[index].candidate_path.parents[1] / "agent" / "usage.json"
                )
                usage_path.unlink(missing_ok=True)
                usages[index] = author_one(
                    repair_tasks[index], config["model"],
                    int(settings["session_timeout_seconds"]),
                    f"turn{iteration} island{index} repair{attempt}")
                report = controller.evaluate_iteration(run_id)
                rows = report.get("islands", [])
                valid = sum(1 for r in rows if r.get("valid"))
                say(f"  turn{iteration}: after repair -> {valid}/{len(rows)} islands valid")

        for index, row in enumerate(rows):
            runtime_us = row.get("runtime_us")
            samples.append({
                "approach": "islands", "task": task_id, "tier": tier, "replication": 1,
                "turn": iteration, "island": index,
                "passed": bool(row.get("valid")),
                "runtime_ms": float(runtime_us) / 1000 if runtime_us else None,
                "speedup": float(row.get("speedup", 0.0) or 0.0),
                "promoted": bool(row.get("promoted")),
                "error": str(row.get("error", "")),
                "repair_count": int(row.get("repair_count", 0) or 0),
                "evaluation_seconds": evaluation_seconds,
                **usages.get(index, {}),
            })
        (arm / "samples").mkdir(parents=True, exist_ok=True)
        (arm / "samples" / f"iteration-{iteration:03d}.json").write_text(
            json.dumps(samples[-len(rows):], indent=2) + "\n", encoding="utf-8")
        controller.advance_iteration(run_id)

    summary = summarize(samples)
    summary["islands"] = islands
    summary["island_turns"] = len(samples)
    (arm / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (arm / "manifest.json").write_text(json.dumps({
        "schema_version": 1, "approach": "islands", "task": task_id, "tier": tier,
        "replication": 1, "model": config["model"], "chains": islands,
        "turns": steps, "islands": islands,
        "max_repairs_per_island": repair_limit,
        "migration_interval": config.get("migration_interval", 3),
        "evaluation": config["evaluator"],
        "documentation_delivery": (config.get("documentation") or {}).get("delivery", "files"),
        "profiler_feedback": bool((config.get("profiling") or {}).get("enabled", False)),
        "critic": "",
    }, indent=2) + "\n", encoding="utf-8")
    say(f"COMPLETE: {summary['pass_count']}/{summary['samples']} island-turns valid, "
        f"best speedup {summary['best_speedup']:.4f}x")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default=os.environ.get("OPENCODE_MODEL", ""))
    parser.add_argument("--islands", type=int)
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    config["model"] = args.model or config["model"]
    if args.islands is not None:
        config["agent"]["islands"] = args.islands
    if args.steps is not None:
        config["agent"]["steps"] = args.steps
    run(args.results.resolve(), config)


if __name__ == "__main__":
    main()
