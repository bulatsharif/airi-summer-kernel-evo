from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Callable

from cute_harness.assembly import (
    candidate_starter,
    install_task_agent_skills,
    install_task_references,
)
from cute_harness.tasks import REPO_ROOT, TaskSpec, discover_tasks

from . import __version__
from .agent import AgentMetrics, run_agent
from .evaluation import EvaluationResult, run_evaluation
from .report import write_reports


WORKSPACE_INSTRUCTIONS = """# Experiment candidate workspace

Solve the task described in TASK.md and task.json.

- Load every skill listed in `task.json.agent_skills`, then read every file
  listed in `task.json.references` before planning or delegating exploration.
- Resolve `task.json.references` relative to this workspace exactly as written.
  Resolve links inside a skill relative to that skill's `SKILL.md`; do not
  search for or reinterpret either path family.
- Edit only submission.py.
- Do not define or call main(); the evaluator owns main and correctness checks.
- Do not inspect repository baselines, task evaluator source, or previous runs.
- Shell access is intentionally limited to plain harness feedback commands
  and the exact `cp ...-template.py submission.py` forms documented by the
  installed skill. Template copies are editable starting points, not locked
  code.
- For a task-selected dense GEMM template, the exact `cp` must be the first
  mutating tool call after the required reads; do not regenerate it with the
  write tool.
- Do not append pipes, redirects, command chains, or other shell utilities. Use the
  read tool for workspace and skill files; do not use `python3 -c` to inspect
  local CUTLASS.
- Treat remote harness compiler output as the installed CuTe API oracle. Fix
  its first concrete diagnostic before delegating any exploration.
- Establish correctness before optimizing.
- Leave the best final candidate in submission.py.
"""


@dataclass(frozen=True)
class ExperimentConfig:
    model: str
    task_ids: tuple[str, ...]
    attempts: int
    agent_timeout: float
    gpu_timeout: float
    seed: int
    warmup: int
    repeats: int
    output_dir: Path
    work_root: Path = REPO_ROOT / "work"

    def __post_init__(self) -> None:
        if not self.model or "/" not in self.model:
            raise ValueError("model must use provider/model format")
        if not self.task_ids:
            raise ValueError("at least one task is required")
        if self.attempts < 1:
            raise ValueError("attempts must be positive")
        if self.agent_timeout <= 0 or self.gpu_timeout <= 0:
            raise ValueError("timeouts must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.warmup < 0 or self.repeats < 1:
            raise ValueError("invalid benchmark warmup/repeats")


def default_output_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    return REPO_ROOT / "runs" / "experiments" / stamp


def _git_commit(repo_root: Path) -> str | None:
    process = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    value = process.stdout.strip()
    return value if process.returncode == 0 and value else None


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")
    return cleaned or "experiment"


def _prepare_workspace(
    task: TaskSpec,
    workspace: Path,
    seed: int,
) -> None:
    if workspace.exists():
        raise RuntimeError(f"workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    shutil.copy2(task.prompt_path, workspace / "TASK.md")
    source = candidate_starter(task)
    source = re.sub(
        r"(?m)^SEED = [0-9]+$",
        f"SEED = {seed}",
        source,
        count=1,
    )
    (workspace / "submission.py").write_text(source, encoding="utf-8")
    public = task.public_manifest()
    problem = public.get("problem")
    if isinstance(problem, dict):
        problem["seed"] = seed
    public["starter"] = "submission.py"
    public["references"] = install_task_references(task, workspace)
    public["agent_skills"] = install_task_agent_skills(task, workspace)
    (workspace / "task.json").write_text(
        json.dumps(public, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (workspace / "AGENTS.md").write_text(
        WORKSPACE_INSTRUCTIONS,
        encoding="utf-8",
    )


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _evaluation_label(
    experiment_id: str,
    model: str,
    task_id: str,
    suffix: str,
) -> str:
    return f"{experiment_id}:{model}:{task_id}:{suffix}"


AgentRunner = Callable[..., AgentMetrics]
EvaluationRunner = Callable[..., EvaluationResult]


def run_experiment(
    config: ExperimentConfig,
    *,
    agent_runner: AgentRunner = run_agent,
    evaluation_runner: EvaluationRunner = run_evaluation,
) -> tuple[bool, list[dict[str, Any]]]:
    tasks = discover_tasks()
    unknown = [task_id for task_id in config.task_ids if task_id not in tasks]
    if unknown:
        raise RuntimeError(f"unknown tasks: {', '.join(unknown)}")
    output_dir = config.output_dir.resolve()
    if output_dir.exists():
        raise RuntimeError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)

    experiment_id = output_dir.name
    work_root = config.work_root.resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "schema_version": 1,
        "experiment_version": __version__,
        "experiment_id": experiment_id,
        "requested_model": config.model,
        "task_ids": list(config.task_ids),
        "attempts": config.attempts,
        "agent_timeout_seconds": config.agent_timeout,
        "gpu_timeout_seconds": config.gpu_timeout,
        "seed": config.seed,
        "benchmark": {
            "warmup": config.warmup,
            "repeats": config.repeats,
        },
        "git_commit": _git_commit(REPO_ROOT),
        "qwen_base_url": os.environ.get("QWEN_BASE_URL"),
        "cute_harness_url": os.environ.get(
            "CUTE_HARNESS_URL",
            "http://109.236.57.62:18080",
        ),
        "started_at": started_at,
    }
    _write_json(output_dir / "manifest.json", manifest)

    rows: list[dict[str, Any]] = []
    all_passed = True
    output_hash = hashlib.sha256(
        str(output_dir).encode("utf-8")
    ).hexdigest()[:8]
    experiment_slug = _slug(f"{experiment_id}-{output_hash}")
    for task_id in config.task_ids:
        task = tasks[task_id]
        task_dir = output_dir / task_id
        task_dir.mkdir()
        baseline_dir = task_dir / "baseline"
        print(f"\n=== {task_id}: baseline evaluation ===", flush=True)
        baseline = evaluation_runner(
            repo_root=REPO_ROOT,
            task_id=task_id,
            output_dir=baseline_dir,
            log_path=task_dir / "baseline-eval.log",
            gpu_timeout=config.gpu_timeout,
            seed=config.seed,
            warmup=config.warmup,
            repeats=config.repeats,
            label=_evaluation_label(
                experiment_id,
                config.model,
                task_id,
                "baseline",
            ),
            baseline=True,
        )
        _write_json(task_dir / "baseline-result.json", baseline.to_dict())
        all_passed = all_passed and baseline.passed
        if not baseline.passed:
            print(
                f"baseline failed; skipping {config.attempts} agent "
                f"attempt(s): {baseline.error or 'unknown evaluator error'}",
                flush=True,
            )
            row = {
                "model": config.model,
                "requested_model": config.model,
                "task": task_id,
                "attempt": None,
                "status": "BASELINE_FAIL",
                "baseline_ms": baseline.kernel_time_ms,
                "agent_ms": None,
                "speedup": None,
                "input_uncached": None,
                "input_cached": None,
                "output_tokens": None,
                "reasoning_tokens": None,
                "agent_wall_seconds": None,
                "agent_session_id": None,
                "agent_sessions": None,
                "agent_exit_code": None,
                "profile_id": None,
                "workspace": None,
                "artifacts": str(task_dir),
            }
            rows.append(row)
            _write_json(task_dir / "result.json", row)
            write_reports(output_dir, rows)
            continue

        for attempt_index in range(1, config.attempts + 1):
            attempt_name = f"attempt-{attempt_index:03d}"
            attempt_dir = task_dir / attempt_name
            attempt_dir.mkdir()
            workspace_name = _slug(
                f"{experiment_slug}-{task_id}-{attempt_name}"
            )
            workspace = work_root / workspace_name
            _prepare_workspace(task, workspace, config.seed)

            print(
                f"\n=== {task_id}: agent {attempt_index}/{config.attempts} ===",
                flush=True,
            )
            agent_metrics = agent_runner(
                repo_root=REPO_ROOT,
                workspace=workspace,
                task_id=task_id,
                model=config.model,
                agent_timeout=config.agent_timeout,
                gpu_timeout=config.gpu_timeout,
                seed=config.seed,
                events_path=attempt_dir / "agent-events.jsonl",
                log_path=attempt_dir / "agent.log",
            )
            _write_json(
                attempt_dir / "agent-metrics.json",
                agent_metrics.to_dict(),
            )

            final_candidate = attempt_dir / "candidate.py"
            shutil.copy2(workspace / "submission.py", final_candidate)
            candidate_eval_dir = attempt_dir / "candidate-eval"
            print(
                f"\n=== {task_id}: authoritative candidate evaluation ===",
                flush=True,
            )
            candidate = evaluation_runner(
                repo_root=REPO_ROOT,
                task_id=task_id,
                candidate_path=final_candidate,
                output_dir=candidate_eval_dir,
                log_path=attempt_dir / "candidate-eval.log",
                gpu_timeout=config.gpu_timeout,
                seed=config.seed,
                warmup=config.warmup,
                repeats=config.repeats,
                label=_evaluation_label(
                    experiment_id,
                    config.model,
                    task_id,
                    attempt_name,
                ),
                baseline=False,
            )
            _write_json(
                attempt_dir / "candidate-result.json",
                candidate.to_dict(),
            )

            baseline_ms = baseline.kernel_time_ms
            candidate_ms = candidate.kernel_time_ms
            speedup = (
                baseline_ms / candidate_ms
                if baseline.passed
                and candidate.passed
                and baseline_ms is not None
                and candidate_ms is not None
                and candidate_ms > 0
                else None
            )
            if not baseline.passed:
                status = "BASELINE_FAIL"
            elif candidate.passed:
                status = "PASS"
            elif agent_metrics.timed_out:
                status = "TIMEOUT"
            else:
                status = "FAIL"
            all_passed = all_passed and candidate.passed
            reported_model = agent_metrics.reported_model or config.model
            candidate_profile_id = (
                candidate.record.get("response", {}).get("profile_id")
                if candidate.record
                and isinstance(candidate.record.get("response"), dict)
                else None
            )
            row = {
                "model": reported_model,
                "requested_model": config.model,
                "task": task_id,
                "attempt": attempt_index,
                "status": status,
                "baseline_ms": baseline_ms,
                "agent_ms": candidate_ms,
                "speedup": speedup,
                "input_uncached": agent_metrics.input_uncached,
                "input_cached": agent_metrics.input_cached,
                "output_tokens": agent_metrics.output,
                "reasoning_tokens": agent_metrics.reasoning,
                "agent_wall_seconds": agent_metrics.wall_seconds,
                "agent_session_id": agent_metrics.session_id,
                "agent_sessions": agent_metrics.sessions,
                "agent_exit_code": agent_metrics.exit_code,
                "agent_metrics_error": agent_metrics.metrics_error,
                "profile_id": candidate_profile_id,
                "workspace": str(workspace),
                "artifacts": str(attempt_dir),
            }
            rows.append(row)
            _write_json(attempt_dir / "result.json", row)
            write_reports(output_dir, rows)

    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["all_passed"] = all_passed
    _write_json(output_dir / "manifest.json", manifest)
    write_reports(output_dir, rows)
    return all_passed, rows
