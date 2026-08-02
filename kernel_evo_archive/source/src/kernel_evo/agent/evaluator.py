"""Pluggable candidate evaluators used by the CLI and direct Python API."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from kernel_evo.agent.errors import ConfigurationError
from kernel_evo.agent.models import EvaluationContext, EvaluationResult
from kernel_evo.cute_harness.b300 import (
    EvaluationConfig,
    baseline_candidate,
    evaluate,
    load_task,
    metrics,
)


@runtime_checkable
class CandidateEvaluator(Protocol):
    def evaluate(self, context: EvaluationContext) -> EvaluationResult | Mapping[str, Any]: ...


class KernelBenchEvaluator:
    """Adapter around the same validator used by headless GigaEvo runs."""

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        if context.problem_dir is None:
            raise ConfigurationError(
                "KernelBench evaluation needs problem_path (or level/problem_id), "
                "or configure evaluation.command."
            )
        code = context.candidate_path.read_text(encoding="utf-8")
        run_config = dict(context.run_config)
        ref_arch_src = str(run_config.get("ref_arch_src", ""))
        if not ref_arch_src:
            raise ConfigurationError("Prepared run_config.json has no reference model source")

        if str(run_config.get("execution_mode", "local_execution")) == "remote_execution":
            metrics = _remote_validate(run_config, code, context)
        else:
            from kernel_evo.resources.validate import run_local_validation

            metrics = run_local_validation(
                context.problem_dir,
                run_config,
                {
                    "id": f"iter-{context.iteration}-island-{context.island}",
                    "code": code,
                },
                code,
                ref_arch_src,
            )
        return EvaluationResult.from_metrics(metrics)


class CuteB300Evaluator:
    """Evaluate repository AIRI tasks with the remote B300 harness."""

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        task = load_task(str(context.config.get("problem_path", "")))
        config = EvaluationConfig(
            seed=int(context.config.get("evaluation_seed", 0)),
            warmup=int(context.config.get("evaluation_warmup", 2)),
            repeats=int(context.config.get("evaluation_repeats", 5)),
            timeout=float(context.config.get("evaluator_timeout", 900.0)),
            profile_timeline=bool(context.config.get("profile_timeline", False)),
        )
        baseline_root = context.run_dir / "b300" / "baseline"
        baseline_result_path = baseline_root / "result.json"
        if baseline_result_path.is_file():
            baseline_record = json.loads(baseline_result_path.read_text(encoding="utf-8"))
        else:
            baseline_source = context.run_dir / "b300" / "baseline.py"
            baseline_source.parent.mkdir(parents=True, exist_ok=True)
            baseline_source.write_text(baseline_candidate(task), encoding="utf-8")
            baseline_record = evaluate(
                task,
                baseline_source,
                baseline_root,
                config,
                harness_url=str(context.config.get("harness_url", "")),
            )
        reference_ms = baseline_record.get("kernel_time_ms")
        if not baseline_record.get("passed") or not isinstance(reference_ms, (int, float)):
            raise RuntimeError("AIRI task baseline failed B300 validation")

        candidate_hash = hashlib.sha256(context.candidate_path.read_bytes()).hexdigest()
        record = (
            baseline_record
            if candidate_hash == baseline_record.get("candidate_sha256")
            else evaluate(
                task,
                context.candidate_path,
                context.island_dir / "b300",
                config,
                harness_url=str(context.config.get("harness_url", "")),
            )
        )
        return EvaluationResult.from_metrics(metrics(record, float(reference_ms)))


class CommandEvaluator:
    """Run a deterministic external harness that prints one JSON metrics object."""

    def __init__(self, command: tuple[str, ...], *, timeout: float = 900.0) -> None:
        if not command:
            raise ConfigurationError("evaluation.command cannot be empty")
        self.command = command
        self.timeout = timeout

    def evaluate(self, context: EvaluationContext) -> EvaluationResult:
        replacements = {
            "{candidate}": str(context.candidate_path),
            "{baseline}": str(context.baseline_path),
            "{run_dir}": str(context.run_dir),
            "{island_dir}": str(context.island_dir),
            "{iteration}": str(context.iteration),
            "{island}": str(context.island),
            "{run_id}": context.run_id,
            "{device}": str(context.config.get("device", "cuda:0")),
            "{measurement_mode}": str(
                context.config.get("measurement_mode", "wall-clock")
            ),
            "{custom_tests}": str(context.config.get("custom_tests", "")),
        }
        command = [_replace_placeholders(argument, replacements) for argument in self.command]
        env = os.environ.copy()
        env.update(
            {
                "KERNELEVO_CANDIDATE": str(context.candidate_path),
                "KERNELEVO_BASELINE": str(context.baseline_path),
                "KERNELEVO_RUN_ID": context.run_id,
                "KERNELEVO_ITERATION": str(context.iteration),
                "KERNELEVO_ISLAND": str(context.island),
                "KERNELEVO_CUSTOM_TESTS": str(context.config.get("custom_tests", "")),
                "KERNELEVO_MEASUREMENT_MODE": str(
                    context.config.get("measurement_mode", "wall-clock")
                ),
            }
        )
        problem_path = str(context.config.get("problem_path", "") or "")
        baseline_path = str(context.config.get("baseline", "") or "")
        task_root_source = problem_path or baseline_path
        if task_root_source:
            source_path = Path(task_root_source).expanduser().resolve()
            env["KERNELEVO_TASK_ROOT"] = str(
                source_path if source_path.is_dir() else source_path.parent
            )
        completed = subprocess.run(
            command,
            cwd=context.run_dir,
            env=env,
            text=True,
            capture_output=True,
            timeout=self.timeout,
            check=False,
        )
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            raise RuntimeError(
                f"Evaluator command exited with {completed.returncode}: {detail[-4_000:]}"
            )
        metrics = _parse_json_output(completed.stdout)
        result = EvaluationResult.from_metrics(metrics)
        if completed.stderr.strip():
            result.metadata["evaluator_stderr"] = completed.stderr.strip()[-4_000:]
        return result


def evaluator_from_config(config: Mapping[str, Any]) -> CandidateEvaluator:
    command_value = config.get("evaluator_command", ())
    if isinstance(command_value, list):
        command = tuple(str(value) for value in command_value)
    elif isinstance(command_value, tuple):
        command = command_value
    else:
        command = ()
    if command:
        return CommandEvaluator(command, timeout=float(config.get("evaluator_timeout", 900.0)))
    if str(config.get("evaluator_kind", "kernelbench")) == "cute_b300":
        return CuteB300Evaluator()
    return KernelBenchEvaluator()


def coerce_evaluation_result(value: EvaluationResult | Mapping[str, Any]) -> EvaluationResult:
    if isinstance(value, EvaluationResult):
        return value
    if isinstance(value, Mapping):
        return EvaluationResult.from_metrics(value)
    raise TypeError(f"Evaluator returned {type(value).__name__}; expected EvaluationResult or mapping")


def _replace_placeholders(argument: str, replacements: Mapping[str, str]) -> str:
    result = argument
    for marker, value in replacements.items():
        result = result.replace(marker, value)
    return result


def _parse_json_output(stdout: str) -> Mapping[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        raise RuntimeError("Evaluator command produced no JSON output")
    candidates = [stripped, *reversed([line.strip() for line in stripped.splitlines() if line.strip()])]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    raise RuntimeError(f"Evaluator command did not print a JSON object: {stripped[-2_000:]}")


def _remote_validate(
    run_config: Mapping[str, Any],
    code: str,
    context: EvaluationContext,
) -> Mapping[str, Any]:
    import requests

    server_url = str(run_config.get("remote_validator_url", "http://localhost:15000")).rstrip("/")
    payload = {
        "id": f"iter-{context.iteration}-island-{context.island}",
        "code": code,
    }
    response = requests.post(
        f"{server_url}/schedule_validate",
        json={"cfg": dict(run_config), "payload": payload},
        timeout=30,
    )
    response.raise_for_status()
    job_id = response.json()["job_id"]
    poll_interval = float(run_config.get("remote_poll_interval", 1.0))
    deadline = time.monotonic() + float(context.config.get("evaluator_timeout", 900.0))
    while time.monotonic() < deadline:
        response = requests.get(
            f"{server_url}/fetch_validate_results",
            params={"job_id": job_id},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if data["status"] == "completed":
            result = data["result"]
            if not isinstance(result, Mapping):
                raise RuntimeError("Remote evaluator returned a non-object result")
            return result
        if data["status"] == "failed":
            raise RuntimeError(str(data.get("error_msg", "Remote validation failed")))
        time.sleep(max(0.05, poll_interval))
    raise TimeoutError(f"Remote evaluation timed out for job {job_id}")
