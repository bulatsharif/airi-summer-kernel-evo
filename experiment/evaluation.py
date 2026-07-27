from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


@dataclass(frozen=True)
class EvaluationResult:
    exit_code: int
    record: dict[str, Any] | None
    error: str | None

    @property
    def passed(self) -> bool:
        if not self.record:
            return False
        acceptance = self.record.get("acceptance")
        return isinstance(acceptance, dict) and acceptance.get("passed") is True

    @property
    def kernel_time_ms(self) -> float | None:
        if not self.record:
            return None
        benchmark = self.record.get("benchmark")
        if not isinstance(benchmark, dict):
            return None
        value = benchmark.get("kernel_time_ms")
        return float(value) if isinstance(value, (int, float)) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "passed": self.passed,
            "kernel_time_ms": self.kernel_time_ms,
            "error": self.error,
            "record": self.record,
        }


def run_evaluation(
    *,
    repo_root: Path,
    task_id: str,
    output_dir: Path,
    log_path: Path,
    gpu_timeout: float,
    seed: int,
    warmup: int,
    repeats: int,
    label: str,
    candidate_path: Path | None = None,
    baseline: bool = False,
) -> EvaluationResult:
    command = [
        sys.executable,
        "-m",
        "cute_harness",
        "run",
        task_id,
    ]
    if baseline:
        command.append("--baseline")
    elif candidate_path is not None:
        command.append(str(candidate_path))
    else:
        raise ValueError("candidate_path is required for a candidate evaluation")
    command.extend(
        [
            "--timeout",
            str(gpu_timeout),
            "--seed",
            str(seed),
            "--warmup",
            str(warmup),
            "--repeats",
            str(repeats),
            "--output",
            str(output_dir),
            "--label",
            label,
        ]
    )
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        str(repo_root)
        if not existing_pythonpath
        else os.pathsep.join((str(repo_root), existing_pythonpath))
    )
    try:
        process = subprocess.run(
            command,
            cwd=repo_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=gpu_timeout + 45.0,
        )
        output = process.stdout
        if process.stderr:
            output += "\n--- evaluator stderr ---\n" + process.stderr
        exit_code = process.returncode
        diagnostic = (process.stderr or process.stdout).strip()
        error = (
            None
            if exit_code == 0
            else diagnostic[-2000:] or "evaluation command failed"
        )
    except subprocess.TimeoutExpired as timeout:
        stdout = timeout.stdout if isinstance(timeout.stdout, str) else ""
        stderr = timeout.stderr if isinstance(timeout.stderr, str) else ""
        output = stdout + "\n--- evaluator timeout ---\n" + stderr
        exit_code = 124
        error = "evaluation process timed out"
    log_path.write_text(output, encoding="utf-8")

    result_path = output_dir / "result.json"
    record = None
    if result_path.is_file():
        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                record = payload
        except (OSError, json.JSONDecodeError) as result_error:
            error = f"cannot read evaluation result: {result_error}"
    elif error is None:
        error = "evaluation produced no result.json"
    return EvaluationResult(exit_code, record, error)
