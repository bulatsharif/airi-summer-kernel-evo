from __future__ import annotations

import json
import os
import signal
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .models import Profiler, RunResponse


@dataclass(frozen=True)
class ProcessResult:
    exit_code: int | None
    stdout: str
    stderr: str
    timed_out: bool


class HarnessRunner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.worker = Path(__file__).with_name("worker.py").resolve()
        self.settings.artifact_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.settings.artifact_dir.chmod(0o700)

    def run(self, code: str, profiler: Profiler | None) -> RunResponse:
        with tempfile.TemporaryDirectory(prefix="cute-harness-") as temp_name:
            job_dir = Path(temp_name)
            source_path = job_dir / "submission.py"
            result_path = job_dir / "result.json"
            trace_path = job_dir / "pytorch-trace.json"
            source_path.write_text(code, encoding="utf-8")

            command = [
                self.settings.python_executable,
                str(self.worker),
                "--source",
                str(source_path),
                "--result",
                str(result_path),
            ]
            if profiler is Profiler.pytorch:
                command.extend(["--trace", str(trace_path)])

            process = self._execute(command, job_dir)
            metadata = self._load_metadata(result_path)
            success = (
                not process.timed_out
                and process.exit_code == 0
                and bool(metadata.get("success"))
            )
            profile_id = None
            profile_error = None

            if success and profiler is Profiler.pytorch:
                if trace_path.is_file():
                    profile_id = self._store_artifact(trace_path, ".json")
                else:
                    profile_error = "PyTorch did not produce a trace"
            elif success and profiler is Profiler.nsys:
                profile_id, profile_error = self._run_nsys(source_path, job_dir)

            return RunResponse(
                success=success,
                exit_code=process.exit_code,
                stdout=process.stdout,
                stderr=process.stderr,
                device_time_ms=metadata.get("device_time_ms") if success else None,
                profile_id=profile_id,
                profile_error=profile_error,
                timed_out=process.timed_out,
            )

    def artifact_path(self, profile_id: str) -> Path | None:
        if not profile_id or any(char not in "0123456789abcdef-" for char in profile_id):
            return None
        matches = list(self.settings.artifact_dir.glob(f"{profile_id}.*"))
        return matches[0] if len(matches) == 1 and matches[0].is_file() else None

    def _run_nsys(self, source_path: Path, job_dir: Path) -> tuple[str | None, str | None]:
        if shutil.which(self.settings.nsys_executable) is None:
            return None, f"{self.settings.nsys_executable!r} is not installed"
        result_path = job_dir / "nsys-result.json"
        output_base = job_dir / "nsys-profile"
        command = [
            self.settings.nsys_executable,
            "profile",
            "--force-overwrite=true",
            "--output",
            str(output_base),
            self.settings.python_executable,
            str(self.worker),
            "--source",
            str(source_path),
            "--result",
            str(result_path),
            "--no-torch-profile",
        ]
        process = self._execute(command, job_dir)
        report_path = output_base.with_suffix(".nsys-rep")
        if process.timed_out:
            return None, "NSight profiling timed out"
        if process.exit_code != 0:
            detail = process.stderr or process.stdout
            return None, f"NSight profiling failed: {detail}"
        if not report_path.is_file():
            return None, "NSight did not produce a .nsys-rep report"
        return self._store_artifact(report_path, ".nsys-rep"), None

    def _execute(self, command: list[str], cwd: Path) -> ProcessResult:
        stdout_path = cwd / f"stdout-{uuid.uuid4().hex}.log"
        stderr_path = cwd / f"stderr-{uuid.uuid4().hex}.log"
        timed_out = False
        exit_code: int | None = None
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                env=self._child_environment(cwd),
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            try:
                exit_code = process.wait(timeout=self.settings.timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()

        stdout = self._read_log(stdout_path)
        stderr = self._read_log(stderr_path)
        if timed_out:
            stderr += f"\nExecution timed out after {self.settings.timeout_seconds}s\n"
        return ProcessResult(
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            timed_out=timed_out,
        )

    def _child_environment(self, job_dir: Path) -> dict[str, str]:
        allowed_exact = {
            "CUDA_HOME",
            "CUDA_PATH",
            "CUDA_VISIBLE_DEVICES",
            "CUTE_ARCH",
            "LANG",
            "LC_ALL",
            "LD_LIBRARY_PATH",
            "PATH",
            "TORCH_EXTENSIONS_DIR",
        }
        allowed_prefixes = (
            "CUDA_",
            "CUTE_",
            "NCCL_",
            "NVCC_",
            "TORCH_",
            "TRITON_",
        )
        environment = {
            key: value
            for key, value in os.environ.items()
            if key in allowed_exact or key.startswith(allowed_prefixes)
        }
        environment.update(
            {
                "HOME": str(job_dir),
                "TMPDIR": str(job_dir),
                "PYTHONNOUSERSITE": "1",
            }
        )
        return environment

    def _read_log(self, path: Path) -> str:
        size = path.stat().st_size
        with path.open("rb") as handle:
            kept = handle.read(self.settings.max_log_bytes)
        if size <= self.settings.max_log_bytes:
            return kept.decode("utf-8", errors="replace")
        omitted = size - len(kept)
        return (
            kept.decode("utf-8", errors="replace")
            + f"\n[log truncated; {omitted} bytes omitted]\n"
        )

    @staticmethod
    def _load_metadata(path: Path) -> dict[str, object]:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def _store_artifact(self, source: Path, suffix: str) -> str:
        profile_id = str(uuid.uuid4())
        destination = self.settings.artifact_dir / f"{profile_id}{suffix}"
        shutil.move(str(source), destination)
        destination.chmod(0o600)
        return profile_id
