"""Constrained MCP service for Nsight Compute profiling.

The server exposes no shell or arbitrary command tool. Candidate and report paths
must resolve beneath one configured experiment root, devices come from a fixed
allowlist, and NCU sections are fixed. NCU and its target run as the same user
that launched this service.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import subprocess
import threading
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP

from kernel_evo.tools.ncu_gpu_guard import NcuGpuGuard

_COUNTERS = {
    "gpu__time_duration.sum": "duration_us",
    "dram__bytes.sum.per_second": "dram_throughput",
    "dram__cycles_active.avg.pct_of_peak_sustained_elapsed": "dram_peak_pct",
    "gpu__compute_memory_access_throughput.avg.pct_of_peak_sustained_elapsed": "memory_peak_pct",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_peak_pct",
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed": "tensor_pipe_peak_pct",
    "sm__maximum_warps_per_active_cycle_pct": "active_warps_pct",
    "launch__registers_per_thread": "registers_per_thread",
    "launch__shared_mem_per_block_allocated": "shared_mem_per_block",
}
_SECTIONS = ("SpeedOfLight", "MemoryWorkloadAnalysis", "ComputeWorkloadAnalysis")
_HOT_KERNEL_REGEX = (
    "regex:(deep_gemm::sm90_fp8_gemm_1d2d_impl|"
    "fused_recurrent_gated_delta_rule_fwd_kernel|direct_copy_kernel_cuda)"
)


def _inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _parse_number(value: str) -> float | str:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return value


def _compact_csv(csv_text: str) -> dict[str, Any]:
    rows = list(csv.reader(io.StringIO(csv_text)))
    header_index = next(
        (index for index, row in enumerate(rows) if "Kernel Name" in row), None
    )
    if header_index is None or header_index + 2 > len(rows):
        return {"status": "failed", "reason": "NCU CSV header was not found"}
    headers = rows[header_index]
    units = rows[header_index + 1]
    kernel_index = headers.index("Kernel Name")
    metric_columns = {
        index: _COUNTERS[name]
        for index, name in enumerate(headers)
        if name in _COUNTERS
    }
    kernels: list[dict[str, Any]] = []
    for row in rows[header_index + 2 :]:
        if len(row) <= kernel_index or not row[kernel_index].strip():
            continue
        metrics: dict[str, Any] = {}
        for index, compact_name in metric_columns.items():
            if index >= len(row) or not row[index].strip():
                continue
            metrics[compact_name] = {
                "value": _parse_number(row[index].strip()),
                "unit": units[index].strip() if index < len(units) else "",
            }
        kernels.append({"kernel_name": row[kernel_index], "metrics": metrics})
    return {
        "status": "completed",
        "kernel_count": len(kernels),
        "kernels": kernels,
    }



def _unprivileged_counters_enabled() -> bool:
    params = Path("/proc/driver/nvidia/params")
    try:
        for line in params.read_text(encoding="utf-8").splitlines():
            if line.startswith("RmProfilingAdminOnly:"):
                return line.partition(":")[2].strip() == "0"
    except OSError:
        return False
    return False


class RestrictedNcuService:
    def __init__(
        self,
        *,
        experiment_root: Path,
        allowed_devices: set[int],
        ncu_path: Path,
        python_path: Path,
        timeout: int,
    ) -> None:
        self.root = experiment_root.resolve()
        self.allowed_devices = allowed_devices
        self.ncu = ncu_path.resolve()
        self.cuda_home = self.ncu.parent.parent
        self.python = python_path.resolve()
        self.timeout = timeout
        self.driver = (self.root / "profile_fp8_baseline.py").resolve()
        self.output_root = (self.root / "external_ncu_reports").resolve()
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.tmpdir = (
            Path.home() / ".cache" / "kernel-evo" / "ncu-mcp-tmp"
        ).resolve()
        self.tmpdir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.jobs: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ncu-mcp")
        self.gpu_guard = NcuGpuGuard(self.allowed_devices)
        self.max_contamination_retries = 3

    def _profile_env(self) -> dict[str, str]:
        env = dict(os.environ)
        cuda_bin = str(self.cuda_home / "bin")
        current_path = str(env.get("PATH", "") or "")
        env.update(
            {
                "CUDA_HOME": str(self.cuda_home),
                "CUDA_PATH": str(self.cuda_home),
                "TMPDIR": str(self.tmpdir),
                "PATH": cuda_bin + (os.pathsep + current_path if current_path else ""),
            }
        )
        return env

    def validate_candidate(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if not _inside(path, self.root) or not path.is_file():
            raise ValueError("candidate must be an existing file inside the experiment root")
        relative = path.relative_to(self.root)
        allowed_seed = relative == Path("production_fp8.py")
        allowed_run_candidate = (
            path.name == "candidate.py"
            and ("runs" in relative.parts or ".kernelevo" in relative.parts)
            and any(part in {"candidate", "seed", "baseline"} for part in relative.parts)
        )
        if not (allowed_seed or allowed_run_candidate):
            raise ValueError("candidate path is not a production seed or KernelEvo candidate")
        return path

    def validate_report(self, raw_path: str) -> Path:
        path = Path(raw_path).expanduser().resolve()
        if (
            not _inside(path, self.root)
            or not path.is_file()
            or path.suffix != ".ncu-rep"
        ):
            raise ValueError("report must be an existing .ncu-rep inside the experiment root")
        return path

    def _import_report(self, report: Path) -> dict[str, Any]:
        proc = subprocess.run(
            [str(self.ncu), "--import", str(report), "--page", "raw", "--csv"],
            text=True,
            capture_output=True,
            timeout=min(self.timeout, 300),
            env=self._profile_env(),
        )
        if proc.returncode != 0:
            return {
                "status": "failed",
                "reason": "ncu report import failed",
                "returncode": proc.returncode,
                "stderr_excerpt": proc.stderr[-1000:],
            }
        compact = _compact_csv(proc.stdout)
        compact["report_file"] = str(report)
        return compact

    def compact_existing(
        self, report_path: str, candidate_path: str
    ) -> dict[str, Any]:
        report = self.validate_report(report_path)
        candidate = self.validate_candidate(candidate_path)
        compact = self._import_report(report)
        output = report.with_suffix(".compact.json")
        output.write_text(json.dumps(compact, indent=2), encoding="utf-8")
        job_id = "imported-" + uuid.uuid4().hex
        job = {
            "job_id": job_id,
            "status": "completed",
            "candidate": str(candidate),
            "device": "external-report",
            "scope": "imported",
            "job_dir": str(report.parent),
            "report_file": str(report),
            "compact_report": str(output),
            "kernel_count": compact.get("kernel_count", 0),
        }
        with self.lock:
            self.jobs[job_id] = job
        return {**job, **compact}

    def start(self, candidate_path: str, scope: str) -> dict[str, Any]:
        candidate = self.validate_candidate(candidate_path)
        if os.geteuid() != 0 and not _unprivileged_counters_enabled():
            raise RuntimeError(
                "NVIDIA counters are admin-only (RmProfilingAdminOnly=1). "
                "Enable NVreg_RestrictProfilingToAdminUsers=0 and restart the driver; "
                "or launch this constrained MCP as root."
            )
        if scope not in {"hot", "full"}:
            raise ValueError("scope must be 'hot' or 'full'")
        job_id = uuid.uuid4().hex
        job_dir = self.output_root / f"{int(time.time())}-{job_id[:8]}"
        job_dir.mkdir(parents=True, exist_ok=False)
        job = {
            "job_id": job_id,
            "status": "queued",
            "candidate": str(candidate),
            "device": "auto",
            "scope": scope,
            "job_dir": str(job_dir),
        }
        with self.lock:
            self.jobs[job_id] = job
        self.executor.submit(self._run_guarded, job_id, candidate, scope, job_dir)
        return dict(job)

    def status(self, job_id: str) -> dict[str, Any]:
        with self.lock:
            if job_id not in self.jobs:
                raise ValueError("unknown job id")
            return dict(self.jobs[job_id])

    def _profile_command(
        self, *, candidate: Path, device: int, scope: str, report_base: Path
    ) -> list[str]:
        command = [
            str(self.ncu),
            "--target-processes",
            "all",
            "--replay-mode",
            "kernel",
            "--nvtx",
            "--nvtx-include",
            "profile_fp8_baseline/",
            "--kernel-name-base",
            "demangled",
        ]
        if scope == "hot":
            command.extend(["--kernel-name", _HOT_KERNEL_REGEX])
        for section in _SECTIONS:
            command.extend(["--section", section])
        command.extend(
            [
                "--force-overwrite",
                "--export",
                str(report_base),
                str(self.python),
                str(self.driver),
                "--candidate",
                str(candidate),
                "--device",
                f"cuda:{device}",
                "--batch",
                "128",
                "--warmups",
                "8",
                "--iterations",
                "1",
            ]
        )
        return command

    def _run_guarded(
        self, job_id: str, candidate: Path, scope: str, job_dir: Path
    ) -> None:
        self._update(job_id, status="waiting_for_idle_gpu", started_at=time.time())
        attempts: list[dict[str, Any]] = []
        for attempt_index in range(1, self.max_contamination_retries + 1):
            try:
                device = self.gpu_guard.select_idle_gpu(timeout=180)
            except TimeoutError as exc:
                self._update(
                    job_id,
                    status="failed",
                    finished_at=time.time(),
                    error=str(exc),
                    attempts=attempts,
                )
                return
            attempt_dir = job_dir / f"attempt_{attempt_index}"
            attempt_dir.mkdir(parents=True, exist_ok=False)
            report_base = attempt_dir / "report"
            command = self._profile_command(
                candidate=candidate,
                device=device,
                scope=scope,
                report_base=report_base,
            )
            self._update(
                job_id,
                status="profiling",
                device=device,
                attempt=attempt_index,
                attempts=attempts,
            )
            result = self.gpu_guard.run_monitored(
                command,
                device=device,
                timeout=self.timeout,
                env=self._profile_env(),
            )
            report = report_base.with_suffix(".ncu-rep")
            attempt = {
                "attempt": attempt_index,
                "device": device,
                "returncode": result.returncode,
                "contaminated": result.contaminated,
                "contaminating_pids": list(result.contaminating_pids),
                "timed_out": result.timed_out,
                "report_exists": report.is_file(),
                "valid_for_review": False,
            }
            attempts.append(attempt)
            status_file = attempt_dir / "status.json"
            status_file.write_text(
                json.dumps(
                    {
                        **attempt,
                        "stdout_excerpt": result.stdout[-2000:],
                        "stderr_excerpt": result.stderr[-2000:],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            if result.contaminated:
                self._update(
                    job_id,
                    status="contaminated_retry",
                    attempts=attempts,
                    invalidated_attempt=attempt_index,
                )
                continue
            if result.timed_out:
                self._update(
                    job_id,
                    status="failed",
                    finished_at=time.time(),
                    error=f"NCU exceeded the fixed {self.timeout}s timeout",
                    attempts=attempts,
                )
                return
            if result.returncode != 0 or not report.is_file():
                self._update(
                    job_id,
                    status="failed",
                    finished_at=time.time(),
                    error="NCU failed; inspect constrained attempt status",
                    status_file=str(status_file),
                    attempts=attempts,
                )
                return
            attempt["valid_for_review"] = True
            compact = self._import_report(report)
            compact["gpu_guard"] = {
                "device": device,
                "attempt": attempt_index,
                "exclusive_through_end": True,
                "contamination_retries": attempt_index - 1,
            }
            compact_file = job_dir / "ncu_compact.json"
            compact_file.write_text(json.dumps(compact, indent=2), encoding="utf-8")
            self._update(
                job_id,
                status="completed",
                finished_at=time.time(),
                device=device,
                report_file=str(report),
                compact_report=str(compact_file),
                kernel_count=compact.get("kernel_count", 0),
                attempts=attempts,
            )
            return
        self._update(
            job_id,
            status="failed",
            finished_at=time.time(),
            error=(
                "all NCU attempts were invalidated by external per-process GPU activity"
            ),
            attempts=attempts,
        )

    def build_review_packet(
        self, job_id: str, torch_profile_path: str
    ) -> dict[str, Any]:
        job = self.status(job_id)
        if job.get("status") != "completed":
            raise ValueError("NCU job is not completed")
        torch_path = Path(torch_profile_path).expanduser().resolve()
        if (
            not _inside(torch_path, self.root)
            or not torch_path.is_file()
            or torch_path.name != "PARENT_PROFILE.md"
        ):
            raise ValueError("Torch packet must be a PARENT_PROFILE.md inside the experiment")
        compact = json.loads(Path(job["compact_report"]).read_text(encoding="utf-8"))
        packet = {
            "candidate": job["candidate"],
            "torch_compact_profile": torch_path.read_text(encoding="utf-8"),
            "ncu_compact_profile": compact,
            "requested_idea_count": 7,
            "review_instructions": (
                "Explain how the kernel/layer should be optimized for a large measurable speedup, "
                "which changes are most valuable, exactly where each belongs, and why. Produce a "
                "causal report followed by 5-7 ranked ideas. For every idea cite measured Torch/NCU "
                "evidence, implementation location, expected mechanism, concrete implementation plan, "
                "estimated upside, confidence, and correctness risk. Do not merely summarize traces."
            ),
        }
        output = Path(job["job_dir"]) / "combined_review_packet.json"
        output.write_text(json.dumps(packet, indent=2), encoding="utf-8")
        return {"review_packet": str(output), "candidate": job["candidate"]}

    def _update(self, job_id: str, **values: Any) -> None:
        with self.lock:
            self.jobs[job_id].update(values)


def create_mcp(service: RestrictedNcuService, *, host: str, port: int) -> FastMCP:
    mcp = FastMCP(
        "restricted-kernel-evo-ncu",
        instructions=(
            "Only profile validated KernelEvo candidates with fixed Nsight Compute "
            "sections and retrieve compact counter summaries."
        ),
        host=host,
        port=port,
        streamable_http_path="/mcp",
        stateless_http=True,
        json_response=True,
    )

    @mcp.tool()
    def start_ncu_profile(
        candidate_path: str, scope: str = "full"
    ) -> dict[str, Any]:
        """Queue a bounded NCU profile for a validated experiment candidate."""
        return service.start(candidate_path, scope)

    @mcp.tool()
    def get_ncu_profile(job_id: str) -> dict[str, Any]:
        """Return status and compact artifact paths for one queued profile."""
        return service.status(job_id)

    @mcp.tool()
    def compact_existing_ncu_report(
        report_path: str, candidate_path: str
    ) -> dict[str, Any]:
        """Compact an existing in-experiment NCU report without rerunning kernels."""
        return service.compact_existing(report_path, candidate_path)

    @mcp.tool()
    def build_combined_review_packet(
        job_id: str, torch_profile_path: str
    ) -> dict[str, Any]:
        """Combine compact NCU counters and a KernelEvo Torch profile packet."""
        return service.build_review_packet(job_id, torch_profile_path)

    return mcp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", required=True)
    parser.add_argument("--devices", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--ncu", default="/usr/local/cuda/bin/ncu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--transport", choices=("stdio", "streamable-http"), default="streamable-http"
    )
    args = parser.parse_args()
    devices = {int(value.strip()) for value in args.devices.split(",") if value.strip()}
    service = RestrictedNcuService(
        experiment_root=Path(args.experiment_root),
        allowed_devices=devices,
        ncu_path=Path(args.ncu),
        python_path=Path(args.python),
        timeout=max(60, min(args.timeout, 1800)),
    )
    mcp = create_mcp(service, host=args.host, port=args.port)
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
