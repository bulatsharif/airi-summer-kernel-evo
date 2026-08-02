"""Advisory device leases and idle checks for trustworthy GPU benchmarks."""

from __future__ import annotations

import fcntl
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class GpuLease:
    """Hold an advisory per-device lock for the lifetime of one evaluator."""

    handle: Any
    metadata: dict[str, Any]
    index: int
    allow_co_resident: bool = False
    trusted_pids: frozenset[int] = frozenset()

    def verify_exclusive(self) -> bool:
        """Confirm no external compute process appeared after lease acquisition."""

        state = _probe_gpu(self.index)
        external = [
            process
            for process in state.get("compute_processes", [])
            if int(process.get("pid", -1)) != os.getpid()
            and int(process.get("pid", -1)) not in self.trusted_pids
        ]
        exclusive = not external
        accepted = exclusive or self.allow_co_resident
        self.metadata["final_state"] = state
        self.metadata["exclusive_through_end"] = exclusive
        self.metadata["co_resident_accepted"] = bool(self.allow_co_resident and external)
        if external:
            self.metadata["external_processes_at_end"] = external
        return accepted

    def close(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle.fileno(), fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None


def acquire_idle_gpu(
    device: str,
    *,
    timeout: float = 120.0,
    poll_interval: float = 1.0,
    consecutive_idle_samples: int = 3,
    max_utilization: int = 5,
    lock_dir: str | Path = "/tmp",
    allow_co_resident: bool | None = None,
) -> GpuLease:
    """Acquire a KernelEvo lease and require repeated externally idle samples."""

    index = _device_index(device)
    if allow_co_resident is None:
        allow_co_resident = os.environ.get("KERNELEVO_ALLOW_CORESIDENT_GPU", "").lower() in {
            "1",
            "true",
            "yes",
        }
    trusted_pids = _trusted_gpu_pids()
    deadline = time.monotonic() + max(0.0, float(timeout))
    lock_path = Path(lock_dir) / f"kernel-evo-gpu-{index}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    lock_acquired_at: float | None = None
    idle_streak = 0
    samples = 0
    last_state: dict[str, Any] = {}
    started = time.monotonic()
    try:
        while time.monotonic() <= deadline:
            if lock_acquired_at is None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    lock_acquired_at = time.monotonic()
                except BlockingIOError:
                    time.sleep(max(0.05, poll_interval))
                    continue
            last_state = _probe_gpu(index)
            samples += 1
            external = [
                process
                for process in last_state.get("compute_processes", [])
                if int(process.get("pid", -1)) not in trusted_pids
            ]
            idle = (
                int(last_state.get("utilization_gpu", 100)) <= int(max_utilization)
                and (bool(allow_co_resident) or not external)
            )
            idle_streak = idle_streak + 1 if idle else 0
            if idle_streak >= max(1, int(consecutive_idle_samples)):
                return GpuLease(
                    handle=handle,
                    index=index,
                    allow_co_resident=bool(allow_co_resident),
                    trusted_pids=frozenset(trusted_pids),
                    metadata={
                        "device": f"cuda:{index}",
                        "lock_path": str(lock_path),
                        "waited_seconds": round(time.monotonic() - started, 3),
                        "samples": samples,
                        "consecutive_idle_samples": idle_streak,
                        "max_utilization": int(max_utilization),
                        "allow_co_resident": bool(allow_co_resident),
                        "trusted_gpu_pids": sorted(trusted_pids),
                        "last_state": last_state,
                    },
                )
            time.sleep(max(0.05, poll_interval))
    except Exception:
        handle.close()
        raise
    handle.close()
    external = last_state.get("compute_processes", [])
    requirement = "idle" if allow_co_resident else "exclusively idle"
    raise RuntimeError(
        f"GPU cuda:{index} did not become {requirement} within {timeout:.1f}s; "
        f"utilization={last_state.get('utilization_gpu', 'unknown')}%, "
        f"compute_processes={external}"
    )


def _device_index(device: str) -> int:
    text = str(device).strip().lower()
    if text.startswith("cuda:"):
        return int(text.split(":", 1)[1])
    if text in {"cuda", ""}:
        return 0
    return int(text)


def _trusted_gpu_pids() -> set[int]:
    trusted = {os.getpid()}
    for value in os.environ.get("KERNELEVO_TRUSTED_GPU_PIDS", "").split(","):
        try:
            trusted.add(int(value.strip()))
        except ValueError:
            continue
    return trusted


def _probe_gpu(index: int) -> dict[str, Any]:
    gpu_rows = _nvidia_smi(
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,pstate,clocks.sm",
    )
    selected = next((row for row in gpu_rows if int(row[0]) == index), None)
    if selected is None:
        raise RuntimeError(f"nvidia-smi did not report GPU index {index}")
    uuid = selected[1]
    process_rows = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        allow_empty=True,
    )
    processes = [
        {
            "pid": int(row[1]),
            "name": row[2],
            "used_memory_mb": _number(row[3]),
        }
        for row in process_rows
        if row and row[0] == uuid
    ]
    return {
        "index": index,
        "uuid": uuid,
        "utilization_gpu": int(_number(selected[2])),
        "memory_used_mb": _number(selected[3]),
        "memory_total_mb": _number(selected[4]),
        "pstate": selected[5],
        "sm_clock_mhz": _number(selected[6]),
        "compute_processes": processes,
    }


def _nvidia_smi(query: str, *, allow_empty: bool = False) -> list[list[str]]:
    completed = subprocess.run(
        ["nvidia-smi", query, "--format=csv,noheader,nounits"],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        if allow_empty and "No running processes" in completed.stderr:
            return []
        raise RuntimeError(f"nvidia-smi query failed: {completed.stderr.strip()}")
    return [
        [item.strip() for item in line.split(",")]
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def _number(value: str) -> float:
    text = str(value).strip()
    if text in {"", "N/A", "[Not Supported]"}:
        return 0.0
    return float(text)
