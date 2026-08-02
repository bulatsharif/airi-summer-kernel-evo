"""GPU-idleness and contamination guard for privileged NCU jobs."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardedProcessResult:
    returncode: int
    stdout: str
    stderr: str
    contaminated: bool
    contaminating_pids: tuple[int, ...]
    timed_out: bool


class NcuGpuGuard:
    def __init__(
        self,
        devices: set[int],
        *,
        max_utilization: int = 5,
        idle_samples: int = 3,
        poll_seconds: float = 1.0,
    ) -> None:
        if not devices:
            raise ValueError("at least one GPU must be allowed")
        self.devices = set(devices)
        self.max_utilization = max_utilization
        self.idle_samples = idle_samples
        self.poll_seconds = poll_seconds

    def select_idle_gpu(self, *, timeout: float) -> int:
        deadline = time.monotonic() + timeout
        consecutive: dict[int, int] = {device: 0 for device in self.devices}
        last_memory: dict[int, int] = {device: 1 << 60 for device in self.devices}
        while time.monotonic() < deadline:
            states = self._gpu_states()
            for device in self.devices:
                state = states.get(device)
                if state is None or state["utilization"] > self.max_utilization:
                    consecutive[device] = 0
                    continue
                consecutive[device] += 1
                last_memory[device] = state["memory_used"]
            eligible = [
                device
                for device in self.devices
                if consecutive[device] >= self.idle_samples
            ]
            if eligible:
                return min(eligible, key=lambda device: (last_memory[device], device))
            time.sleep(self.poll_seconds)
        raise TimeoutError(
            f"no allowed GPU stayed below {self.max_utilization}% utilization "
            f"for {self.idle_samples} samples"
        )

    def run_monitored(
        self,
        command: list[str],
        *,
        device: int,
        timeout: float,
        env: dict[str, str],
    ) -> GuardedProcessResult:
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + timeout
        contaminating: set[int] = set()
        timed_out = False
        while process.poll() is None:
            if time.monotonic() >= deadline:
                timed_out = True
                self._terminate_group(process)
                break
            descendants = self._descendants(process.pid)
            descendants.add(process.pid)
            contaminating.update(
                pid
                for pid, sm_util in self._process_utilization(device).items()
                if pid not in descendants and sm_util > self.max_utilization
            )
            if contaminating:
                self._terminate_group(process)
                break
            time.sleep(self.poll_seconds)
        stdout, stderr = process.communicate()
        return GuardedProcessResult(
            returncode=int(process.returncode or 0),
            stdout=stdout,
            stderr=stderr,
            contaminated=bool(contaminating),
            contaminating_pids=tuple(sorted(contaminating)),
            timed_out=timed_out,
        )

    def _gpu_states(self) -> dict[int, dict[str, int]]:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            timeout=10,
            check=True,
        )
        states: dict[int, dict[str, int]] = {}
        for line in proc.stdout.splitlines():
            fields = [field.strip() for field in line.split(",")]
            if len(fields) != 3:
                continue
            try:
                index, utilization, memory_used = map(int, fields)
            except ValueError:
                continue
            states[index] = {
                "utilization": utilization,
                "memory_used": memory_used,
            }
        return states

    def _process_utilization(self, device: int) -> dict[int, int]:
        proc = subprocess.run(
            ["nvidia-smi", "pmon", "-i", str(device), "-c", "1", "-s", "u"],
            text=True,
            capture_output=True,
            timeout=10,
        )
        if proc.returncode != 0:
            return {}
        usage: dict[int, int] = {}
        for line in proc.stdout.splitlines():
            text = line.strip()
            if not text or text.startswith("#"):
                continue
            fields = text.split()
            if len(fields) < 4:
                continue
            try:
                gpu_index = int(fields[0])
                pid = int(fields[1])
                sm_util = int(fields[3]) if fields[3] != "-" else 0
            except ValueError:
                continue
            if gpu_index == device and pid > 0:
                usage[pid] = sm_util
        return usage

    @staticmethod
    def _descendants(root_pid: int) -> set[int]:
        parents: dict[int, int] = {}
        for entry in os.scandir("/proc"):
            if not entry.name.isdigit():
                continue
            try:
                fields = open(
                    f"/proc/{entry.name}/stat", encoding="utf-8"
                ).read().split()
                parents[int(entry.name)] = int(fields[3])
            except (OSError, ValueError, IndexError):
                continue
        descendants: set[int] = set()
        changed = True
        while changed:
            changed = False
            for pid, parent in parents.items():
                if pid in descendants:
                    continue
                if parent == root_pid or parent in descendants:
                    descendants.add(pid)
                    changed = True
        return descendants

    @staticmethod
    def _terminate_group(process: subprocess.Popen[str]) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
