from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Lock, Thread
import time
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StreamingProcessResult:
    exit_code: int
    timed_out: bool


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(
            [
                "taskkill",
                "/PID",
                str(process.pid),
                "/T",
                "/F",
            ],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def run_streaming(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    log_path: Path,
    timeout: float,
    heartbeat_interval: float | None = 30.0,
) -> StreamingProcessResult:
    if timeout <= 0:
        raise ValueError("process timeout must be positive")
    if heartbeat_interval is not None and heartbeat_interval <= 0:
        raise ValueError("heartbeat interval must be positive")

    creationflags = (
        subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    )
    process = subprocess.Popen(
        list(command),
        cwd=cwd,
        env=dict(environment),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        errors="replace",
        bufsize=1,
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    assert process.stdout is not None

    with log_path.open("w", encoding="utf-8") as log:
        output_lock = Lock()
        started = time.monotonic()
        last_child_output = started

        def emit(line: str) -> None:
            with output_lock:
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()

        def pump_output() -> None:
            nonlocal last_child_output
            try:
                for line in process.stdout:
                    last_child_output = time.monotonic()
                    emit(line)
            except (OSError, ValueError):
                # The parent closes the pipe if a detached descendant keeps it
                # open after the command itself has already exited.
                pass

        reader = Thread(target=pump_output, daemon=True)
        reader.start()
        timed_out = False
        deadline = started + timeout
        next_heartbeat = (
            started + heartbeat_interval
            if heartbeat_interval is not None
            else deadline
        )
        try:
            while True:
                now = time.monotonic()
                remaining = deadline - now
                if remaining <= 0:
                    timed_out = True
                    emit(f"\nprocess timed out after {timeout:.1f}s\n")
                    _stop_process_group(process)
                    exit_code = 124
                    break
                wait_seconds = min(1.0, remaining)
                if heartbeat_interval is not None:
                    wait_seconds = min(
                        wait_seconds,
                        max(0.01, next_heartbeat - now),
                    )
                try:
                    exit_code = process.wait(timeout=wait_seconds)
                    break
                except subprocess.TimeoutExpired:
                    now = time.monotonic()
                    if (
                        heartbeat_interval is not None
                        and now >= next_heartbeat
                        and now - last_child_output >= heartbeat_interval
                    ):
                        elapsed = now - started
                        silent = now - last_child_output
                        emit(
                            "[experiment] still running "
                            f"({elapsed:.0f}s elapsed; "
                            f"no output for {silent:.0f}s)\n"
                        )
                        next_heartbeat = now + heartbeat_interval
        except KeyboardInterrupt:
            _stop_process_group(process)
            raise
        finally:
            reader.join(timeout=5.0)
            process.stdout.close()
            if reader.is_alive():
                reader.join(timeout=1.0)
    return StreamingProcessResult(exit_code, timed_out)
