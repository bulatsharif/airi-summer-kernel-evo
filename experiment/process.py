from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import signal
import subprocess
import sys
from threading import Thread
from typing import Mapping, Sequence


@dataclass(frozen=True)
class StreamingProcessResult:
    exit_code: int
    timed_out: bool


def _stop_process_group(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
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
) -> StreamingProcessResult:
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
        start_new_session=True,
    )
    assert process.stdout is not None

    with log_path.open("w", encoding="utf-8") as log:
        def pump_output() -> None:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    sys.stdout.write(line)
                    sys.stdout.flush()
            except (OSError, ValueError):
                # The parent closes the pipe if a detached descendant keeps it
                # open after the command itself has already exited.
                pass

        reader = Thread(target=pump_output, daemon=True)
        reader.start()
        timed_out = False
        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            marker = f"\nprocess timed out after {timeout:.1f}s\n"
            log.write(marker)
            log.flush()
            sys.stdout.write(marker)
            sys.stdout.flush()
            _stop_process_group(process)
            exit_code = 124
        except KeyboardInterrupt:
            _stop_process_group(process)
            raise
        finally:
            reader.join(timeout=5.0)
            process.stdout.close()
            if reader.is_alive():
                reader.join(timeout=1.0)
    return StreamingProcessResult(exit_code, timed_out)
