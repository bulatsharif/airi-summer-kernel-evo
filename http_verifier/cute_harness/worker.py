from __future__ import annotations

import argparse
import json
import os
import runpy
import statistics
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _device_time_ms(profiler: object) -> float:
    total_us = 0.0
    for event in profiler.key_averages():  # type: ignore[attr-defined]
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        total_us += float(value or 0.0)
    return total_us / 1000.0


def _run_job(
    source: str,
    result_path: str,
    iterations: int,
    trace: str | None,
    no_torch_profile: bool = False,
) -> int:
    result: dict[str, object] = {
        "success": False,
        "device_time_ms": None,
        "device_times_ms": [],
        "exit_code": 1,
    }
    exit_code = 0
    try:
        # The uploaded file is a complete program. Do not leak this worker's
        # --source/--result arguments into its argparse-based main section.
        sys.argv = [source]
        if no_torch_profile:
            runpy.run_path(source, run_name="__main__")
        else:
            import torch

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            # Compile/import/allocation work happens during this unmeasured
            # warmup. CUDA synchronization keeps it out of measured iterations.
            runpy.run_path(source, run_name="__main__")
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            device_times: list[float] = []
            for iteration in range(iterations):
                with torch.profiler.profile(activities=activities) as prof:
                    runpy.run_path(source, run_name="__main__")
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                device_times.append(_device_time_ms(prof))
                if trace and iteration == iterations - 1:
                    prof.export_chrome_trace(trace)
            result["device_times_ms"] = device_times
            result["device_time_ms"] = statistics.median(device_times)
        result["success"] = True
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    finally:
        result["exit_code"] = exit_code
        destination = Path(result_path)
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8")
        os.replace(temporary, destination)
    return exit_code


@contextmanager
def _redirect_fds(stdout_path: str, stderr_path: str) -> Iterator[None]:
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    try:
        with open(stdout_path, "wb") as stdout_file, open(
            stderr_path, "wb"
        ) as stderr_file:
            os.dup2(stdout_file.fileno(), 1)
            os.dup2(stderr_file.fileno(), 2)
            yield
            sys.stdout.flush()
            sys.stderr.flush()
    finally:
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def _persistent_main() -> int:
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with _redirect_fds(request["stdout"], request["stderr"]):
                _run_job(
                    request["source"],
                    request["result"],
                    int(request["iterations"]),
                    request.get("trace"),
                )
        except BaseException:
            # A malformed control message is a server bug. Stop so the parent
            # notices and replaces this worker instead of accepting more jobs.
            traceback.print_exc(file=sys.stderr)
            return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source")
    parser.add_argument("--result")
    parser.add_argument("--trace")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--no-torch-profile", action="store_true")
    parser.add_argument("--persistent", action="store_true")
    args = parser.parse_args()
    if args.persistent:
        return _persistent_main()
    if not args.source or not args.result:
        parser.error("--source and --result are required")
    return _run_job(
        args.source,
        args.result,
        args.iterations,
        args.trace,
        args.no_torch_profile,
    )


if __name__ == "__main__":
    raise SystemExit(main())
