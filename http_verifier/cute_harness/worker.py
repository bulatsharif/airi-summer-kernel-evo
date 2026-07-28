from __future__ import annotations

import argparse
import json
import runpy
import sys
import traceback
from pathlib import Path


def _device_time_ms(profiler: object) -> float:
    total_us = 0.0
    for event in profiler.key_averages():  # type: ignore[attr-defined]
        value = getattr(event, "self_device_time_total", None)
        if value is None:
            value = getattr(event, "self_cuda_time_total", 0.0)
        total_us += float(value or 0.0)
    return total_us / 1000.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--result", required=True)
    parser.add_argument("--trace")
    parser.add_argument("--no-torch-profile", action="store_true")
    args = parser.parse_args()

    result = {"success": False, "device_time_ms": None}
    exit_code = 0
    try:
        # The uploaded file is a complete program. Do not leak this worker's
        # --source/--result arguments into its argparse-based main section.
        sys.argv = [args.source]
        if args.no_torch_profile:
            runpy.run_path(args.source, run_name="__main__")
        else:
            import torch

            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                runpy.run_path(args.source, run_name="__main__")
            result["device_time_ms"] = _device_time_ms(prof)
            if args.trace:
                prof.export_chrome_trace(args.trace)
        result["success"] = True
    except BaseException:
        traceback.print_exc(file=sys.stderr)
        exit_code = 1
    finally:
        Path(args.result).write_text(json.dumps(result), encoding="utf-8")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
