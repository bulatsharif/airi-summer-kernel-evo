"""Performance measurement modes used by KernelEvo evaluation."""

from __future__ import annotations

from typing import Any, Callable

import torch
from torch.profiler import ProfilerActivity, profile


MEASUREMENT_MODES = ("wall-clock", "device-time")


def validate_measurement_mode(mode: str) -> str:
    """Return a supported measurement mode or raise a concise error."""
    normalized = str(mode).strip().lower()
    if normalized not in MEASUREMENT_MODES:
        choices = ", ".join(MEASUREMENT_MODES)
        raise ValueError(f"measurement_mode must be one of: {choices}")
    return normalized


def _event_self_device_time_us(event: Any) -> float:
    for attribute in (
        "self_device_time_total",
        "self_cuda_time_total",
        "self_privateuse1_time_total",
    ):
        value = getattr(event, attribute, None)
        if value is not None:
            return float(value or 0.0)
    return 0.0


def device_activity_time_ms(profiler: Any) -> float:
    """Sum CUDA kernel and memcpy activity recorded by a Torch profiler."""
    total_us = sum(_event_self_device_time_us(event) for event in profiler.key_averages())
    if total_us <= 0:
        raise RuntimeError(
            "device-time measurement observed no CUDA kernel or memcpy activity"
        )
    return total_us / 1_000.0


def time_execution_with_device_time(
    kernel_fn: Callable[..., Any],
    args: list[Any],
    num_warmup: int = 3,
    num_trials: int = 10,
    discard_first: int = 1,
    verbose: bool = True,
    device: torch.device | int | str | None = None,
) -> list[float]:
    """Measure active CUDA work only, excluding host dispatch gaps.

    All measured invocations share one profiler session. The session's summed
    CUDA kernel and memcpy activity is divided by the number of invocations;
    profiler setup, Python execution, graph launch overhead, and idle gaps
    between CUDA operations are deliberately excluded. Repeating the average in
    the returned sample list preserves the timing-function interface without
    paying profiler startup cost once per trial.
    """
    if device is None:
        device = torch.cuda.current_device()

    with torch.cuda.device(device):
        prepare_for_timing = getattr(kernel_fn, "prepare_for_timing", None)
        if callable(prepare_for_timing):
            prepare_for_timing(*args)
        for _ in range(num_warmup):
            kernel_fn(*args)
        torch.cuda.synchronize(device=device)
        torch.cuda.empty_cache()

        if verbose:
            print(
                f"[Profiling] Using device-time measurement on {device} "
                f"{torch.cuda.get_device_name(device)}, warm up {num_warmup}, "
                f"trials {num_trials}"
            )

        for _ in range(discard_first):
            kernel_fn(*args)
        torch.cuda.synchronize(device=device)

        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for _ in range(num_trials):
                kernel_fn(*args)
            torch.cuda.synchronize(device=device)

        elapsed_ms = device_activity_time_ms(prof) / num_trials
        if verbose:
            print(
                f"Trials 1-{num_trials}: {elapsed_ms:.3g} ms mean active device time "
                "(one profiler session)"
            )

    return [elapsed_ms] * num_trials


def get_measurement_function(mode: str, timing_method: str) -> Callable[..., list[float]]:
    """Resolve the configured objective to its concrete timing function."""
    normalized = validate_measurement_mode(mode)
    if normalized == "device-time":
        return time_execution_with_device_time

    from kernelbench import timing

    return timing.get_timing_function(timing_method)
