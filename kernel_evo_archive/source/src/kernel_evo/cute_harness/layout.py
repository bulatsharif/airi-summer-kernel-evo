"""Small deterministic host-side layout probes for CuTe coordinate reasoning."""

from __future__ import annotations

import itertools
import json
import math
import subprocess
import sys
from typing import Iterable, Sequence


def compact_stride(shape: Sequence[int], *, order: Sequence[int] | None = None) -> tuple[int, ...]:
    dims = tuple(int(value) for value in shape)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("shape dimensions must be positive")
    traversal = tuple(order) if order is not None else tuple(reversed(range(len(dims))))
    if sorted(traversal) != list(range(len(dims))):
        raise ValueError("order must be a permutation of shape dimensions")
    stride = [0] * len(dims)
    running = 1
    for mode in traversal:
        stride[mode] = running
        running *= dims[mode]
    return tuple(stride)


def linear_index(coord: Sequence[int], shape: Sequence[int], stride: Sequence[int]) -> int:
    if not (len(coord) == len(shape) == len(stride)):
        raise ValueError("coord, shape, and stride must have the same rank")
    if any(index < 0 or index >= extent for index, extent in zip(coord, shape, strict=True)):
        raise ValueError(f"coordinate {tuple(coord)} is outside shape {tuple(shape)}")
    return sum(index * step for index, step in zip(coord, stride, strict=True))


def probe_layout(
    shape: Sequence[int],
    *,
    stride: Sequence[int] | None = None,
    order: Sequence[int] | None = None,
    coordinates: Iterable[Sequence[int]] = (),
    max_table_entries: int = 32,
) -> dict[str, object]:
    dims = tuple(int(value) for value in shape)
    steps = tuple(int(value) for value in stride) if stride is not None else compact_stride(dims, order=order)
    if len(dims) != len(steps):
        raise ValueError("shape and stride must have the same rank")
    coords = [tuple(int(value) for value in coord) for coord in coordinates]
    if not coords:
        coords = list(itertools.islice(itertools.product(*(range(value) for value in dims)), max_table_entries))
    mapping = [{"coordinate": list(coord), "linear_index": linear_index(coord, dims, steps)} for coord in coords]
    max_index = max(
        linear_index(tuple(value - 1 for value in dims), dims, steps),
        max((item["linear_index"] for item in mapping), default=0),
    )
    return {
        "shape": list(dims),
        "stride": list(steps),
        "rank": len(dims),
        "size": math.prod(dims),
        "cosize": max_index + 1,
        "injective_in_sample": len({item["linear_index"] for item in mapping}) == len(mapping),
        "mapping": mapping,
    }


def probe_cute_layout(
    shape: Sequence[int],
    *,
    stride: Sequence[int] | None = None,
    coordinate: Sequence[int] | None = None,
    tile: Sequence[int] | None = None,
    timeout: float = 60.0,
) -> dict[str, object]:
    """Compile an isolated rank-2 probe with the installed Python CuTe DSL.

    The ordinary mapping remains useful structured output; the DSL trace proves
    that the same layout and transformations are accepted by the pinned package.
    """
    dims = tuple(int(value) for value in shape)
    if len(dims) != 2:
        raise ValueError("The CuTe DSL layout trace currently supports rank-2 layouts")
    steps = tuple(int(value) for value in stride) if stride is not None else compact_stride(dims)
    coord = tuple(int(value) for value in (coordinate or (0, 0)))
    tiler = tuple(int(value) for value in (tile or (0, 0)))
    if not all(len(value) == 2 for value in (steps, coord, tiler)):
        raise ValueError("stride, coordinate, and tile must have rank 2")
    command = [
        sys.executable,
        "-m",
        "kernel_evo.cute_harness.dsl_layout_probe",
        "--shape",
        ",".join(str(value) for value in dims),
        "--stride",
        ",".join(str(value) for value in steps),
        "--coord",
        ",".join(str(value) for value in coord),
        "--tile",
        ",".join(str(value) for value in tiler),
    ]
    completed = subprocess.run(command, text=True, capture_output=True, timeout=timeout, check=False)
    payload: dict[str, object]
    try:
        value = json.loads(completed.stdout.strip().splitlines()[-1])
        payload = dict(value) if isinstance(value, dict) else {}
    except (IndexError, json.JSONDecodeError):
        payload = {}
    payload.setdefault("success", False)
    payload.setdefault("error", (completed.stderr or completed.stdout)[-4_000:])
    payload["returncode"] = completed.returncode
    payload["host_mapping"] = probe_layout(dims, stride=steps, coordinates=[coord])
    payload["tile"] = list(tiler)
    return payload
