"""Isolated CuTe DSL compilation used by the public layout probe."""

from __future__ import annotations

import argparse
import io
import json
from contextlib import redirect_stderr, redirect_stdout
from typing import Sequence

import cutlass
import cutlass.cute as cute


@cute.jit
def _trace_layout(
    m: cutlass.Constexpr,
    n: cutlass.Constexpr,
    stride_m: cutlass.Constexpr,
    stride_n: cutlass.Constexpr,
    coord_m: cutlass.Constexpr,
    coord_n: cutlass.Constexpr,
    tile_m: cutlass.Constexpr,
    tile_n: cutlass.Constexpr,
):
    layout = cute.make_layout((m, n), stride=(stride_m, stride_n))
    print("layout:", layout)
    print("shape:", layout.shape, "stride:", layout.stride)
    print("rank:", cute.rank(layout), "size:", cute.size(layout), "cosize:", cute.cosize(layout))
    print("coordinate:", (coord_m, coord_n), "index:", cute.crd2idx((coord_m, coord_n), layout))
    print("coalesced:", cute.coalesce(layout))
    if tile_m > 0 and tile_n > 0:
        print("logical_divide:", cute.logical_divide(layout, (tile_m, tile_n)))


def _ints(text: str) -> tuple[int, ...]:
    return tuple(int(item.strip()) for item in text.split(",") if item.strip())


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", required=True)
    parser.add_argument("--stride", required=True)
    parser.add_argument("--coord", default="0,0")
    parser.add_argument("--tile", default="0,0")
    args = parser.parse_args(argv)
    shape, stride, coord, tile = (_ints(value) for value in (args.shape, args.stride, args.coord, args.tile))
    if not all(len(value) == 2 for value in (shape, stride, coord, tile)):
        raise SystemExit("The isolated DSL probe currently accepts rank-2 shape/stride/coord/tile values")

    capture = io.StringIO()
    success = True
    error = ""
    try:
        with redirect_stdout(capture), redirect_stderr(capture):
            _trace_layout(*shape, *stride, *coord, *tile)
    except Exception as exc:
        success = False
        error = f"{type(exc).__name__}: {exc}"
    print(
        json.dumps(
            {
                "success": success,
                "dialect": "cute_dsl_python",
                "trace": capture.getvalue().strip()[-12_000:],
                "error": error,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
