"""Operation-aware correctness contracts for CuTe evaluator adapters."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


_TOLERANCES = {
    "fp32": {"atol": 1e-5, "rtol": 1e-5},
    "fp16": {"atol": 1e-2, "rtol": 1e-2},
    "bf16": {"atol": 1e-2, "rtol": 1e-2},
    "fp8": {"atol": 5e-2, "rtol": 5e-2},
}


def build_correctness_contract(
    *,
    operation: str,
    precision: str,
    representative_shapes: Sequence[Sequence[int]] = (),
    tile_shape: Sequence[int] = (),
    supports_strides: bool = False,
    supports_misalignment: bool = False,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a symbolic matrix that an evaluator can instantiate from task inputs."""
    op = str(operation or "elementwise").lower()
    dtype = str(precision or "bf16").lower()
    tolerance = dict(_TOLERANCES.get(dtype, _TOLERANCES["bf16"]))
    tolerance.update(
        {
            str(key): float(value)
            for key, value in (overrides or {}).items()
            if key in {"atol", "rtol"} and isinstance(value, (int, float))
        }
    )
    cases: list[dict[str, Any]] = [
        {
            "id": "tiny",
            "purpose": "Expose indexing, launch, and synchronization assumptions with minimal work.",
            "shape_policy": "smallest supported non-empty shape",
        },
        {
            "id": "representative",
            "purpose": "Protect the actual performance domain.",
            "shapes": [list(shape) for shape in representative_shapes[:8]],
        },
    ]
    if tile_shape:
        tile = [int(value) for value in tile_shape]
        cases.extend(
            (
                {
                    "id": "tile-exact",
                    "purpose": "Exercise the unpredicated/full-tile path.",
                    "shape_policy": {"tile_multiple": tile},
                },
                {
                    "id": "tile-boundaries",
                    "purpose": "Prove residue predicates and stage bounds.",
                    "shape_policy": {"around_each_tile_extent": [-1, 1]},
                },
            )
        )
    else:
        cases.append(
            {
                "id": "ragged",
                "purpose": "Reject hidden divisibility assumptions.",
                "shape_policy": "prime and non-power-of-two extents",
            }
        )
    if supports_strides:
        cases.append(
            {
                "id": "nontrivial-strides",
                "purpose": "Prove the claimed tensor ABI instead of relying on contiguous materialization.",
                "shape_policy": "valid sliced/transposed inputs",
            }
        )
    if supports_misalignment:
        cases.append(
            {
                "id": "misalignment",
                "purpose": "Prove the narrow/residue path for pointers that do not meet the fast-path alignment.",
                "shape_policy": "valid offset views",
            }
        )
    cases.append(
        {
            "id": "numerical-extremes",
            "purpose": "Expose overflow, saturation, NaN/Inf, and reduction-order behavior.",
            "values": ["zeros", "signed small", "large finite", "dtype boundary"],
        }
    )
    if op == "attention":
        cases.append(
            {
                "id": "masking",
                "purpose": "Validate causal/noncausal mask edges and all-masked rows where supported.",
            }
        )
    if dtype == "fp8":
        cases.append(
            {
                "id": "fp8-quantization",
                "purpose": "Validate E4M3/E5M2 decode, scaling, saturation, and cache invalidation.",
                "formats": ["e4m3fn", "e5m2", "mixed"],
            }
        )
    return {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "operation": op,
        "precision": dtype,
        "tolerance": tolerance,
        "required_statistics": [
            "max_abs_error",
            "max_rel_error",
            "mean_abs_error",
            "first_bad_index",
            "nan_count",
            "inf_count",
            "sentinel_corruption",
        ],
        "cases": cases,
        "sanitizer_order": ["memcheck", "racecheck", "initcheck", "synccheck"],
        "note": "KernelEvo's evaluator instantiates these policies; author turns do not run the full matrix.",
    }
