"""Conservative Hopper configuration checks for the validated CuTe DSL corpus."""

from __future__ import annotations

from typing import Any, Sequence


_DTYPE_BYTES = {
    "bf16": 2,
    "fp16": 2,
    "fp8": 1,
    "fp8_e4m3fn": 1,
    "fp8_e5m2": 1,
}


def check_hopper_gemm_config(
    *,
    tile_shape_mnk: Sequence[int],
    cluster_shape_mn: Sequence[int] = (1, 1),
    stages: int = 2,
    dtype: str = "bf16",
    output_dtype: str = "bf16",
    arch: str = "sm_90a",
    shared_memory_limit_bytes: int = 232_448,
) -> dict[str, Any]:
    """Check a proposal against the envelope of the verified Hopper GEMM.

    This is a pre-compilation filter, not a CUDA occupancy oracle. The estimate is
    intentionally explicit so authors can understand why a configuration is rejected.
    """
    tile = tuple(int(value) for value in tile_shape_mnk)
    cluster = tuple(int(value) for value in cluster_shape_mn)
    if len(tile) != 3:
        raise ValueError("tile_shape_mnk must contain M,N,K")
    if len(cluster) != 2:
        raise ValueError("cluster_shape_mn must contain M,N")
    if any(value <= 0 for value in (*tile, *cluster)):
        raise ValueError("tile, cluster, and stage values must be positive")
    if int(stages) <= 0:
        raise ValueError("stages must be positive")

    dtype_name = str(dtype).lower()
    output_name = str(output_dtype).lower()
    if dtype_name not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported corpus dtype: {dtype}")
    if output_name not in _DTYPE_BYTES:
        raise ValueError(f"Unsupported corpus output dtype: {output_dtype}")

    m, n, k = tile
    input_bytes = _DTYPE_BYTES[dtype_name]
    output_bytes = _DTYPE_BYTES[output_name]
    stage_bytes_a = m * k * input_bytes
    stage_bytes_b = n * k * input_bytes
    mainloop_smem = int(stages) * (stage_bytes_a + stage_bytes_b)
    epilogue_smem = m * n * output_bytes
    estimated_smem = mainloop_smem + epilogue_smem
    margin = int(shared_memory_limit_bytes) - estimated_smem

    issues: list[dict[str, str]] = []
    if arch != "sm_90a":
        issues.append(
            {
                "severity": "error",
                "code": "arch",
                "message": "The validated TMA/WGMMA corpus requires `sm_90a`.",
            }
        )
    if m not in {64, 128}:
        issues.append(
            {
                "severity": "error",
                "code": "tile-m",
                "message": "Validated CTA M choices are 64 or 128.",
            }
        )
    if n not in {64, 128, 256}:
        issues.append(
            {
                "severity": "error",
                "code": "tile-n",
                "message": "Validated CTA N choices are 64, 128, or 256.",
            }
        )
    instruction_k = 32 if dtype_name.startswith("fp8") or dtype_name == "fp8" else 16
    if k % instruction_k:
        issues.append(
            {
                "severity": "error",
                "code": "tile-k",
                "message": f"Tile K must be divisible by the {instruction_k}-element WGMMA K.",
            }
        )
    if any(value & (value - 1) for value in cluster) or cluster[0] * cluster[1] > 4:
        issues.append(
            {
                "severity": "error",
                "code": "cluster-shape",
                "message": "Validated cluster modes are powers of two with at most four CTAs.",
            }
        )
    if estimated_smem > int(shared_memory_limit_bytes):
        issues.append(
            {
                "severity": "error",
                "code": "shared-memory-budget",
                "message": f"Estimated {estimated_smem} B exceeds the {shared_memory_limit_bytes} B limit.",
            }
        )
    elif margin < 16_384:
        issues.append(
            {
                "severity": "warning",
                "code": "shared-memory-margin",
                "message": f"Only {margin} B remains for alignment, barriers, and implementation overhead.",
            }
        )

    return {
        "schema_version": 1,
        "dialect": "cute_dsl_python",
        "scope": "verified_hopper_wgmma_gemm_envelope",
        "configuration": {
            "arch": arch,
            "dtype": dtype_name,
            "output_dtype": output_name,
            "tile_shape_mnk": list(tile),
            "cluster_shape_mn": list(cluster),
            "stages": int(stages),
        },
        "instruction": {"wgmma_k": instruction_k},
        "shared_memory": {
            "a_bytes_per_stage": stage_bytes_a,
            "b_bytes_per_stage": stage_bytes_b,
            "mainloop_bytes": mainloop_smem,
            "epilogue_upper_bound_bytes": epilogue_smem,
            "estimated_total_bytes": estimated_smem,
            "limit_bytes": int(shared_memory_limit_bytes),
            "margin_bytes": margin,
        },
        "issues": issues,
        "feasible": not any(item["severity"] == "error" for item in issues),
        "note": (
            "The epilogue estimate is a conservative upper bound. Confirm exact dynamic shared memory, "
            "registers, and active CTAs from the compiled artifact and profiler."
        ),
    }
