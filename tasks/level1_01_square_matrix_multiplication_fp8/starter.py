import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05


N = 4096
SEED = 20260726
FP8_MAX = 448.0
INPUT_SCALE = 1.0 / FP8_MAX
OUTPUT_SCALE = INPUT_SCALE * INPUT_SCALE
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def square_gemm_kernel(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: TMA/SMEM pipeline, FP8 tcgen05 MMA and FP32 epilogue.
    pass


@cute.jit
def square_gemm(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: construct layouts/MMA atoms and launch square_gemm_kernel.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _harness_torch

import cutlass as _harness_cutlass
import cutlass.cute as _harness_cute
from cutlass.cute.runtime import from_dlpack as _harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _harness_create_cute_tensor_for_fp8,
)


_HARNESS_N = 4096
_HARNESS_SEED = 20260726
_HARNESS_FP8_MAX = 448.0
_HARNESS_INPUT_SCALE = 1.0 / _HARNESS_FP8_MAX
_HARNESS_FP8_DTYPE = _harness_cutlass.Float8E4M3FN


def main():
    if not _harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _harness_torch.manual_seed(_HARNESS_SEED)
    source_a = _harness_torch.rand(
        (_HARNESS_N, _HARNESS_N),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    source_b_nk = _harness_torch.rand(
        (_HARNESS_N, _HARNESS_N),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    storage_a = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_N),
        device="cuda",
        dtype=_harness_torch.uint8,
    )
    storage_b = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_N),
        device="cuda",
        dtype=_harness_torch.uint8,
    )
    output = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_N),
        device="cuda",
        dtype=_harness_torch.float32,
    )

    matrix_a = _harness_create_cute_tensor_for_fp8(
        storage_a,
        _HARNESS_FP8_DTYPE,
        1,
        source_a * _HARNESS_FP8_MAX,
    )
    matrix_b_nk = _harness_create_cute_tensor_for_fp8(
        storage_b,
        _HARNESS_FP8_DTYPE,
        1,
        source_b_nk * _HARNESS_FP8_MAX,
    )
    output_tensor = _harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _harness_cute.compile(
        square_gemm,
        matrix_a,
        matrix_b_nk,
        output_tensor,
    )
    compiled(matrix_a, matrix_b_nk, output_tensor)

    a_fp8 = storage_a.view(_harness_torch.float8_e4m3fn)
    b_fp8 = storage_b.view(_harness_torch.float8_e4m3fn)
    scale = _harness_torch.tensor(
        _HARNESS_INPUT_SCALE,
        device="cuda",
        dtype=_harness_torch.float32,
    )
    reference = _harness_torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        scale_a=scale,
        scale_b=scale,
        out_dtype=_harness_torch.float32,
    )
    full_max_abs = (output - reference).abs().max().item()

    rows = _harness_torch.tensor(
        [0, 1, 17, 255, 1023, 2047, 3071, 4095],
        device="cuda",
    )
    columns = _harness_torch.tensor(
        [4095, 2048, 511, 7, 3072, 1536, 63, 0],
        device="cuda",
    )
    fp32_reference = (
        source_a.index_select(0, rows)
        * source_b_nk.index_select(0, columns)
    ).sum(dim=1)
    actual = output[rows, columns]
    sample_max_rel = (
        (actual - fp32_reference).abs()
        / fp32_reference.abs().clamp_min(1.0e-6)
    ).max().item()

    if (
        not _harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.125
        or sample_max_rel > 0.01
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, "
            f"sample_rel={sample_max_rel:.6f}"
        )

    print(
        "task=level1_01_square_matrix_multiplication "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_rel={sample_max_rel:.6f} PASS"
    )
    _harness_torch.cuda.synchronize()


main()
