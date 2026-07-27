import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05


M = 1024
N = 8192
K = 8192
SEED = 20260727
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
MULTIPLIER = 2.0
NEGATIVE_SLOPE = 0.1
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def fp8_gemm_kernel(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: FP8 tcgen05 GEMM with FP32 accumulator/output.
    pass


@cute.kernel
def multiply_leaky_relu_kernel(output: cute.Tensor, bias: cute.Tensor):
    # TODO: output = leaky_relu((scaled_gemm + bias) * MULTIPLIER).
    pass


@cute.jit
def gemm_multiply_leaky_relu(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: construct/launch GEMM and elementwise kernels.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _harness_torch

import cutlass as _harness_cutlass
import cutlass.cute as _harness_cute
from cutlass.cute.runtime import from_dlpack as _harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _harness_create_cute_tensor_for_fp8,
)


_HARNESS_M = 1024
_HARNESS_N = 8192
_HARNESS_K = 8192
_HARNESS_SEED = 20260727
_HARNESS_FP8_MAX = 448.0
_HARNESS_WEIGHT_BOUND = _HARNESS_K ** -0.5
_HARNESS_SCALE_A = 1.0 / _HARNESS_FP8_MAX
_HARNESS_SCALE_B = _HARNESS_WEIGHT_BOUND / _HARNESS_FP8_MAX
_HARNESS_MULTIPLIER = 2.0
_HARNESS_NEGATIVE_SLOPE = 0.1
_HARNESS_FP8_DTYPE = _harness_cutlass.Float8E4M3FN


def main():
    if not _harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _harness_torch.manual_seed(_HARNESS_SEED)
    source_a = _harness_torch.rand(
        (_HARNESS_M, _HARNESS_K),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    source_b_nk = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_K),
        device="cuda",
        dtype=_harness_torch.float32,
    ).uniform_(-_HARNESS_WEIGHT_BOUND, _HARNESS_WEIGHT_BOUND)
    bias = _harness_torch.empty(
        (_HARNESS_N,), device="cuda", dtype=_harness_torch.float32
    ).uniform_(-_HARNESS_WEIGHT_BOUND, _HARNESS_WEIGHT_BOUND)
    storage_a = _harness_torch.empty(
        (_HARNESS_M, _HARNESS_K), device="cuda", dtype=_harness_torch.uint8
    )
    storage_b = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_K), device="cuda", dtype=_harness_torch.uint8
    )
    output = _harness_torch.empty(
        (_HARNESS_M, _HARNESS_N), device="cuda", dtype=_harness_torch.float32
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
        source_b_nk * (_HARNESS_FP8_MAX / _HARNESS_WEIGHT_BOUND),
    )
    bias_tensor = _harness_from_dlpack(bias)
    output_tensor = _harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _harness_cute.compile(
        gemm_multiply_leaky_relu,
        matrix_a,
        matrix_b_nk,
        bias_tensor,
        output_tensor,
    )
    compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)

    a_fp8 = storage_a.view(_harness_torch.float8_e4m3fn)
    b_fp8 = storage_b.view(_harness_torch.float8_e4m3fn)
    scale_a = _harness_torch.tensor(
        _HARNESS_SCALE_A, device="cuda", dtype=_harness_torch.float32
    )
    scale_b = _harness_torch.tensor(
        _HARNESS_SCALE_B, device="cuda", dtype=_harness_torch.float32
    )
    fp8_linear = _harness_torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=_harness_torch.float32,
    ) + bias
    reference = _harness_torch.nn.functional.leaky_relu(
        fp8_linear * _HARNESS_MULTIPLIER,
        negative_slope=_HARNESS_NEGATIVE_SLOPE,
    )
    full_max_abs = (output - reference).abs().max().item()

    rows = _harness_torch.tensor(
        [0, 1, 7, 31, 127, 255, 511, 1023], device="cuda"
    )
    columns = _harness_torch.tensor(
        [0, 3, 31, 255, 1023, 2047, 4095, 8191], device="cuda"
    )
    fp32_linear = (
        source_a.index_select(0, rows)
        @ source_b_nk.index_select(0, columns).t()
        + bias.index_select(0, columns)
    )
    fp32_reference = _harness_torch.nn.functional.leaky_relu(
        fp32_linear * _HARNESS_MULTIPLIER,
        negative_slope=_HARNESS_NEGATIVE_SLOPE,
    )
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()

    if (
        not _harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.01
        or sample_max_abs > 0.2
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, "
            f"sample_abs={sample_max_abs:.6f}"
        )

    print(
        "task=level2_12_gemm_multiply_leaky_relu "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} PASS"
    )
    _harness_torch.cuda.synchronize()


main()
