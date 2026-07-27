import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05


M = 1024
N = 8192
K = 8192
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
DIVISOR = 2.0
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def fp8_gemm_kernel(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: FP8 tcgen05 GEMM with FP32 accumulation and output.
    pass


@cute.kernel
def relu_divide_kernel(output: cute.Tensor, bias: cute.Tensor):
    # TODO: apply the declared elementwise operations in exact order.
    pass


@cute.jit
def gemm_relu_divide(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: construct and launch GEMM, then launch the epilogue.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _cute_harness_torch

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import from_dlpack as _cute_harness_from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8 as _cute_harness_create_fp8


_CUTE_HARNESS_M = 1024
_CUTE_HARNESS_N = 8192
_CUTE_HARNESS_K = 8192
_CUTE_HARNESS_FP8_MAX = 448.0
_CUTE_HARNESS_WEIGHT_BOUND = _CUTE_HARNESS_K ** -0.5
_CUTE_HARNESS_SCALE_A = 1.0 / _CUTE_HARNESS_FP8_MAX
_CUTE_HARNESS_SCALE_B = _CUTE_HARNESS_WEIGHT_BOUND / _CUTE_HARNESS_FP8_MAX
_CUTE_HARNESS_FP8_DTYPE = _cute_harness_cutlass.Float8E4M3FN


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")
    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    source_a = _cute_harness_torch.rand(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_K), device="cuda", dtype=_cute_harness_torch.float32
    )
    source_b_nk = _cute_harness_torch.empty(
        (_CUTE_HARNESS_N, _CUTE_HARNESS_K), device="cuda", dtype=_cute_harness_torch.float32
    ).uniform_(-_CUTE_HARNESS_WEIGHT_BOUND, _CUTE_HARNESS_WEIGHT_BOUND)
    bias = _cute_harness_torch.randn(
        (_CUTE_HARNESS_N,), device="cuda", dtype=_cute_harness_torch.float32
    )
    storage_a = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_K), device="cuda", dtype=_cute_harness_torch.uint8
    )
    storage_b = _cute_harness_torch.empty(
        (_CUTE_HARNESS_N, _CUTE_HARNESS_K), device="cuda", dtype=_cute_harness_torch.uint8
    )
    output = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_N), device="cuda", dtype=_cute_harness_torch.float32
    )
    matrix_a = _cute_harness_create_fp8(
        storage_a, _CUTE_HARNESS_FP8_DTYPE, 1, source_a * _CUTE_HARNESS_FP8_MAX
    )
    matrix_b_nk = _cute_harness_create_fp8(
        storage_b,
        _CUTE_HARNESS_FP8_DTYPE,
        1,
        source_b_nk * (_CUTE_HARNESS_FP8_MAX / _CUTE_HARNESS_WEIGHT_BOUND),
    )
    bias_tensor = _cute_harness_from_dlpack(bias)
    output_tensor = _cute_harness_from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = _cute_harness_cute.compile(
        gemm_relu_divide, matrix_a, matrix_b_nk, bias_tensor, output_tensor
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)
    _cute_harness_torch.cuda.synchronize()
    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]
    a_fp8 = storage_a.view(_cute_harness_torch.float8_e4m3fn)
    b_fp8 = storage_b.view(_cute_harness_torch.float8_e4m3fn)
    scale_a = _cute_harness_torch.tensor(
        _CUTE_HARNESS_SCALE_A, device="cuda", dtype=_cute_harness_torch.float32
    )
    scale_b = _cute_harness_torch.tensor(
        _CUTE_HARNESS_SCALE_B, device="cuda", dtype=_cute_harness_torch.float32
    )
    linear = _cute_harness_torch._scaled_mm(
        a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b,
        out_dtype=_cute_harness_torch.float32,
    ) + bias
    reference = _cute_harness_torch.relu(linear) / 2.0
    full_max_abs = (output - reference).abs().max().item()
    rows = _cute_harness_torch.tensor([0, 1, 7, 31, 127, 255, 511, 1023], device="cuda")
    columns = _cute_harness_torch.tensor(
        [0, 3, 31, 255, 1023, 2047, 4095, 8191], device="cuda"
    )
    linear = (
        source_a.index_select(0, rows)
        @ source_b_nk.index_select(0, columns).t()
        + bias.index_select(0, columns)
    )
    fp32_reference = _cute_harness_torch.relu(linear) / 2.0
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()
    if (
        not _cute_harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.01
        or sample_max_abs > 0.1
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, sample_abs={sample_max_abs:.6f}"
        )
    print(
        "task=level2_63_gemm_relu_divide "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
