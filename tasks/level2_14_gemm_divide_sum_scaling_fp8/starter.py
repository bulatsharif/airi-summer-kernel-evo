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
SCALING_FACTOR = 1.5
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def fp8_gemm_kernel(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    scratch: cute.Tensor,
):
    # TODO: FP8 tcgen05 GEMM with FP32 accumulation and scratch output.
    pass


@cute.kernel
def divide_sum_scale_kernel(scratch: cute.Tensor, output: cute.Tensor):
    # TODO: one-warp FP32 row reduction into output[row, 0].
    pass


@cute.jit
def gemm_divide_sum_scaling(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    scratch: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: launch GEMM then the row reduction.
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
    storage_a = _cute_harness_torch.empty_like(source_a, dtype=_cute_harness_torch.uint8)
    storage_b = _cute_harness_torch.empty_like(source_b_nk, dtype=_cute_harness_torch.uint8)
    scratch = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_N), device="cuda", dtype=_cute_harness_torch.float32
    )
    output = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, 1), device="cuda", dtype=_cute_harness_torch.float32
    )
    matrix_a = _cute_harness_create_fp8(
        storage_a, _CUTE_HARNESS_FP8_DTYPE, 1, source_a * _CUTE_HARNESS_FP8_MAX
    )
    matrix_b_nk = _cute_harness_create_fp8(
        storage_b, _CUTE_HARNESS_FP8_DTYPE, 1,
        source_b_nk * (_CUTE_HARNESS_FP8_MAX / _CUTE_HARNESS_WEIGHT_BOUND),
    )
    scratch_tensor = _cute_harness_from_dlpack(scratch).mark_layout_dynamic(leading_dim=1)
    output_tensor = _cute_harness_from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = _cute_harness_cute.compile(
        gemm_divide_sum_scaling, matrix_a, matrix_b_nk, scratch_tensor, output_tensor
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(matrix_a, matrix_b_nk, scratch_tensor, output_tensor)
    _cute_harness_torch.cuda.synchronize()
    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(matrix_a, matrix_b_nk, scratch_tensor, output_tensor)
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
    reference = (
        _cute_harness_torch._scaled_mm(
            a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b,
            out_dtype=_cute_harness_torch.float32,
        ) / 2.0
    ).sum(dim=1, keepdim=True) * 1.5
    full_max_abs = (output - reference).abs().max().item()
    rows = _cute_harness_torch.tensor([0, 1, 7, 31, 127, 255, 511, 1023], device="cuda")
    fp32_reference = (
        (source_a.index_select(0, rows) @ source_b_nk.t()) / 2.0
    ).sum(dim=1, keepdim=True) * 1.5
    sample_max_abs = (output.index_select(0, rows) - fp32_reference).abs().max().item()
    if (
        not _cute_harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.05
        or sample_max_abs > 3.0
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, sample_abs={sample_max_abs:.6f}"
        )
    print(
        "task=level2_14_gemm_divide_sum_scaling "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
