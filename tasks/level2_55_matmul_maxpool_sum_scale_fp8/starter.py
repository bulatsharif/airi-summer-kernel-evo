import cutlass
import cutlass.cute as cute


@cute.kernel
def fp8_gemm_kernel(matrix_a: cute.Tensor, matrix_b_nk: cute.Tensor, scratch: cute.Tensor):
    # TODO: FP8 GEMM with FP32 accumulation and correctly restored scales.
    pass


@cute.kernel
def maxpool_sum_scale_kernel(scratch: cute.Tensor, bias: cute.Tensor, output: cute.Tensor):
    # TODO: pairwise max-pool, row sum, and multiply by 0.5.
    pass


@cute.jit
def matmul_maxpool_sum_scale(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    scratch: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: launch the GEMM and reduction kernels.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _h_torch
import torch.nn.functional as _h_functional
import cutlass as _h_cutlass
import cutlass.cute as _h_cute
from cutlass.cute.runtime import from_dlpack as _h_from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8 as _h_create_fp8


_H_M, _H_N, _H_K = 128, 32768, 32768
_H_FP8_MAX = 448.0
_H_WEIGHT_BOUND = _H_K ** -0.5
_H_DTYPE = _h_cutlass.Float8E4M3FN


def main():
    if not _h_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")
    _h_torch.manual_seed(_CUTE_HARNESS_SEED)
    source_a = _h_torch.rand((_H_M, _H_K), device="cuda", dtype=_h_torch.float32)
    source_b = _h_torch.empty((_H_N, _H_K), device="cuda", dtype=_h_torch.float32).uniform_(
        -_H_WEIGHT_BOUND, _H_WEIGHT_BOUND
    )
    bias = _h_torch.randn((_H_N,), device="cuda", dtype=_h_torch.float32)
    storage_a = _h_torch.empty((_H_M, _H_K), device="cuda", dtype=_h_torch.uint8)
    storage_b = _h_torch.empty((_H_N, _H_K), device="cuda", dtype=_h_torch.uint8)
    scratch = _h_torch.empty((_H_M, _H_N), device="cuda", dtype=_h_torch.float32)
    output = _h_torch.empty((_H_M,), device="cuda", dtype=_h_torch.float32)
    matrix_a = _h_create_fp8(storage_a, _H_DTYPE, 1, source_a * _H_FP8_MAX)
    matrix_b = _h_create_fp8(
        storage_b, _H_DTYPE, 1, source_b * (_H_FP8_MAX / _H_WEIGHT_BOUND)
    )
    bias_tensor = _h_from_dlpack(bias)
    scratch_tensor = _h_from_dlpack(scratch).mark_layout_dynamic(leading_dim=1)
    output_tensor = _h_from_dlpack(output)
    compiled = _h_cute.compile(
        matmul_maxpool_sum_scale,
        matrix_a,
        matrix_b,
        bias_tensor,
        scratch_tensor,
        output_tensor,
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(matrix_a, matrix_b, bias_tensor, scratch_tensor, output_tensor)
    _h_torch.cuda.synchronize()
    timings = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _h_torch.cuda.Event(enable_timing=True)
        end = _h_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(matrix_a, matrix_b, bias_tensor, scratch_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings)[len(timings) // 2]
    a_fp8 = storage_a.view(_h_torch.float8_e4m3fn)
    b_fp8 = storage_b.view(_h_torch.float8_e4m3fn)
    scale_a = _h_torch.tensor(1.0 / _H_FP8_MAX, device="cuda")
    scale_b = _h_torch.tensor(_H_WEIGHT_BOUND / _H_FP8_MAX, device="cuda")
    linear = _h_torch._scaled_mm(
        a_fp8, b_fp8.t(), scale_a=scale_a, scale_b=scale_b, out_dtype=_h_torch.float32
    ) + bias
    reference = _h_functional.max_pool1d(linear.unsqueeze(1), 2).squeeze(1).sum(dim=1) * 0.5
    absolute = (output - reference).abs()
    full_max_abs = absolute.max().item()
    full_max_rel = (absolute / reference.abs().clamp_min(1.0)).max().item()
    if (
        not _h_torch.isfinite(output).all().item()
        or full_max_abs > 1.0
        or full_max_rel > 0.0002
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, full_rel={full_max_rel:.6f}"
        )
    print(
        "task=level2_55_matmul_maxpool_sum_scale "
        f"full_max_abs={full_max_abs:.6f} full_max_rel={full_max_rel:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _h_torch.cuda.synchronize()


main()
