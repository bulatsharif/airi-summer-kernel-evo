import cutlass
import cutlass.cute as cute


@cute.kernel
def fp8_gemm_kernel(matrix_a: cute.Tensor, matrix_b_nk: cute.Tensor, output: cute.Tensor):
    # TODO: FP8 GEMM with FP32 accumulation and correctly restored scales.
    pass


@cute.kernel
def mish_mish_kernel(output: cute.Tensor, bias: cute.Tensor):
    # TODO: add bias and apply Mish twice.
    pass


@cute.jit
def matmul_mish_mish(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: launch the GEMM and post-op kernels.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _h_torch
import torch.nn.functional as _h_functional
import cutlass as _h_cutlass
import cutlass.cute as _h_cute
from cutlass.cute.runtime import from_dlpack as _h_from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8 as _h_create_fp8


_H_M, _H_N, _H_K = 1024, 8192, 8192
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
    output = _h_torch.empty((_H_M, _H_N), device="cuda", dtype=_h_torch.float32)
    matrix_a = _h_create_fp8(storage_a, _H_DTYPE, 1, source_a * _H_FP8_MAX)
    matrix_b = _h_create_fp8(
        storage_b, _H_DTYPE, 1, source_b * (_H_FP8_MAX / _H_WEIGHT_BOUND)
    )
    bias_tensor = _h_from_dlpack(bias)
    output_tensor = _h_from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = _h_cute.compile(matmul_mish_mish, matrix_a, matrix_b, bias_tensor, output_tensor)
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(matrix_a, matrix_b, bias_tensor, output_tensor)
    _h_torch.cuda.synchronize()
    timings = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _h_torch.cuda.Event(enable_timing=True)
        end = _h_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(matrix_a, matrix_b, bias_tensor, output_tensor)
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
    reference = _h_functional.mish(_h_functional.mish(linear))
    full_max_abs = (output - reference).abs().max().item()
    rows = _h_torch.tensor([0, 7, 127, 511, 1023], device="cuda")
    columns = _h_torch.tensor([0, 31, 255, 2047, 8191], device="cuda")
    fp32_linear = source_a.index_select(0, rows) @ source_b.index_select(0, columns).t()
    fp32_linear += bias.index_select(0, columns)
    fp32_reference = _h_functional.mish(_h_functional.mish(fp32_linear))
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()
    if not _h_torch.isfinite(output).all().item() or full_max_abs > 0.02 or sample_max_abs > 0.2:
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, sample_abs={sample_max_abs:.6f}"
        )
    print(
        "task=level2_29_matmul_mish_mish "
        f"full_max_abs={full_max_abs:.6f} sample_max_abs={sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _h_torch.cuda.synchronize()


main()
