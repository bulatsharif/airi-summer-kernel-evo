import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


M = 1024
N = 8192
K = 8192
SEED = 20260726
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
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
def bias_relu_kernel(output: cute.Tensor, bias: cute.Tensor):
    # TODO: output = max(output + bias[column], 0).
    pass


@cute.jit
def gemm_add_relu(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: construct/launch GEMM and BiasAdd+ReLU kernels.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    torch.manual_seed(_CUTE_HARNESS_SEED)
    source_a = torch.rand((M, K), device="cuda", dtype=torch.float32)
    source_b_nk = torch.empty(
        (N, K),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-WEIGHT_BOUND, WEIGHT_BOUND)
    bias = torch.randn((N,), device="cuda", dtype=torch.float32)
    storage_a = torch.empty((M, K), device="cuda", dtype=torch.uint8)
    storage_b = torch.empty((N, K), device="cuda", dtype=torch.uint8)
    output = torch.empty((M, N), device="cuda", dtype=torch.float32)

    matrix_a = create_cute_tensor_for_fp8(
        storage_a,
        FP8_DTYPE,
        1,
        source_a * FP8_MAX,
    )
    matrix_b_nk = create_cute_tensor_for_fp8(
        storage_b,
        FP8_DTYPE,
        1,
        source_b_nk * (FP8_MAX / WEIGHT_BOUND),
    )
    bias_tensor = from_dlpack(bias)
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)

    compiled = cute.compile(
        gemm_add_relu,
        matrix_a,
        matrix_b_nk,
        bias_tensor,
        output_tensor,
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    a_fp8 = storage_a.view(torch.float8_e4m3fn)
    b_fp8 = storage_b.view(torch.float8_e4m3fn)
    scale_a = torch.tensor(SCALE_A, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(SCALE_B, device="cuda", dtype=torch.float32)
    reference = torch.relu(
        torch._scaled_mm(
            a_fp8,
            b_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float32,
        )
        + bias
    )
    full_max_abs = (output - reference).abs().max().item()

    rows = torch.tensor(
        [0, 1, 7, 31, 127, 255, 511, 1023],
        device="cuda",
    )
    columns = torch.tensor(
        [0, 3, 31, 255, 1023, 2047, 4095, 8191],
        device="cuda",
    )
    fp32_reference = torch.relu(
        source_a.index_select(0, rows)
        @ source_b_nk.index_select(0, columns).t()
        + bias.index_select(0, columns)
    )
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()

    if (
        not torch.isfinite(output).all().item()
        or full_max_abs > 0.01
        or sample_max_abs > 0.1
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, "
            f"sample_abs={sample_max_abs:.6f}"
        )

    print(
        "task=level2_76_gemm_add_relu "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    torch.cuda.synchronize()


main()
