# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Full license text: ../NVIDIA_BSD_3_CLAUSE.txt

import torch

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


ROWS = 4096
COLUMNS = 4096
THREADS = 256
WARPS = THREADS // 32
VALUES_PER_THREAD = COLUMNS // THREADS
EPSILON = 1.0e-5
FP8_MAX = 448.0
INPUT_SCALE = 1.0 / FP8_MAX
FP8_DTYPE = cutlass.Float8E4M3FN
SEED = 20260731


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.jit
def cta_sum(value, scratch: cute.Tensor, thread_idx):
    warp_idx = thread_idx >> 5
    lane_idx = thread_idx & 31
    value = warp_sum(value)
    if lane_idx == 0:
        scratch[warp_idx] = value
    cute.arch.sync_threads()

    if warp_idx == 0:
        if lane_idx < WARPS:
            value = scratch[lane_idx]
        else:
            value = cutlass.Float32(0.0)
        value = warp_sum(value)
        if lane_idx == 0:
            scratch[WARPS] = value
    cute.arch.sync_threads()
    return scratch[WARPS]


@cute.kernel
def residual_rmsnorm_kernel(
    input_tensor: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    scratch = smem.allocate_tensor(cutlass.Float32, WARPS + 1)

    sum_of_squares = cutlass.Float32(0.0)
    for iteration in cutlass.range(VALUES_PER_THREAD):
        column_idx = iteration * THREADS + thread_idx
        value = (
            input_tensor[row_idx, column_idx].to(cutlass.Float32)
            * INPUT_SCALE
            + residual[row_idx, column_idx]
        )
        sum_of_squares += value * value

    inverse_rms = cute.rsqrt(
        cta_sum(sum_of_squares, scratch, thread_idx) / COLUMNS + EPSILON
    )
    for iteration in cutlass.range(VALUES_PER_THREAD):
        column_idx = iteration * THREADS + thread_idx
        value = (
            input_tensor[row_idx, column_idx].to(cutlass.Float32)
            * INPUT_SCALE
            + residual[row_idx, column_idx]
        )
        output[row_idx, column_idx] = (
            value * inverse_rms * weight[column_idx]
        )


@cute.jit
def residual_rmsnorm(
    input_tensor: cute.Tensor,
    residual: cute.Tensor,
    weight: cute.Tensor,
    output: cute.Tensor,
):
    residual_rmsnorm_kernel(
        input_tensor,
        residual,
        weight,
        output,
    ).launch(grid=(ROWS, 1, 1), block=(THREADS, 1, 1))


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    torch.manual_seed(SEED)
    source = torch.empty(
        (ROWS, COLUMNS), device="cuda", dtype=torch.float32
    ).uniform_(-1.0, 1.0)
    residual = torch.randn(
        (ROWS, COLUMNS), device="cuda", dtype=torch.float32
    ) * 0.1
    weight = torch.randn((COLUMNS,), device="cuda", dtype=torch.float32)
    input_storage = torch.empty_like(source, dtype=torch.uint8)
    output = torch.empty_like(source)

    input_tensor = create_cute_tensor_for_fp8(
        input_storage, FP8_DTYPE, 1, source * FP8_MAX
    )
    residual_tensor = from_dlpack(residual).mark_layout_dynamic(leading_dim=1)
    weight_tensor = from_dlpack(weight)
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = cute.compile(
        residual_rmsnorm,
        input_tensor,
        residual_tensor,
        weight_tensor,
        output_tensor,
    )

    compiled(input_tensor, residual_tensor, weight_tensor, output_tensor)
    dequantized = (
        input_storage.view(torch.float8_e4m3fn).float() * INPUT_SCALE
    )
    combined = dequantized + residual
    reference = combined * torch.rsqrt(
        combined.square().mean(dim=1, keepdim=True) + EPSILON
    ) * weight
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not torch.isfinite(output).all().item() or max_abs > 0.002:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    for _ in range(5):
        compiled(input_tensor, residual_tensor, weight_tensor, output_tensor)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        compiled(input_tensor, residual_tensor, weight_tensor, output_tensor)
    end.record()
    end.synchronize()
    kernel_time_ms = start.elapsed_time(end) / 20
    print(
        "kernel=fused_residual_rmsnorm_fp8 "
        f"shape=({ROWS},{COLUMNS}) max_abs={max_abs:.6f} "
        f"mean_abs={mean_abs:.9f} kernel_time_ms={kernel_time_ms:.6f} PASS"
    )


main()
