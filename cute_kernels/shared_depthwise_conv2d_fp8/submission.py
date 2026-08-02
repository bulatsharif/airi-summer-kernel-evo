# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Full license text: ../NVIDIA_BSD_3_CLAUSE.txt

import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


BATCH = 8
CHANNELS = 64
HEIGHT = 256
WIDTH = 256
TILE_HEIGHT = 8
TILE_WIDTH = 32
SHARED_HEIGHT = TILE_HEIGHT + 2
SHARED_WIDTH = TILE_WIDTH + 2
SHARED_ELEMENTS = SHARED_HEIGHT * SHARED_WIDTH
THREADS = TILE_HEIGHT * TILE_WIDTH
FP8_MAX = 448.0
WEIGHT_BOUND = 0.25
INPUT_SCALE = 1.0 / FP8_MAX
WEIGHT_SCALE = WEIGHT_BOUND / FP8_MAX
FP8_DTYPE = cutlass.Float8E4M3FN
SEED = 20260731


@cute.kernel
def depthwise_conv2d_kernel(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output: cute.Tensor,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    tile_x, tile_y, plane_idx = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    input_tile = smem.allocate_tensor(FP8_DTYPE, SHARED_ELEMENTS)
    weight_tile = smem.allocate_tensor(FP8_DTYPE, 9)

    for load_round in cutlass.range_constexpr(2):
        shared_idx = load_round * THREADS + thread_idx
        if shared_idx < SHARED_ELEMENTS:
            shared_y = shared_idx // SHARED_WIDTH
            shared_x = shared_idx - shared_y * SHARED_WIDTH
            input_y = tile_y * TILE_HEIGHT + shared_y - 1
            input_x = tile_x * TILE_WIDTH + shared_x - 1
            if input_y >= 0 and input_y < HEIGHT:
                if input_x >= 0 and input_x < WIDTH:
                    input_tile[shared_idx] = input_tensor[
                        plane_idx, input_y * WIDTH + input_x
                    ]
                else:
                    input_tile[shared_idx] = FP8_DTYPE(0.0)
            else:
                input_tile[shared_idx] = FP8_DTYPE(0.0)

    channel_idx = plane_idx % CHANNELS
    if thread_idx < 9:
        weight_tile[thread_idx] = weight_tensor[channel_idx, thread_idx]
    cute.arch.sync_threads()

    local_y = thread_idx // TILE_WIDTH
    local_x = thread_idx - local_y * TILE_WIDTH
    accumulator = cutlass.Float32(0.0)
    for kernel_y in cutlass.range_constexpr(3):
        for kernel_x in cutlass.range_constexpr(3):
            input_value = input_tile[
                (local_y + kernel_y) * SHARED_WIDTH + local_x + kernel_x
            ].to(cutlass.Float32)
            weight_value = weight_tile[
                kernel_y * 3 + kernel_x
            ].to(cutlass.Float32)
            accumulator += input_value * weight_value

    output_y = tile_y * TILE_HEIGHT + local_y
    output_x = tile_x * TILE_WIDTH + local_x
    output[plane_idx, output_y * WIDTH + output_x] = accumulator * (
        INPUT_SCALE * WEIGHT_SCALE
    )


@cute.jit
def depthwise_conv2d(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output: cute.Tensor,
):
    depthwise_conv2d_kernel(
        input_tensor, weight_tensor, output
    ).launch(
        grid=(
            WIDTH // TILE_WIDTH,
            HEIGHT // TILE_HEIGHT,
            BATCH * CHANNELS,
        ),
        block=(THREADS, 1, 1),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    torch.manual_seed(SEED)
    source = torch.empty(
        (BATCH * CHANNELS, HEIGHT * WIDTH),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-1.0, 1.0)
    source_weight = torch.zeros(
        (CHANNELS, 16), device="cuda", dtype=torch.float32
    )
    source_weight[:, :9].uniform_(-WEIGHT_BOUND, WEIGHT_BOUND)
    input_storage = torch.empty_like(source, dtype=torch.uint8)
    weight_storage = torch.empty_like(source_weight, dtype=torch.uint8)
    output = torch.empty_like(source)

    input_tensor = create_cute_tensor_for_fp8(
        input_storage, FP8_DTYPE, 1, source * FP8_MAX
    )
    weight_tensor = create_cute_tensor_for_fp8(
        weight_storage,
        FP8_DTYPE,
        1,
        source_weight * (FP8_MAX / WEIGHT_BOUND),
    )
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = cute.compile(
        depthwise_conv2d, input_tensor, weight_tensor, output_tensor
    )

    compiled(input_tensor, weight_tensor, output_tensor)
    dequantized = (
        input_storage.view(torch.float8_e4m3fn).float() * INPUT_SCALE
    ).reshape(BATCH, CHANNELS, HEIGHT, WIDTH)
    dequantized_weight = (
        weight_storage.view(torch.float8_e4m3fn).float()[:, :9]
        * WEIGHT_SCALE
    ).reshape(CHANNELS, 1, 3, 3)
    reference = F.conv2d(
        dequantized, dequantized_weight, padding=1, groups=CHANNELS
    ).reshape(BATCH * CHANNELS, HEIGHT * WIDTH)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not torch.isfinite(output).all().item() or max_abs > 0.001:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    for _ in range(5):
        compiled(input_tensor, weight_tensor, output_tensor)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        compiled(input_tensor, weight_tensor, output_tensor)
    end.record()
    end.synchronize()
    kernel_time_ms = start.elapsed_time(end) / 20
    print(
        "kernel=shared_depthwise_conv2d_fp8 "
        f"shape=({BATCH},{CHANNELS},{HEIGHT},{WIDTH}) "
        f"max_abs={max_abs:.6f} mean_abs={mean_abs:.9f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )


main()
