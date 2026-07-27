import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


BATCH_SIZE = 2
IN_CHANNELS = 8
OUT_CHANNELS = 12
GROUPS = 2
IN_CHANNELS_PER_GROUP = IN_CHANNELS // GROUPS
OUT_CHANNELS_PER_GROUP = OUT_CHANNELS // GROUPS
IN_D = 7
IN_H = 9
IN_W = 11
KD = 3
KH = 5
KW = 7
STRIDE_D = 2
STRIDE_H = 2
STRIDE_W = 3
PAD_D = 1
PAD_H = 2
PAD_W = 3
OUTPUT_PAD_D = 1
OUTPUT_PAD_H = 1
OUTPUT_PAD_W = 2
OUT_D = 14
OUT_H = 18
OUT_W = 33
INPUT_ROW_SIZE = IN_CHANNELS * IN_D * IN_H * IN_W
WEIGHT_LOGICAL_ROW_SIZE = OUT_CHANNELS_PER_GROUP * KD * KH * KW
WEIGHT_STORAGE_ROW_SIZE = 640
OUTPUT_ROW_SIZE = OUT_CHANNELS * OUT_D * OUT_H * OUT_W
TOTAL_OUTPUT_ELEMENTS = BATCH_SIZE * OUTPUT_ROW_SIZE
THREADS_PER_CTA = 128
SEED = 20260727
FP8_MAX = 448.0
WEIGHT_BOUND = 0.25
INPUT_SCALE = 1.0 / FP8_MAX
WEIGHT_SCALE = WEIGHT_BOUND / FP8_MAX
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def conv_transpose3d_kernel(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
):
    thread_x, _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_idx()
    linear = block_x * THREADS_PER_CTA + thread_x
    if linear < TOTAL_OUTPUT_ELEMENTS:
        batch = linear // OUTPUT_ROW_SIZE
        output_offset = linear % OUTPUT_ROW_SIZE
        output_w = output_offset % OUT_W
        remaining = output_offset // OUT_W
        output_h = remaining % OUT_H
        remaining = remaining // OUT_H
        output_d = remaining % OUT_D
        output_channel = remaining // OUT_D

        group = output_channel // OUT_CHANNELS_PER_GROUP
        output_channel_in_group = (
            output_channel - group * OUT_CHANNELS_PER_GROUP
        )
        input_channel_start = group * IN_CHANNELS_PER_GROUP
        accumulator = 0.0

        for input_channel_in_group in cutlass.range(
            IN_CHANNELS_PER_GROUP
        ):
            input_channel = input_channel_start + input_channel_in_group
            for kernel_d in cutlass.range(KD):
                input_d_numerator = output_d + PAD_D - kernel_d
                if input_d_numerator >= 0:
                    if input_d_numerator % STRIDE_D == 0:
                        input_d = input_d_numerator // STRIDE_D
                        if input_d < IN_D:
                            for kernel_h in cutlass.range(KH):
                                input_h_numerator = (
                                    output_h + PAD_H - kernel_h
                                )
                                if input_h_numerator >= 0:
                                    if input_h_numerator % STRIDE_H == 0:
                                        input_h = (
                                            input_h_numerator // STRIDE_H
                                        )
                                        if input_h < IN_H:
                                            for kernel_w in cutlass.range(KW):
                                                input_w_numerator = (
                                                    output_w
                                                    + PAD_W
                                                    - kernel_w
                                                )
                                                if input_w_numerator >= 0:
                                                    if (
                                                        input_w_numerator
                                                        % STRIDE_W
                                                        == 0
                                                    ):
                                                        input_w = (
                                                            input_w_numerator
                                                            // STRIDE_W
                                                        )
                                                        if input_w < IN_W:
                                                            input_offset = (
                                                                (
                                                                    (
                                                                        input_channel
                                                                        * IN_D
                                                                        + input_d
                                                                    )
                                                                    * IN_H
                                                                    + input_h
                                                                )
                                                                * IN_W
                                                                + input_w
                                                            )
                                                            weight_offset = (
                                                                (
                                                                    (
                                                                        output_channel_in_group
                                                                        * KD
                                                                        + kernel_d
                                                                    )
                                                                    * KH
                                                                    + kernel_h
                                                                )
                                                                * KW
                                                                + kernel_w
                                                            )
                                                            input_value = input_tensor[
                                                                batch,
                                                                input_offset,
                                                            ].to(cutlass.Float32)
                                                            weight_value = weight_tensor[
                                                                input_channel,
                                                                weight_offset,
                                                            ].to(cutlass.Float32)
                                                            accumulator = (
                                                                accumulator
                                                                + input_value
                                                                * weight_value
                                                            )

        output_tensor[batch, output_offset] = accumulator * (
            INPUT_SCALE * WEIGHT_SCALE
        )


@cute.jit
def conv_transpose3d(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
):
    conv_transpose3d_kernel(
        input_tensor,
        weight_tensor,
        output_tensor,
    ).launch(
        grid=(
            (TOTAL_OUTPUT_ELEMENTS + THREADS_PER_CTA - 1)
            // THREADS_PER_CTA,
            1,
            1,
        ),
        block=(THREADS_PER_CTA, 1, 1),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")
    torch.manual_seed(SEED)
    source_input = torch.empty(
        (BATCH_SIZE, INPUT_ROW_SIZE),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-1.0, 1.0)
    source_weight = torch.zeros(
        (IN_CHANNELS, WEIGHT_STORAGE_ROW_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    source_weight[:, :WEIGHT_LOGICAL_ROW_SIZE].uniform_(
        -WEIGHT_BOUND,
        WEIGHT_BOUND,
    )
    input_storage = torch.empty_like(source_input, dtype=torch.uint8)
    weight_storage = torch.empty_like(source_weight, dtype=torch.uint8)
    output = torch.empty(
        (BATCH_SIZE, OUTPUT_ROW_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    input_tensor = create_cute_tensor_for_fp8(
        input_storage,
        FP8_DTYPE,
        1,
        source_input * FP8_MAX,
    )
    weight_tensor = create_cute_tensor_for_fp8(
        weight_storage,
        FP8_DTYPE,
        1,
        source_weight * (FP8_MAX / WEIGHT_BOUND),
    )
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = cute.compile(
        conv_transpose3d,
        input_tensor,
        weight_tensor,
        output_tensor,
    )
    compiled(input_tensor, weight_tensor, output_tensor)
    dequantized_input = (
        input_storage.view(torch.float8_e4m3fn).float() * INPUT_SCALE
    ).reshape(BATCH_SIZE, IN_CHANNELS, IN_D, IN_H, IN_W)
    dequantized_weight = (
        weight_storage.view(torch.float8_e4m3fn)
        .float()[:, :WEIGHT_LOGICAL_ROW_SIZE]
        * WEIGHT_SCALE
    ).reshape(
        IN_CHANNELS,
        OUT_CHANNELS_PER_GROUP,
        KD,
        KH,
        KW,
    )
    reference = F.conv_transpose3d(
        dequantized_input,
        dequantized_weight,
        stride=(STRIDE_D, STRIDE_H, STRIDE_W),
        padding=(PAD_D, PAD_H, PAD_W),
        output_padding=(OUTPUT_PAD_D, OUTPUT_PAD_H, OUTPUT_PAD_W),
        groups=GROUPS,
    ).reshape(BATCH_SIZE, OUTPUT_ROW_SIZE)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not torch.isfinite(output).all().item() or max_abs > 0.01:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )
    print(
        "task=level1_72_conv_transpose3d "
        f"full_max_abs={max_abs:.6f} "
        f"full_mean_abs={mean_abs:.9f} PASS"
    )
    torch.cuda.synchronize()


main()
