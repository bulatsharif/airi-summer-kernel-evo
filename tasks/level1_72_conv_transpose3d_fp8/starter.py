import cutlass
import cutlass.cute as cute


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
    # TODO: output-centric grouped ConvTranspose3d with FP32 accumulation.
    pass


@cute.jit
def conv_transpose3d(
    input_tensor: cute.Tensor,
    weight_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
):
    # TODO: launch conv_transpose3d_kernel.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _harness_torch
import torch.nn.functional as _harness_F

import cutlass as _harness_cutlass
import cutlass.cute as _harness_cute
from cutlass.cute.runtime import from_dlpack as _harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _harness_create_cute_tensor_for_fp8,
)


_HARNESS_BATCH_SIZE = 2
_HARNESS_IN_CHANNELS = 8
_HARNESS_OUT_CHANNELS = 12
_HARNESS_GROUPS = 2
_HARNESS_OUT_CHANNELS_PER_GROUP = (
    _HARNESS_OUT_CHANNELS // _HARNESS_GROUPS
)
_HARNESS_IN_D = 7
_HARNESS_IN_H = 9
_HARNESS_IN_W = 11
_HARNESS_KD = 3
_HARNESS_KH = 5
_HARNESS_KW = 7
_HARNESS_STRIDE = (2, 2, 3)
_HARNESS_PADDING = (1, 2, 3)
_HARNESS_OUTPUT_PADDING = (1, 1, 2)
_HARNESS_OUT_D = 14
_HARNESS_OUT_H = 18
_HARNESS_OUT_W = 33
_HARNESS_INPUT_SHAPE = (
    _HARNESS_BATCH_SIZE,
    _HARNESS_IN_CHANNELS,
    _HARNESS_IN_D,
    _HARNESS_IN_H,
    _HARNESS_IN_W,
)
_HARNESS_WEIGHT_SHAPE = (
    _HARNESS_IN_CHANNELS,
    _HARNESS_OUT_CHANNELS_PER_GROUP,
    _HARNESS_KD,
    _HARNESS_KH,
    _HARNESS_KW,
)
_HARNESS_OUTPUT_SHAPE = (
    _HARNESS_BATCH_SIZE,
    _HARNESS_OUT_CHANNELS,
    _HARNESS_OUT_D,
    _HARNESS_OUT_H,
    _HARNESS_OUT_W,
)
_HARNESS_INPUT_ROW_SIZE = (
    _HARNESS_IN_CHANNELS
    * _HARNESS_IN_D
    * _HARNESS_IN_H
    * _HARNESS_IN_W
)
_HARNESS_WEIGHT_LOGICAL_ROW_SIZE = (
    _HARNESS_OUT_CHANNELS_PER_GROUP
    * _HARNESS_KD
    * _HARNESS_KH
    * _HARNESS_KW
)
_HARNESS_WEIGHT_STORAGE_ROW_SIZE = 640
_HARNESS_OUTPUT_ROW_SIZE = (
    _HARNESS_OUT_CHANNELS
    * _HARNESS_OUT_D
    * _HARNESS_OUT_H
    * _HARNESS_OUT_W
)
_HARNESS_SEED = 20260727
_HARNESS_FP8_MAX = 448.0
_HARNESS_WEIGHT_BOUND = 0.25
_HARNESS_INPUT_SCALE = 1.0 / _HARNESS_FP8_MAX
_HARNESS_WEIGHT_SCALE = _HARNESS_WEIGHT_BOUND / _HARNESS_FP8_MAX
_HARNESS_FP8_DTYPE = _harness_cutlass.Float8E4M3FN


def main():
    if not _harness_torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    _harness_torch.manual_seed(_HARNESS_SEED)
    source_input = _harness_torch.empty(
        (_HARNESS_BATCH_SIZE, _HARNESS_INPUT_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.float32,
    ).uniform_(-1.0, 1.0)
    source_weight = _harness_torch.zeros(
        (_HARNESS_IN_CHANNELS, _HARNESS_WEIGHT_STORAGE_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    source_weight[:, :_HARNESS_WEIGHT_LOGICAL_ROW_SIZE].uniform_(
        -_HARNESS_WEIGHT_BOUND,
        _HARNESS_WEIGHT_BOUND,
    )
    input_storage = _harness_torch.empty_like(
        source_input,
        dtype=_harness_torch.uint8,
    )
    weight_storage = _harness_torch.empty_like(
        source_weight,
        dtype=_harness_torch.uint8,
    )
    output = _harness_torch.empty(
        (_HARNESS_BATCH_SIZE, _HARNESS_OUTPUT_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.float32,
    )

    input_tensor = _harness_create_cute_tensor_for_fp8(
        input_storage,
        _HARNESS_FP8_DTYPE,
        1,
        source_input * _HARNESS_FP8_MAX,
    )
    weight_tensor = _harness_create_cute_tensor_for_fp8(
        weight_storage,
        _HARNESS_FP8_DTYPE,
        1,
        source_weight * (_HARNESS_FP8_MAX / _HARNESS_WEIGHT_BOUND),
    )
    output_tensor = _harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _harness_cute.compile(
        conv_transpose3d,
        input_tensor,
        weight_tensor,
        output_tensor,
    )
    compiled(input_tensor, weight_tensor, output_tensor)

    dequantized_input = (
        input_storage.view(_harness_torch.float8_e4m3fn).float()
        * _HARNESS_INPUT_SCALE
    ).reshape(_HARNESS_INPUT_SHAPE)
    dequantized_weight = (
        weight_storage.view(_harness_torch.float8_e4m3fn)
        .float()[:, :_HARNESS_WEIGHT_LOGICAL_ROW_SIZE]
        * _HARNESS_WEIGHT_SCALE
    ).reshape(_HARNESS_WEIGHT_SHAPE)
    reference = _harness_F.conv_transpose3d(
        dequantized_input,
        dequantized_weight,
        bias=None,
        stride=_HARNESS_STRIDE,
        padding=_HARNESS_PADDING,
        output_padding=_HARNESS_OUTPUT_PADDING,
        groups=_HARNESS_GROUPS,
    ).reshape(_HARNESS_BATCH_SIZE, _HARNESS_OUTPUT_ROW_SIZE)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not _harness_torch.isfinite(output).all().item() or max_abs > 0.01:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    print(
        "task=level1_72_conv_transpose3d "
        f"full_max_abs={max_abs:.6f} "
        f"full_mean_abs={mean_abs:.9f} PASS"
    )
    _harness_torch.cuda.synchronize()


main()
