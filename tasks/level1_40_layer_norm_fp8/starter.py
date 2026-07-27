import cutlass
import cutlass.cute as cute


BATCH_SIZE = 16
FEATURES = 64
DIM_1 = 256
DIM_2 = 256
ROW_SIZE = FEATURES * DIM_1 * DIM_2
INPUT_SHAPE = (BATCH_SIZE, FEATURES, DIM_1, DIM_2)
NORMALIZED_SHAPE = (FEATURES, DIM_1, DIM_2)
EPSILON = 1.0e-5
SEED = 20260726
FP8_MAX = 448.0
INPUT_SCALE = 1.0 / FP8_MAX
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.jit
def warp_sum(value):
    # TODO: butterfly shuffle reduction.
    return value


@cute.kernel
def layer_norm_kernel(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
):
    # TODO: streaming mean, centered variance, normalization and store.
    pass


@cute.jit
def layer_norm(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
):
    # TODO: launch layer_norm_kernel.
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


_HARNESS_BATCH_SIZE = 16
_HARNESS_FEATURES = 64
_HARNESS_DIM_1 = 256
_HARNESS_DIM_2 = 256
_HARNESS_ROW_SIZE = (
    _HARNESS_FEATURES * _HARNESS_DIM_1 * _HARNESS_DIM_2
)
_HARNESS_INPUT_SHAPE = (
    _HARNESS_BATCH_SIZE,
    _HARNESS_FEATURES,
    _HARNESS_DIM_1,
    _HARNESS_DIM_2,
)
_HARNESS_NORMALIZED_SHAPE = (
    _HARNESS_FEATURES,
    _HARNESS_DIM_1,
    _HARNESS_DIM_2,
)
_HARNESS_EPSILON = 1.0e-5
_HARNESS_SEED = 20260726
_HARNESS_FP8_MAX = 448.0
_HARNESS_INPUT_SCALE = 1.0 / _HARNESS_FP8_MAX
_HARNESS_FP8_DTYPE = _harness_cutlass.Float8E4M3FN


def main():
    if not _harness_torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    _harness_torch.manual_seed(_HARNESS_SEED)
    source = _harness_torch.rand(
        (_HARNESS_BATCH_SIZE, _HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    storage = _harness_torch.empty(
        (_HARNESS_BATCH_SIZE, _HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.uint8,
    )
    output = _harness_torch.empty(
        (_HARNESS_BATCH_SIZE, _HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    input_tensor = _harness_create_cute_tensor_for_fp8(
        storage,
        _HARNESS_FP8_DTYPE,
        1,
        source * _HARNESS_FP8_MAX,
    )
    output_tensor = _harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _harness_cute.compile(layer_norm, input_tensor, output_tensor)
    compiled(input_tensor, output_tensor)

    dequantized = (
        storage.view(_harness_torch.float8_e4m3fn).float()
        * _HARNESS_INPUT_SCALE
    )
    reference = _harness_F.layer_norm(
        dequantized.reshape(_HARNESS_INPUT_SHAPE),
        _HARNESS_NORMALIZED_SHAPE,
        weight=None,
        bias=None,
        eps=_HARNESS_EPSILON,
    ).reshape(_HARNESS_BATCH_SIZE, _HARNESS_ROW_SIZE)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not _harness_torch.isfinite(output).all().item() or max_abs > 0.01:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    print(
        "task=level1_40_layer_norm "
        f"full_max_abs={max_abs:.6f} "
        f"full_mean_abs={mean_abs:.9f} PASS"
    )
    _harness_torch.cuda.synchronize()


main()
