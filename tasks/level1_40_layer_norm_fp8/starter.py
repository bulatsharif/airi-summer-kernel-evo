import torch
import torch.nn.functional as F

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


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
import torch as _cute_harness_torch
import torch.nn.functional as _cute_harness_functional

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import from_dlpack as _cute_harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _cute_harness_create_fp8,
)


_CUTE_HARNESS_BATCH_SIZE = 16
_CUTE_HARNESS_FEATURES = 64
_CUTE_HARNESS_DIM_1 = 256
_CUTE_HARNESS_DIM_2 = 256
_CUTE_HARNESS_ROW_SIZE = (
    _CUTE_HARNESS_FEATURES * _CUTE_HARNESS_DIM_1 * _CUTE_HARNESS_DIM_2
)
_CUTE_HARNESS_INPUT_SHAPE = (
    _CUTE_HARNESS_BATCH_SIZE,
    _CUTE_HARNESS_FEATURES,
    _CUTE_HARNESS_DIM_1,
    _CUTE_HARNESS_DIM_2,
)
_CUTE_HARNESS_NORMALIZED_SHAPE = (
    _CUTE_HARNESS_FEATURES,
    _CUTE_HARNESS_DIM_1,
    _CUTE_HARNESS_DIM_2,
)
_CUTE_HARNESS_EPSILON = 1.0e-5
_CUTE_HARNESS_FP8_MAX = 448.0
_CUTE_HARNESS_INPUT_SCALE = 1.0 / _CUTE_HARNESS_FP8_MAX
_CUTE_HARNESS_FP8_DTYPE = _cute_harness_cutlass.Float8E4M3FN


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    source = _cute_harness_torch.rand(
        (_CUTE_HARNESS_BATCH_SIZE, _CUTE_HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    storage = _cute_harness_torch.empty(
        (_CUTE_HARNESS_BATCH_SIZE, _CUTE_HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_cute_harness_torch.uint8,
    )
    output = _cute_harness_torch.empty(
        (_CUTE_HARNESS_BATCH_SIZE, _CUTE_HARNESS_ROW_SIZE),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    input_tensor = _cute_harness_create_fp8(
        storage,
        _CUTE_HARNESS_FP8_DTYPE,
        1,
        source * _CUTE_HARNESS_FP8_MAX,
    )
    output_tensor = _cute_harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1,
    )

    compiled = _cute_harness_cute.compile(
        layer_norm,
        input_tensor,
        output_tensor,
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(input_tensor, output_tensor)
    _cute_harness_torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(input_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    dequantized = (
        storage.view(_cute_harness_torch.float8_e4m3fn).float()
        * _CUTE_HARNESS_INPUT_SCALE
    )
    reference = _cute_harness_functional.layer_norm(
        dequantized.reshape(_CUTE_HARNESS_INPUT_SHAPE),
        _CUTE_HARNESS_NORMALIZED_SHAPE,
        weight=None,
        bias=None,
        eps=_CUTE_HARNESS_EPSILON,
    ).reshape(_CUTE_HARNESS_BATCH_SIZE, _CUTE_HARNESS_ROW_SIZE)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if (
        not _cute_harness_torch.isfinite(output).all().item()
        or max_abs > 0.01
    ):
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    print(
        "task=level1_40_layer_norm "
        f"full_max_abs={max_abs:.6f} "
        f"full_mean_abs={mean_abs:.9f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
