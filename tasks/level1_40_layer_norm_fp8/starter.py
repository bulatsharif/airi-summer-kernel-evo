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
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP8 support is required")

    torch.manual_seed(_CUTE_HARNESS_SEED)
    source = torch.rand(
        (BATCH_SIZE, ROW_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    storage = torch.empty(
        (BATCH_SIZE, ROW_SIZE),
        device="cuda",
        dtype=torch.uint8,
    )
    output = torch.empty(
        (BATCH_SIZE, ROW_SIZE),
        device="cuda",
        dtype=torch.float32,
    )
    input_tensor = create_cute_tensor_for_fp8(
        storage,
        FP8_DTYPE,
        1,
        source * FP8_MAX,
    )
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)

    compiled = cute.compile(layer_norm, input_tensor, output_tensor)
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(input_tensor, output_tensor)
    torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(input_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    dequantized = storage.view(torch.float8_e4m3fn).float() * INPUT_SCALE
    reference = F.layer_norm(
        dequantized.reshape(INPUT_SHAPE),
        NORMALIZED_SHAPE,
        weight=None,
        bias=None,
        eps=EPSILON,
    ).reshape(BATCH_SIZE, ROW_SIZE)
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not torch.isfinite(output).all().item() or max_abs > 0.01:
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
    torch.cuda.synchronize()


main()
