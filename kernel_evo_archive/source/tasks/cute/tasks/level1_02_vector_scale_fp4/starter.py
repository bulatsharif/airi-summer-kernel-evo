# ruff: noqa: E402, F401, F821

import cutlass
import cutlass.cute as cute


N = 67_108_864
THREADS = 256


@cute.kernel
def scale_fp4_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    # TODO: load one packed byte, decode both E2M1 nibbles, and store FP16.
    pass


@cute.jit
def scale_fp4(input_ptr: cute.Pointer, output_ptr: cute.Pointer):
    # TODO: create logical tensors and launch scale_fp4_kernel.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _cute_harness_torch

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import make_ptr as _cute_harness_make_ptr


_CUTE_HARNESS_N = 67_108_864
_CUTE_HARNESS_UINT8 = _cute_harness_cutlass.Uint8
_CUTE_HARNESS_FP16 = _cute_harness_cutlass.Float16


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP4 support is required")

    packed = (
        _cute_harness_torch.arange(
            _CUTE_HARNESS_N // 2,
            device="cuda",
            dtype=_cute_harness_torch.int64,
        )
        % 256
    ).to(_cute_harness_torch.uint8)
    output = _cute_harness_torch.empty(
        _CUTE_HARNESS_N,
        device="cuda",
        dtype=_cute_harness_torch.float16,
    )
    input_ptr = _cute_harness_make_ptr(
        _CUTE_HARNESS_UINT8,
        packed.data_ptr(),
        _cute_harness_cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = _cute_harness_make_ptr(
        _CUTE_HARNESS_FP16,
        output.data_ptr(),
        _cute_harness_cute.AddressSpace.gmem,
        assumed_align=16,
    )
    compiled = _cute_harness_cute.compile(
        scale_fp4,
        input_ptr,
        output_ptr,
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(input_ptr, output_ptr)
    _cute_harness_torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(input_ptr, output_ptr)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    values = _cute_harness_torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=_cute_harness_torch.float16,
    )
    reference = _cute_harness_torch.empty_like(output)
    reference[0::2] = values[(packed & 15).long()] * 0.5
    reference[1::2] = values[(packed >> 4).long()] * 0.5
    max_abs = (output - reference).abs().max().item()
    if max_abs != 0.0:
        _cute_harness_out_abs = output.abs().max().item()
        raise RuntimeError(f"validation failed: max_abs={max_abs:.6f}, out_abs={_cute_harness_out_abs:.6f}")

    print(
        "task=level1_02_vector_scale_fp4 "
        f"elements={_CUTE_HARNESS_N} "
        f"kernel_time_ms={kernel_time_ms:.6f} "
        f"max_abs={max_abs:.6f} PASS"
    )


main()
