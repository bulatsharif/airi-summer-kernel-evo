import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_ptr


N = 67_108_864
THREADS = 256


@cute.jit
def decode_e2m1(code):
    magnitude = code & 7
    value = cutlass.Float32(0.0)
    if magnitude == 1:
        value = 0.5
    if magnitude == 2:
        value = 1.0
    if magnitude == 3:
        value = 1.5
    if magnitude == 4:
        value = 2.0
    if magnitude == 5:
        value = 3.0
    if magnitude == 6:
        value = 4.0
    if magnitude == 7:
        value = 6.0
    if code & 8:
        value = -value
    return value


@cute.kernel
def scale_fp4_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    byte_index = block_idx * THREADS + thread_idx
    packed = input_tensor[byte_index].to(cutlass.Int32)
    output_index = byte_index * 2
    output_tensor[output_index] = (decode_e2m1(packed & 15) * 0.5).to(cutlass.Float16)
    output_tensor[output_index + 1] = (decode_e2m1(packed >> 4) * 0.5).to(cutlass.Float16)


@cute.jit
def scale_fp4(input_ptr: cute.Pointer, output_ptr: cute.Pointer):
    input_tensor = cute.make_tensor(
        input_ptr,
        cute.make_layout((N // 2,), stride=(1,)),
    )
    output_tensor = cute.make_tensor(
        output_ptr,
        cute.make_layout((N,), stride=(1,)),
    )
    scale_fp4_kernel(input_tensor, output_tensor).launch(
        grid=(cute.ceil_div(N // 2, THREADS), 1, 1),
        block=(THREADS, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device with FP4 support is required")

    packed = (torch.arange(N // 2, device="cuda", dtype=torch.int64) % 256).to(torch.uint8)
    output = torch.empty(N, device="cuda", dtype=torch.float16)
    input_ptr = make_ptr(
        cutlass.Uint8,
        packed.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    output_ptr = make_ptr(
        cutlass.Float16,
        output.data_ptr(),
        cute.AddressSpace.gmem,
        assumed_align=16,
    )
    compiled = cute.compile(scale_fp4, input_ptr, output_ptr)
    compiled(input_ptr, output_ptr)

    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
        device="cuda",
        dtype=torch.float16,
    )
    reference = torch.empty_like(output)
    reference[0::2] = values[(packed & 15).long()] * 0.5
    reference[1::2] = values[(packed >> 4).long()] * 0.5
    max_abs = (output - reference).abs().max().item()
    if max_abs != 0.0:
        raise RuntimeError(f"validation failed: max_abs={max_abs:.6f}")

    print(f"task=level1_02_vector_scale_fp4 elements={N} max_abs={max_abs:.6f} PASS")


main()
