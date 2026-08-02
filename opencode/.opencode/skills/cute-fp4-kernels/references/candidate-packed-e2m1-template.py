"""Neutral candidate-only packed E2M1 decode pattern.

Assumptions: two values per byte, low nibble first, no odd tail. Replace the
shape, scale, output dtype, and public entrypoint with the task contract.
This draft contains no harness-owned main, inputs, reference, timing, or PASS.
"""

import cutlass
import cutlass.cute as cute


ELEMENTS = 4096
PACKED_BYTES = ELEMENTS // 2
THREADS_PER_CTA = 128
DEQUANT_SCALE = 1.0
OUTPUT_DTYPE = cutlass.Float32


@cute.jit
def decode_e2m1(nibble):
    magnitude = nibble & 0x7
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
    if nibble & 0x8:
        value = -value
    return value


@cute.kernel
def unpack_e2m1_kernel(
    packed: cute.Tensor,
    output: cute.Tensor,
):
    thread_x, _, _ = cute.arch.thread_idx()
    block_x, _, _ = cute.arch.block_idx()
    block_size_x, _, _ = cute.arch.block_dim()
    byte_index = block_x * block_size_x + thread_x

    if byte_index < PACKED_BYTES:
        byte = packed[byte_index].to(cutlass.Int32)
        output_index = byte_index * 2
        output[output_index] = (
            decode_e2m1(byte & 0xF) * DEQUANT_SCALE
        ).to(OUTPUT_DTYPE)
        output[output_index + 1] = (
            decode_e2m1(byte >> 4) * DEQUANT_SCALE
        ).to(OUTPUT_DTYPE)


@cute.jit
def unpack_e2m1(
    packed: cute.Tensor,
    output: cute.Tensor,
):
    unpack_e2m1_kernel(packed, output).launch(
        grid=((PACKED_BYTES + THREADS_PER_CTA - 1) // THREADS_PER_CTA, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )
