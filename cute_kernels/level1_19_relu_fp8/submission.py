import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


M = 4096
N = 393216
THREADS_PER_CTA = 128
COPY_BITS = 128
SEED = 20260726
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def relu_fp8_kernel(
    tiled_input: cute.Tensor,
    tiled_output: cute.Tensor,
    thread_layout: cute.Layout,
    value_layout: cute.Layout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()

    # zipped_divide added a tile mode and a rest/grid mode. Select the tile
    # owned by this CTA while keeping both coordinates inside the tile.
    block_coord = ((None, None), block_idx)
    block_input = tiled_input[block_coord]
    block_output = tiled_output[block_coord]

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        tiled_input.element_type,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        tiled_output.element_type,
    )
    tiled_load = cute.make_tiled_copy_tv(
        load_atom,
        thread_layout,
        value_layout,
    )
    tiled_store = cute.make_tiled_copy_tv(
        store_atom,
        thread_layout,
        value_layout,
    )

    thread_load = tiled_load.get_slice(thread_idx)
    thread_store = tiled_store.get_slice(thread_idx)
    thread_input = thread_load.partition_S(block_input)
    thread_output = thread_store.partition_S(block_output)

    input_fragment = cute.make_fragment_like(thread_input)
    output_fragment = cute.make_fragment_like(thread_output)

    # GMEM FP8 -> registers FP8.
    cute.copy(load_atom, thread_input, input_fragment)

    # FP8 is the storage type. Arithmetic is deliberately performed in FP32.
    input_fp32 = input_fragment.load().to(cutlass.Float32)
    zero = cute.full_like(input_fp32, 0.0)
    output_fp32 = cute.where(input_fp32 > zero, input_fp32, zero)

    # Registers FP32 -> registers FP8 -> GMEM FP8.
    output_fragment.store(output_fp32.to(output_fragment.element_type))
    cute.copy(store_atom, output_fragment, thread_output)


@cute.jit
def relu_fp8(
    input_tensor: cute.Tensor,
    output_tensor: cute.Tensor,
    copy_bits: cutlass.Constexpr = COPY_BITS,
):
    vector_size = copy_bits // input_tensor.element_type.width

    # 128 threads are arranged as a logical 4x32 tile. Every thread owns a
    # logical 4x16 value tile for 128-bit vectorized FP8 loads and stores.
    thread_layout = cute.make_ordered_layout((4, 32), order=(1, 0))
    value_layout = cute.make_ordered_layout(
        (4, vector_size),
        order=(1, 0),
    )
    tile_shape, thread_value_layout = cute.make_layout_tv(
        thread_layout,
        value_layout,
    )

    tiled_input = cute.zipped_divide(input_tensor, tile_shape)
    tiled_output = cute.zipped_divide(output_tensor, tile_shape)

    print(f"[CuTe] CTA tile: {tile_shape}")
    print(f"[CuTe] thread/value layout: {thread_value_layout}")

    relu_fp8_kernel(
        tiled_input,
        tiled_output,
        thread_layout,
        value_layout,
    ).launch(
        grid=[cute.size(tiled_output, mode=[1]), 1, 1],
        block=[cute.size(thread_value_layout, mode=[0]), 1, 1],
    )


def make_output_tensor(storage):
    tensor = from_dlpack(storage)
    tensor.element_type = FP8_DTYPE
    return tensor.mark_layout_dynamic(leading_dim=1)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    torch.manual_seed(SEED)
    source_fp32 = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-1.0, 1.0)

    input_storage = torch.empty(
        (M, N),
        device="cuda",
        dtype=torch.uint8,
    )
    output_storage = torch.empty_like(input_storage)

    input_tensor = create_cute_tensor_for_fp8(
        input_storage,
        FP8_DTYPE,
        1,
        source_fp32,
    )
    output_tensor = make_output_tensor(output_storage)

    compiled_relu = cute.compile(
        relu_fp8,
        input_tensor,
        output_tensor,
    )
    compiled_relu(input_tensor, output_tensor)

    # ReLU on finite E4M3 data has a simple exact byte-level reference:
    # keep non-negative encodings and replace encodings with the sign bit by 0.
    sign_bit = torch.tensor(128, device="cuda", dtype=torch.uint8)
    zero_byte = torch.zeros((), device="cuda", dtype=torch.uint8)
    negative = torch.bitwise_and(input_storage, sign_bit) != 0
    reference_storage = torch.where(negative, zero_byte, input_storage)
    mismatches = (output_storage != reference_storage).sum().item()
    if mismatches != 0:
        raise RuntimeError(f"FP8 ReLU validation failed: {mismatches} mismatches")

    output_fp8 = output_storage.view(torch.float8_e4m3fn)
    checksum = output_fp8[::4, ::64].float().sum().item()
    print(f"result={checksum:.6f}")
    print(
        "task=level1_19_relu "
        f"shape={tuple(output_fp8.shape)} "
        f"storage={output_fp8.dtype} compute=torch.float32 "
        f"mismatches={mismatches}"
    )
    torch.cuda.synchronize()


main()
