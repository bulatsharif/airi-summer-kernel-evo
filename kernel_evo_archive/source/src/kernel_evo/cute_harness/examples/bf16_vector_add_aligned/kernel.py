"""Aligned BF16 vector add fast path for nvidia-cutlass-dsl 4.2.1."""

import json
import os

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


@cute.kernel
def add_aligned_kernel(
    g_a: cute.Tensor,
    g_b: cute.Tensor,
    g_c: cute.Tensor,
    thread_layout: cute.Layout,
    value_layout: cute.Layout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    block_coord = (None, block_idx)
    block_a = g_a[block_coord]
    block_b = g_b[block_coord]
    block_c = g_c[block_coord]

    load_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        g_a.element_type,
        num_bits_per_copy=128,
    )
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(),
        g_c.element_type,
        num_bits_per_copy=128,
    )
    tiled_load = cute.make_tiled_copy_tv(load_atom, thread_layout, value_layout)
    tiled_store = cute.make_tiled_copy_tv(store_atom, thread_layout, value_layout)
    load_slice = tiled_load.get_slice(thread_idx)
    store_slice = tiled_store.get_slice(thread_idx)

    thread_a = load_slice.partition_S(block_a)
    thread_b = load_slice.partition_S(block_b)
    thread_c = store_slice.partition_D(block_c)
    fragment_a = cute.make_fragment_like(thread_a)
    fragment_b = cute.make_fragment_like(thread_b)
    fragment_c = cute.make_fragment_like(thread_c)

    cute.copy(load_atom, thread_a, fragment_a)
    cute.copy(load_atom, thread_b, fragment_b)
    fragment_c.store(fragment_a.load() + fragment_b.load())
    cute.copy(store_atom, fragment_c, thread_c)


@cute.jit
def add_aligned_jit(
    tensor_a: cute.Tensor,
    tensor_b: cute.Tensor,
    tensor_c: cute.Tensor,
    copy_bits: cutlass.Constexpr = 128,
):
    vector_size = copy_bits // tensor_a.element_type.width
    thread_layout = cute.make_layout(128)
    value_layout = cute.make_layout(vector_size)
    tiler, layout_tv = cute.make_layout_tv(thread_layout, value_layout)
    tiled_a = cute.zipped_divide(tensor_a, tiler)
    tiled_b = cute.zipped_divide(tensor_b, tiler)
    tiled_c = cute.zipped_divide(tensor_c, tiler)
    add_aligned_kernel(
        tiled_a,
        tiled_b,
        tiled_c,
        thread_layout,
        value_layout,
    ).launch(
        grid=[cute.size(tiled_c, mode=[1]), 1, 1],
        block=[cute.size(layout_tv, mode=[0]), 1, 1],
    )


class ModelNew(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._compiled = None

    def forward(self, tensor_a: torch.Tensor, tensor_b: torch.Tensor) -> torch.Tensor:
        if tensor_a.dtype != torch.bfloat16 or tensor_b.dtype != torch.bfloat16:
            raise TypeError("This example's public ABI is BF16")
        if not tensor_a.is_cuda or not tensor_b.is_cuda or tensor_a.shape != tensor_b.shape:
            raise ValueError("A and B must be equally shaped CUDA tensors")
        if tensor_a.numel() % 1024:
            raise ValueError("The aligned fast path requires a total element count divisible by 1024")
        if not tensor_a.is_contiguous() or not tensor_b.is_contiguous():
            raise ValueError("The aligned fast path requires contiguous inputs")
        tensor_a = tensor_a.detach()
        tensor_b = tensor_b.detach()
        flat_a = tensor_a.reshape(-1)
        flat_b = tensor_b.reshape(-1)
        flat_c = torch.empty_like(flat_a)
        cute_a = from_dlpack(flat_a, assumed_align=16).mark_layout_dynamic()
        cute_b = from_dlpack(flat_b, assumed_align=16).mark_layout_dynamic()
        cute_c = from_dlpack(flat_c, assumed_align=16).mark_layout_dynamic()
        if self._compiled is None:
            self._compiled = cute.compile(add_aligned_jit, cute_a, cute_b, cute_c)
            if artifact_dir := os.environ.get("CUTE_HARNESS_ARTIFACT_DIR"):
                from kernel_evo.cute_harness.artifacts import dump_compiled_artifacts

                dump_compiled_artifacts(
                    self._compiled,
                    artifact_dir,
                    prefix="bf16_vector_add_aligned",
                )
        self._compiled(cute_a, cute_b, cute_c)
        return flat_c.reshape(tensor_a.shape)


def main() -> None:
    torch.manual_seed(0)
    model = ModelNew().cuda()
    results = []
    for shape in ((16, 256), (1024, 1024)):
        tensor_a = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        tensor_b = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        actual = model(tensor_a, tensor_b)
        expected = tensor_a + tensor_b
        max_abs_error = float((actual.float() - expected.float()).abs().max())
        passed = bool(torch.allclose(actual.float(), expected.float(), atol=1e-2, rtol=1e-2))
        results.append({"shape": list(shape), "passed": passed, "max_abs_error": max_abs_error})
    print(json.dumps({"correctness": all(item["passed"] for item in results), "cases": results}))


if __name__ == "__main__":
    main()
