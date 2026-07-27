"""Numerically verified neutral scale + bias + ReLU epilogue.

This standalone documentation example uses two rows and does not implement the
Level 2 GEMM. Benchmark candidates must not copy main() or torch validation.
"""

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


ROWS = 2
COLS = 1024
THREADS = 128
SCALE = 0.125
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def scale_bias_relu_kernel(
    input_tensor: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(COLS // THREADS):
        column = iteration * THREADS + thread_idx
        value = input_tensor[row_idx, column].to(cutlass.Float32) * SCALE
        value = value + bias[column].to(cutlass.Float32)
        output[row_idx, column] = value * (value > 0.0)


@cute.jit
def scale_bias_relu(
    input_tensor: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    scale_bias_relu_kernel(input_tensor, bias, output).launch(
        grid=(ROWS, 1, 1),
        block=(THREADS, 1, 1),
    )


def main():
    torch.manual_seed(20260727)
    source = torch.randn((ROWS, COLS), device="cuda", dtype=torch.float32)
    bias = torch.randn((COLS,), device="cuda", dtype=torch.float32)
    storage = torch.empty((ROWS, COLS), device="cuda", dtype=torch.uint8)
    output = torch.empty((ROWS, COLS), device="cuda", dtype=torch.float32)
    input_tensor = create_cute_tensor_for_fp8(storage, FP8_DTYPE, 1, source)
    bias_tensor = from_dlpack(bias)
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = cute.compile(
        scale_bias_relu,
        input_tensor,
        bias_tensor,
        output_tensor,
    )
    compiled(input_tensor, bias_tensor, output_tensor)
    reference = torch.relu(
        storage.view(torch.float8_e4m3fn).float() * SCALE + bias
    )
    torch.testing.assert_close(output, reference, atol=0.0, rtol=0.0)
    print("example_elementwise_scale_bias_relu=ok max_abs_error=0.0")


main()
