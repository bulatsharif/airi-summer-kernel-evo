"""Compile-verified candidate-only two-dimensional elementwise pattern.

Adapt only the expression assigned to `result` and the public function names.
The input/output matrix is `[M, N]`; bias is `[N]`.
"""

import cutlass
import cutlass.cute as cute


M = 1024
N = 8192
THREADS_PER_CTA = 128
ACC_DTYPE = cutlass.Float32


@cute.kernel
def elementwise_kernel(
    output: cute.Tensor,
    bias: cute.Tensor,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    columns_per_thread = N // THREADS_PER_CTA

    for column_block in cutlass.range(columns_per_thread):
        column = column_block * THREADS_PER_CTA + thread_idx
        value = output[row, column].to(ACC_DTYPE)
        bias_value = bias[column].to(ACC_DTYPE)

        # Replace only this neutral expression with the task's declared order.
        # ReLU: summed = value + bias_value
        #       result = summed * (summed > 0.0)
        # LeakyReLU(slope): result = summed * (
        #       (summed >= 0.0) + slope * (summed < 0.0)
        #   )
        # Do not invent cutlass.relu, cute.fmax, cute.maximum, break, while,
        # thread_rank(), or num_threads().
        result = value + bias_value
        output[row, column] = result


@cute.jit
def run_elementwise(
    output: cute.Tensor,
    bias: cute.Tensor,
):
    elementwise_kernel(output, bias).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )
