# Complete worked kernels

The fragments in the other files show one boundary at a time. These two are
whole programs, included so the shape of a finished kernel is visible end to end.

Neither is a task solution. The first computes an elementwise epilogue and never
constructs an MMA. The second targets Hopper sm_90a; this Blackwell device
rejects it at the architecture gate, so it can be read for structure but neither
run nor reused here.

## A complete Blackwell FP8 elementwise kernel

Verified on B300: `max_abs_error=0.0`, `device_time_ms=0.044`. It shows the whole
path from decorated kernel to launch, FP8 converted into FP32 for arithmetic, and
a fused epilogue -- with no MMA anywhere. `main()` and its torch validation are
omitted: the harness owns those and a candidate must not define them.

```python
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
```

Note `create_cute_tensor_for_fp8` for building the FP8 view, the `.to(cutlass.Float32)`
before arithmetic, and the branchless ReLU written as a multiply by a predicate.

## The mainloop of a complete Hopper GEMM

Excerpted from a 1,648-line NVIDIA CUTLASS example for `sm_90a`. Blackwell refuses
it -- `OpError: expects arch to be sm_90a, but got sm_103a` -- so the Blackwell MMA
path is a different one and this cannot be copied into a candidate. What transfers
is the shape of a pipelined mainloop: wait for a buffer, issue the MMA across
K-blocks, set the accumulate flag after the first, commit, advance.

```python
# Wait for the A/B buffer to be ready
mainloop_pipeline.consumer_wait(mainloop_consumer_read_state, peek_ab_full_status)

cute.nvgpu.warpgroup.fence()
for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
    k_block_coord = (None, None, k_block_idx, mainloop_consumer_read_state.index)
    tCrA_1phase = tCrA[k_block_coord]
    tCrB_1phase = tCrB[k_block_coord]

    cute.gemm(
        tiled_mma,
        accumulators,
        tCrA_1phase,
        tCrB_1phase,
        accumulators,
    )
    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)

cute.nvgpu.warpgroup.commit_group()
mainloop_consumer_read_state.advance()
```

Three things generalize. The accumulator is both a source and a destination of
`cute.gemm`. The accumulate field is set *after* the first K-block, so the first
iteration overwrites and later ones add. And the K loop runs inside a buffer a
pipeline hands over, rather than being a bare loop over global memory.

The Blackwell equivalent uses a different namespace and a different accumulate
field; the API sections of the documentation name both.
