# Correctness-first FP8 reductions on CUTLASS 4.6.1

Use this route for LayerNorm and other row reductions before considering
shared-memory or multi-warp optimization. The pattern below compiled, launched,
and passed numerical validation on the shared B300 on 2026-07-27.

## One warp per row

When the row length is divisible by 32, one warp can stream through the row.
Each lane owns columns `iteration * 32 + lane`. Keep the accumulator in FP32.

```python
THREADS = 32


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def row_kernel(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    partial = cutlass.Float32(0.0)
    for iteration in cutlass.range(ROW_SIZE // THREADS):
        column = iteration * THREADS + lane
        value = input_tensor[row, column].to(cutlass.Float32) * INPUT_SCALE
        partial += value
    mean = warp_sum(partial) / ROW_SIZE

    # Make a second streaming pass for centered variance, reduce it with the
    # same warp_sum, then compute cute.rsqrt(variance + EPSILON). Make a third
    # pass and write every output_tensor[row, column].


@cute.jit
def operation(input_tensor: cute.Tensor, output_tensor: cute.Tensor):
    row_kernel(input_tensor, output_tensor).launch(
        grid=(ROWS, 1, 1),
        block=(THREADS, 1, 1),
    )
```

Important details:

- Device index functions take no arguments and return `(x, y, z)` tuples.
- Convert FP8 to `cutlass.Float32` before applying the task's dequantization
  scale. Do not normalize the raw FP8 magnitude.
- Use the centered variance from a second pass. `E[x*x] - E[x]**2` can lose
  precision.
- Call the decorated kernel as `kernel(args...).launch(...)`; assigning a
  `.grid` attribute or calling the kernel without `.launch()` does not execute
  it.
- A single-warp streaming baseline is deliberately slow but simple. Optimize
  with multiple warps/shared memory only after this version passes.
