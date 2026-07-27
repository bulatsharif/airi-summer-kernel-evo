# Task reference: FP8 LayerNorm

Read `references/CUTE_DSL_REFERENCE.md` first.

Treat the input as 16 independent rows, each containing:

```text
64 * 256 * 256 = 4,194,304 values
```

The supplied storage is FP8 E4M3FN and the output is FP32. Convert each load to
FP32 and apply `INPUT_SCALE` before accumulation.

## Stable low-memory design

A practical implementation uses one 256-thread CTA per row and streams the row
three times:

1. Accumulate the FP32 sum and reduce it to the mean.
2. Accumulate centered squared deviations and reduce to the variance.
3. Normalize with `cute.rsqrt(variance + epsilon)` and write FP32.

Do not materialize the full row in registers.

## CTA reduction

For each reduction:

1. Reduce within each 32-thread warp with
   `cute.arch.shuffle_sync_bfly` offsets `16, 8, 4, 2, 1`.
2. Lane zero writes one partial to a small FP32 SMEM scratch tensor.
3. Synchronize the CTA.
4. Warp zero reduces the warp partials.
5. Publish the CTA result in scratch and synchronize before all threads read.

Required primitives include `cute.arch.shuffle_sync_bfly`, `cute.rsqrt`,
`cute.arch.sync_threads`, `utils.SmemAllocator`, `cutlass.range` and a normal
kernel launch.
