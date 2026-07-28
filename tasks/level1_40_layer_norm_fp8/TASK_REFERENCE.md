# Task reference: FP8 LayerNorm

Load the `cute-fp8-kernels` skill listed in `task.json.agent_skills` first.

Treat the input as 16 independent rows, each containing:

```text
64 * 256 * 256 = 4,194,304 values
```

The supplied storage is FP8 E4M3FN and the output is FP32. Convert each load to
FP32 and apply `INPUT_SCALE` before accumulation.

Before coding, read the skill's
`.opencode/skills/cute-fp8-kernels/references/reductions.md`. Use its verified
one-warp-per-row pattern exactly for the correctness-first version.

Launch one 32-thread warp per row and stream the row three times:

1. Accumulate the FP32 sum and reduce it to the mean.
2. Accumulate centered squared deviations and reduce to the variance.
3. Normalize with `cute.rsqrt(variance + epsilon)` and write FP32.

Do not materialize the full row in registers.

## Exact lane ownership

Each lane must own distinct columns:

```python
lane, _, _ = cute.arch.thread_idx()
for iteration in cutlass.range(ROW_SIZE // 32):
    column = iteration * 32 + lane
```

Do not assign a full `ROW_SIZE // 32` interval to every lane or warp and then
loop over that interval without adding the lane. That repeats every value 32
times. With a single warp there is no SMEM second-stage reduction: reduce the
FP32 scalar directly with `cute.arch.shuffle_sync_bfly` offsets
`16, 8, 4, 2, 1`.

Every lane receives the reduced mean and variance from the butterfly result,
so every lane can perform its third-pass stores. Use `cutlass.range`, not
`range_constexpr`, for the 131072 streaming iterations.

Keep the launchable `layer_norm_kernel` decorated with exactly `@cute.kernel`.
The reduction helper may be `@cute.jit`, but keep `layer_norm_kernel` decorated
with exactly `@cute.kernel`. Changing it to `@cute.jit` makes the local policy
check report zero kernels.
