# Task reference: FP8 GEMM + Bias + ReLU

Load the `cute-fp8-kernels` skill listed in `task.json.agent_skills` first.
Before coding, read the skill's
`.opencode/skills/cute-fp8-kernels/references/candidate-gemm-api.md` chapter.
It contains the exact candidate-mode dense GEMM signatures installed on the
worker.

## Data contract

- A: `[1024, 8192]`, FP8 storage.
- B is supplied as `B_nk`: `[8192, 8192]`, FP8 storage.
- bias: `[8192]`, FP32.
- output: `[1024, 8192]`, FP32.
- Result: `relu(A @ B_nk.T + bias)`.

## Recommended decomposition

First establish a correct FP8 GEMM using the Blackwell flow in the common
reference. A known-compatible starting design is a single-CTA
`(128, 256, 128)` MMA tile with 128 threads, four A/B stages and FP32 TMEM
accumulation.

Keep the GEMM and activation concerns separate initially:

1. GEMM kernel writes the correctly dequantized FP32 matrix.
2. A simple 1D kernel adds `bias[column]` and clamps negative values to zero.

This makes correctness/debugging easier than fusing the epilogue immediately.
Once correct, fusion is an optional optimization.

The two operands have different input scales. Restore the product of the A and
B scales exactly once when copying the accumulator to output. Bias is already
FP32 and is added after GEMM scaling.

Construct `cute.nvgpu.make_tiled_tma_atom_A/B` in the JIT launcher exactly as
shown by the common reference and pass the returned atoms and tensor views into
the kernel. The helpers are not members of `cpasync`, need no
`TmaOperandMajorMode`, and must not be moved into device code. Construct
`mma_global_a/b/c` before calling `tma_partition`.
