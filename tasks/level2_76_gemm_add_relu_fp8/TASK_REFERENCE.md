# Task reference: FP8 GEMM + Bias + ReLU

Load the `cute-fp8-kernels` skill listed in `task.json.agent_skills` first.
Before coding, read the skill's
`.opencode/skills/cute-fp8-kernels/references/candidate-dense-gemm-template.py`.
Copy its compile-verified 4.6.1 dense GEMM core without changing the
TMA/pipeline/TMEM flow.
Also read
`.opencode/skills/cute-fp8-kernels/references/candidate-elementwise-template.py`
and preserve its two-dimensional indexing and launch; replace only its neutral
expression.

Required structure:

```text
@cute.kernel dense_fp8_gemm_kernel(typed MMA/TMA/layout arguments)
@cute.jit    dense_fp8_gemm(matrix_a, matrix_b_nk, output)
               -> construct objects and launch dense_fp8_gemm_kernel
@cute.kernel bias_relu_kernel(output, bias)
@cute.jit    gemm_add_relu(...)
               -> dense_fp8_gemm(...)
               -> launch bias_relu_kernel
```

It is valid to replace the starter's internal `fp8_gemm_kernel` name. Do not
add an `@cute.kernel` wrapper around `dense_fp8_gemm`, and never call a
decorated kernel without `.launch()`. Implement ReLU as
`summed * (summed > 0.0)`; `cutlass.relu`, `cute.fmax`, and `cute.maximum` do
not exist on this worker.

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
shown by the template and pass `.atom` and `.tma_tensor` from the returned
descriptor objects into the kernel. The helpers are not members of `cpasync`,
need no `TmaOperandMajorMode`, and must not be moved into device code.
Construct `mma_global_a/b/c` before calling `tma_partition`.

After a concrete remote failure, route the diagnostic through
`references/candidate-error-atlas.md`; do not browse other API families.
