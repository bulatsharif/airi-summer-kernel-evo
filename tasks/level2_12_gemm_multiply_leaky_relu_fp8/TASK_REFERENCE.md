# Task reference: KernelBench Level 2/12

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-dense-gemm-template.py`. Copy its compile-verified dense
FP8 GEMM core without changing its TMA/pipeline/TMEM flow; this task owns only
the operation-specific elementwise epilogue.
Also read `references/candidate-elementwise-template.py` and preserve its
two-dimensional indexing and launch; replace only its neutral expression.

Required structure:

```text
@cute.kernel dense_fp8_gemm_kernel(typed MMA/TMA/layout arguments)
@cute.jit    dense_fp8_gemm(matrix_a, matrix_b_nk, output)
               -> construct objects and launch dense_fp8_gemm_kernel
@cute.kernel multiply_leaky_relu_kernel(output, bias)
@cute.jit    gemm_multiply_leaky_relu(...)
               -> dense_fp8_gemm(...)
               -> launch multiply_leaky_relu_kernel
```

Do not turn `dense_fp8_gemm` into a kernel, wrap it in another kernel, or call a
decorated kernel without `.launch()`.

The exact mathematical order is `Y = leaky_relu((X @ W.T + bias) * 2.0, 0.1)`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.

Use the common reference's exact `(128, 256, 128)` TMEM epilogue to write the
correctly dequantized FP32 GEMM matrix first. Then launch the starter's
elementwise kernel and index output as `output[row, column]` and bias as
`bias[column]`. In-place global read/modify/write is valid after the GEMM
launch; do not allocate an unavailable temporary tensor or fuse the activation
while basic correctness is still failing.

After a concrete remote failure, route the diagnostic through
`references/candidate-error-atlas.md`; do not browse other API families.
