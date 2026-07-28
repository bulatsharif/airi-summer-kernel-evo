# Task reference: dense FP8 GEMM followed by GELU and row softmax

Load the `cute-fp8-kernels` skill and reuse its compile-verified dense GEMM
flow. Restore FP8 scales once before adding FP32 bias.
Read the skill's `references/candidate-math-api.md` before implementing GELU
or softmax.

Exact GELU is `0.5*x*(1 + erf(x/sqrt(2)))`. `cute.erf` and `cute.exp` are
available on this worker; explicitly cast their arithmetic arguments to
`cutlass.Float32` when necessary.

For correctness, a one-warp-per-row post-op is valid: compute a warp maximum,
then a warp sum of `exp(gelu(x)-maximum)`, then write normalized values. Use
`cute.arch.shuffle_sync_bfly` for both reductions. Recomputing GELU in each
pass is acceptable because performance is not an acceptance condition.

Required ABI:

```text
@cute.jit matmul_gelu_softmax(matrix_a, matrix_b_nk, bias, scratch, output)
```
