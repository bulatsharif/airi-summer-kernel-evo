# Task reference: dense FP8 GEMM followed by Mish twice

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`. Reuse the
compile-verified dense GEMM core and its exact TMA/pipeline/TMEM flow. Restore
the product of the two FP8 input scales exactly once before adding FP32 bias.
Read the skill's `references/candidate-math-api.md` before implementing Mish.

Use a separate elementwise kernel for the post-op. Mish is
`x * tanh(log(1 + exp(x)))`. On this worker `cute.exp`, `cute.log`, and
`cute.tanh` are available, but chained scalar intrinsics need an explicit FP32
materialization at every boundary:

```python
value = cutlass.Float32(value)
softplus = cute.log(cutlass.Float32(1.0 + cute.exp(value)))
result = cutlass.Float32(
    value * cute.tanh(cutlass.Float32(softplus))
)
```

Do not wrap these scalar values in a temporary CuTe tensor or construct a
one-element layout.

Required ABI:

```text
@cute.jit matmul_mish_mish(matrix_a, matrix_b_nk, bias, output)
```

Inputs A and B_nk have shapes `[1024,8192]` and `[8192,8192]`; output is
`[1024,8192]` FP32. B is supplied in N-by-K order, so GEMM computes A @ B_nk.T.
