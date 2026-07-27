# Task: GEMM + Multiply + LeakyReLU, CuTe FP8

Adapt KernelBench Level 2 task 12 to a correctness-first CuTe DSL candidate:

```text
X:    [1024, 8192]
W:    [8192, 8192]
bias: [8192]

linear = X @ W.T + bias
scaled = linear * 2.0
Y = scaled                    when scaled >= 0
Y = scaled * 0.1              when scaled < 0
```

The evaluator passes `matrix_a = X` with shape `[M,K]` and
`matrix_b_nk = W` with physical shape `[N,K]`. The right operand is already in
the N-K order used by the documented Blackwell dense GEMM path; do not
transpose it inside the candidate.

## Precision contract

- X and W storage is FP8 E4M3FN.
- GEMM accumulation is FP32.
- Bias and output are FP32.
- Convert the raw FP8 GEMM accumulator to the task value with
  `raw * SCALE_A * SCALE_B` before adding bias.
- Apply operations in the declared order: add bias, multiply by `2.0`, then
  LeakyReLU with negative slope `0.1`.
- A correctness-first implementation may use two launches: dense FP8 GEMM,
  followed by the elementwise epilogue. Speed is not scored.

## Candidate ABI

- Edit only the prepared `submission.py`.
- Keep at least two `@cute.kernel` functions, one `@cute.jit`, and a real
  `cute.gemm` operation.
- `gemm_multiply_leaky_relu(matrix_a, matrix_b_nk, bias, output)` is the
  evaluator entrypoint and must be decorated with `@cute.jit`.
- Candidate output must be fully computed and written by launched CuTe kernels.
- Do not define/call `main()`, create evaluator inputs, call PyTorch, compute a
  reference, or print a PASS marker. The harness appends those owned parts.

## Acceptance

- Full max absolute error versus the PyTorch FP8 pipeline: `<= 0.01`.
- Sample max absolute error versus the original FP32 pipeline: `<= 0.2`.
- Every output value must be finite.

## Iteration loop

```text
python -m cute_harness check level2_12_gemm_multiply_leaky_relu_fp8 submission.py
python -m cute_harness run level2_12_gemm_multiply_leaky_relu_fp8 submission.py
```
