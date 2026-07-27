# Task: GEMM + BiasAdd + ReLU, CuTe FP8

Implement candidate code for:

```text
X:    [1024, 8192]
W:    [8192, 8192]
bias: [8192]
Y = relu(X @ W.T + bias)
```

The evaluator passes `matrix_a = X` with shape `[M,K]` and
`matrix_b_nk = W` with physical shape `[N,K]`. The right operand is therefore
already stored in the N-K order expected by the Blackwell dense GEMM path.

## Precision contract

- X and W storage is FP8 E4M3FN.
- GEMM accumulation is FP32.
- Bias and output are FP32.
- The FP8 GEMM result must be multiplied by `SCALE_A * SCALE_B` before adding
  bias and applying ReLU.
- A correctness-first implementation may use two CuTe launches: FP8 GEMM,
  followed by BiasAdd + ReLU. Speed is not scored yet.

## Candidate ABI

- Edit only the prepared `submission.py`.
- Keep at least two `@cute.kernel` functions, one `@cute.jit`, and a real
  `cute.gemm` operation.
- `gemm_add_relu(matrix_a, matrix_b_nk, bias, output)` is the evaluator entry
  point and must be decorated with `@cute.jit`.
- Candidate output must be fully computed and written by launched CuTe kernels.
- Do not define/call `main()`, create inputs, compute a PyTorch reference, or
  print a PASS marker. The harness appends those evaluator-owned parts.

## Acceptance

- Full max absolute error versus the PyTorch FP8 pipeline: `<= 0.01`.
- Sample max absolute error versus the original FP32 pipeline: `<= 0.1`.

## Iteration loop

```text
python -m cute_harness check level2_76_gemm_add_relu_fp8 submission.py
python -m cute_harness run level2_76_gemm_add_relu_fp8 submission.py
```
