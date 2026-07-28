# Task: Matmul + Mish + Mish, CuTe FP8

Implement `Y = mish(mish(X @ W.T + bias))` for `X=[1024,8192]`,
`W=[8192,8192]`, and `bias=[8192]`.

X and W use FP8 E4M3FN storage, GEMM accumulates in FP32, and bias/output are
FP32. The candidate must use CuTe kernels and expose
`matmul_mish_mish(matrix_a, matrix_b_nk, bias, output)` as an `@cute.jit`
entrypoint. Keep at least two `@cute.kernel` functions and do not define
`main()`; the harness owns inputs, timing, and validation.

Correctness thresholds are 0.02 max absolute error against the PyTorch FP8
pipeline and 0.2 on a sample against the original FP32 pipeline. Performance
is reported but is not an acceptance condition.

Run:

```powershell
python -m cute_harness check level2_29_matmul_mish_mish_fp8 submission.py
python -m cute_harness run level2_29_matmul_mish_mish_fp8 submission.py
```
