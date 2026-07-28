# Task: Matmul + GELU + Softmax, CuTe FP8

Implement `Y = softmax(gelu(X @ W.T + bias), dim=1)` for
`X=[1024,8192]`, `W=[8192,8192]`, and `bias=[8192]`.

X/W use FP8 E4M3FN storage; GEMM accumulation, bias, scratch, and output are
FP32. GELU uses the exact erf definition rather than the tanh approximation.
Softmax is over each complete 8192-element row and must be numerically stable.

Expose
`matmul_gelu_softmax(matrix_a, matrix_b_nk, bias, scratch, output)` as the
`@cute.jit` entrypoint. Scratch and output are both `[1024,8192]`. Use at least
two CuTe kernels and do not define `main()`.

```powershell
python -m cute_harness check level2_99_matmul_gelu_softmax_fp8 submission.py
python -m cute_harness run level2_99_matmul_gelu_softmax_fp8 submission.py
```
