# Task: Matmul plus Scaling plus ResidualAdd with CuTe FP8

Implement this correctness-first CuTe DSL operation:

```text
X: [1024, 8192], W: [8192, 8192], bias: [8192]
linear = X @ W.T + bias; Y = linear * 0.5 + linear
```

`matrix_b_nk` is physically `[N,K]`; do not transpose it in the candidate.
FP8 E4M3FN inputs must accumulate in FP32. Restore `SCALE_A * SCALE_B`
exactly once before adding FP32 bias, then preserve the linear value, multiply it by `0.5`, then add the preserved value.

Use the candidate ABI declared in `starter.py`. Candidate code must launch CuTe
kernels, must contain a real `cute.gemm`, and must not define `main()` or use
PyTorch. Correctness is scored before performance.
