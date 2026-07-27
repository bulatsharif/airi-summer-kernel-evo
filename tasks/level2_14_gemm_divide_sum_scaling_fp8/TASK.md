# Task: GEMM + Divide + Sum + Scaling, CuTe FP8

Implement the correctness-first operation:

```text
X: [1024,8192], W: [8192,8192]
scratch = X @ W.T
Y = sum(scratch / 2.0, dim=1, keepdim=True) * 1.5
```

The evaluator supplies FP32 `scratch: [1024,8192]` and `output: [1024,1]`.
Use FP8 E4M3FN inputs, FP32 GEMM accumulation and an FP32 row reduction.
Restore `SCALE_A * SCALE_B` exactly once before or during the reduction.
Candidate code must launch CuTe kernels, contain a real `cute.gemm`, and must
not define `main()` or use PyTorch.
