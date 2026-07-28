# Task: Matmul + MaxPool + Sum + Scale, CuTe FP8

Implement a dense linear layer for `X=[128,32768]`, `W=[32768,32768]`, and
`bias=[32768]`. Treat each linear-output row as a length-32768 1D signal,
apply non-overlapping max pooling with kernel and stride 2, sum all pooled
values, and multiply the scalar for each row by 0.5.

X/W use FP8 E4M3FN storage; accumulation, bias, scratch, and final output are
FP32. Expose
`matmul_maxpool_sum_scale(matrix_a, matrix_b_nk, bias, scratch, output)`.
The scratch matrix is `[128,32768]` and output is `[128]`.

Use CuTe kernels, at least two `@cute.kernel` functions, and at least one
`@cute.jit` function. Do not define `main()`.

```powershell
python -m cute_harness check level2_55_matmul_maxpool_sum_scale_fp8 submission.py
python -m cute_harness run level2_55_matmul_maxpool_sum_scale_fp8 submission.py
```
