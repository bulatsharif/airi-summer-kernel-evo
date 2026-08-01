# Task reference: dense FP8 GEMM followed by pooling and reduction

Load the `cute-fp8-kernels` skill first and reuse its compile-verified dense
GEMM core. Restore the product of FP8 input scales exactly once, then add FP32
bias in the reduction kernel.

The post-op consumes adjacent pairs `(2*i, 2*i+1)`, keeps their maximum, sums
all 16384 maxima per row, and scales the result by 0.5. A correctness-first
implementation can give one warp to each row: each lane accumulates strided
pairs and a `cute.arch.shuffle_sync_bfly` tree reduces the 32 partial sums.

Required ABI:

```text
@cute.jit matmul_maxpool_sum_scale(matrix_a, matrix_b_nk, bias, scratch, output)
```
