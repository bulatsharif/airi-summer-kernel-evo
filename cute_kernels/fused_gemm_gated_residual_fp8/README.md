# Fused FP8 GEMM + gate + residual

```text
A:        [2048, 4096], FP8 E4M3FN
B_nk:     [4096, 4096], FP8 E4M3FN (logical B transpose)
gate:     [2048, 4096], FP32
residual: [2048, 4096], FP32
Y = dequant(A @ B_nk.T) * gate + residual: FP32
```

Each CTA computes a `128 x 256` tile with Blackwell `tcgen05` FP8 MMA. TMA
feeds a four-stage shared-memory pipeline while the accumulator remains in
TMEM. The epilogue reads one quarter-tile at a time and performs the gate and
residual operations in registers before the only output store.

This is deliberately not a repository task answer: its elementwise gate is a
full matrix and its two FP32 operands are fused directly into the tensor-core
epilogue. It demonstrates how identically partitioned C-shaped tensors can be
mapped onto the TMEM copy's per-thread ownership.

The standalone program validates every result against `torch._scaled_mm`, then
warms up and reports CUDA-event time.

Run on a Blackwell CUDA host with CUTLASS 4.6.1:

```bash
python submission.py
```

B300 validation (20 timed launches after five warmups):

```text
max_abs=0.000001
mean_abs=0.000000020
kernel_time_ms=0.140909
profile_id=e43e1fce-1a5f-48a4-b56f-d067220a7f24
PASS
```

The GEMM pipeline is adapted from NVIDIA's CuTe DSL Blackwell tutorial. License:
[NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
