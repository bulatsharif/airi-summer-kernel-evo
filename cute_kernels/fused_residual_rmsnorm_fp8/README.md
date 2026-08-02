# Fused residual RMSNorm, FP8 input

```text
x:        [4096, 4096], FP8 E4M3FN
residual: [4096, 4096], FP32
weight:   [4096], FP32
y = RMSNorm(dequant(x) + residual) * weight: FP32
```

One 256-thread CTA owns a row. Eight warps stream through the row twice, keep
FP32 partial sums in registers, and use only nine FP32 shared-memory values for
the cross-warp reduction. Fusing the residual addition avoids materializing a
16.8-million-element intermediate tensor.

The file is standalone: it creates FP8 storage, compiles the JIT entrypoint,
checks every output against PyTorch, warms up, and reports CUDA-event time.

Run on a Blackwell CUDA host with CUTLASS 4.6.1:

```bash
python submission.py
```

B300 validation (20 timed launches after five warmups):

```text
max_abs=0.000002
mean_abs=0.000000022
kernel_time_ms=0.034174
profile_id=8a9a8b6b-ba5f-4400-979f-8fef7fe355ab
PASS
```

License: [NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
