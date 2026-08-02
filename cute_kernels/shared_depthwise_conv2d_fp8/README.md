# Shared-memory depthwise Conv2d, FP8

```text
x: [8, 64, 256, 256], FP8 E4M3FN
w: [64, 1, 3, 3], FP8 E4M3FN
y = depthwise_conv2d(x, w, padding=1): FP32
```

A 256-thread CTA computes an `8 x 32` output tile. It cooperatively loads the
corresponding `10 x 34` input tile and nine channel weights into shared memory,
then completely unrolls the 3x3 stencil. The halo load replaces up to 2,304
global input reads per output tile with 340 coalesced reads.

The dimensions intentionally differ from the repository's transposed 3D
convolution task. This example teaches tiled stencil reuse, zero padding,
group/channel indexing, FP8 conversion, and compile-time loop unrolling.

Run on a Blackwell CUDA host with CUTLASS 4.6.1:

```bash
python submission.py
```

B300 validation (20 timed launches after five warmups):

```text
max_abs=0.000000
mean_abs=0.000000023
kernel_time_ms=0.188059
profile_id=2d4e1a23-b75b-4a67-81c3-5a78bf43aa34
PASS
```

License: [NVIDIA_BSD_3_CLAUSE.txt](../NVIDIA_BSD_3_CLAUSE.txt).
