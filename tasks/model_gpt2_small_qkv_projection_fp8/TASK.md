# Task: GPT-2 Small QKV projection, CuTe FP8

Implement the combined query, key, and value projection from a GPT-2 Small
attention block:

```text
X:        [8192, 768]
W_qkv_nk: [2304, 768]
bias_qkv: [2304]

Y = SCALE_X * SCALE_W * (X @ W_qkv_nk.T) + bias_qkv
```

`8192` is batch 8 times sequence length 1024. The output contains contiguous
`Q`, `K`, and `V` regions of width 768 each.
The evaluator uses deterministic representative tensors rather than checkpoint
assets; the layer equation, dimensions, and packed inference layout are
architecture-faithful.

## Precision and layout contract

- `X` and `W_qkv_nk` are physically FP8 E4M3FN.
- `W_qkv_nk` is already packed as `[N,K]`; do not transpose it in the
  candidate.
- MMA accumulation, bias, and output are FP32.
- Restore `SCALE_X * SCALE_W` exactly once before adding bias.
- Every dimension is an exact multiple of the supported tensor-core tiles; no
  tail path is required.

## Candidate ABI

Preserve the `gpt2_qkv_projection` JIT entrypoint declared in `starter.py`.
Candidate code must launch CuTe kernels and contain a real `cute.gemm`.
It may use a separate bias kernel or fuse bias into the GEMM epilogue.

Do not define `main()`, use PyTorch, create inputs or references, or print a
PASS marker. The harness owns compilation, validation, and timing.

Correctness is a hard gate. Performance is reported as speedup over the private
baseline.

```text
python3 -m cute_harness check model_gpt2_small_qkv_projection_fp8 submission.py
python3 -m cute_harness run model_gpt2_small_qkv_projection_fp8 submission.py
```
