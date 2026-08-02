# Task: complete GPT-2 Small transformer block, CuTe FP8

Implement one complete inference-time GPT-2 Small decoder block in CuTe DSL.
The timed entrypoint must perform all of the following, in order:

```text
n1       = layer_norm(hidden, ln1_weight, ln1_bias)
q, k, v  = split(linear_fp8(n1, qkv_weight, qkv_bias))
context  = causal_softmax(q @ k.T / sqrt(64)) @ v
residual = hidden + linear_fp8(context, out_weight, out_bias)
n2       = layer_norm(residual, ln2_weight, ln2_bias)
mlp      = linear_fp8(gelu_new(linear_fp8(n2, fc_weight, fc_bias)),
                      proj_weight, proj_bias)
output   = residual + mlp
```

This is the whole `transformer.h[0]` block: both LayerNorms, causal multi-head
attention, all four dense projections, GELU, and both residual connections.
Dropout and KV caching are intentionally absent.

## Fixed GPT-2 Small shape

```text
tokens = 128       hidden = 768       heads = 12
head_dim = 64      mlp = 3072         qkv = 2304
```

Batch size is one, so the token dimension is also the sequence dimension.
All dimensions are exact multiples of common tensor-core tiles; no tail path
is required.

## Precision contract

- Hidden state and all four weight matrices are physically FP8 E4M3FN.
- LayerNorm output, packed QKV, attention context, and GELU output are
  requantized to FP8 using the fixed scales in `starter.py`.
- Dense accumulation, LayerNorm statistics, attention scores/softmax,
  biases, residuals, and final output are FP32.
- The reference applies the same intermediate FP8 rounding. Do not silently
  keep an intermediate in FP32 across a declared FP8 boundary.
- Weights use `[N,K]` packed layout and must not be transposed in candidate
  storage.

## Candidate ABI

Preserve the `gpt2_transformer_block` signature in `starter.py`. The supplied
workspace tensors are owned by the evaluator and may be used, overwritten, or
ignored, but the output must be written by launched CuTe GPU kernels.

The private baseline is a deliberately simple scalar CuTe implementation.
Optimization is the point of the task: tensor-core `cute.gemm`, fused
epilogues, fewer materialized workspaces, and fused attention are all allowed
without changing the ABI or numerical contract.

Do not define `main()`, call PyTorch, allocate test inputs, or print a PASS
marker. The harness owns compilation, validation, and whole-block timing.

```text
python3 -m cute_harness check model_gpt2_small_transformer_block_fp8 submission.py
python3 -m cute_harness run model_gpt2_small_transformer_block_fp8 submission.py
```
