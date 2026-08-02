# Task: complete Qwen3.5-4B vision block, CuTe FP8

Implement one complete inference-time vision transformer block from
Qwen3.5-4B in CuTe DSL:

```text
n1       = layer_norm(hidden, norm1_weight, norm1_bias)
q, k, v  = split(linear_fp8(n1, qkv_weight, qkv_bias))
q, k     = rope(q, k, cos, sin)
context  = softmax(q @ k.T / 8) @ v
residual = hidden + linear_fp8(context, out_weight, out_bias)
n2       = layer_norm(residual, norm2_weight, norm2_bias)
mlp      = linear_fp8(gelu_tanh(linear_fp8(n2, fc1_weight, fc1_bias)),
                      fc2_weight, fc2_bias)
output   = residual + mlp
```

This is the entire `visual.blocks[0]` residual block. Attention is non-causal.
The evaluator provides deterministic FP32 `cos` and `sin`, just as the model
provides precomputed rotary embeddings to each block.

## Published Qwen3.5 vision dimensions

```text
tokens = 128          hidden = 1024       heads = 16
head_dim = 64         intermediate = 4096 qkv = 3072
```

## Precision contract

- Hidden state and all four weights are physically FP8 E4M3FN.
- LayerNorm outputs, packed QKV before and after RoPE, attention context, and
  GELU output are requantized with the fixed scales in `starter.py`.
- GEMM accumulation, LayerNorm statistics, RoPE, attention scores/softmax,
  biases, residuals, and final output are FP32.
- Weights use `[N,K]` layout; do not transpose candidate storage.
- The owned Torch reference applies identical intermediate FP8 rounding.

## Candidate ABI

Preserve `qwen35_vision_block`. Workspaces are evaluator-owned and may be used,
overwritten, or ignored. The result must be written by launched CuTe kernels.
Fusion, tensor-core GEMMs, online attention, and workspace elimination are all
allowed while preserving the numerical contract.

Do not define `main()`, call PyTorch, allocate test inputs, or print a PASS
marker. The harness owns compilation, validation, and whole-block timing.

```text
python3 -m cute_harness check model_qwen35_4b_vision_block_fp8 submission.py
python3 -m cute_harness run model_qwen35_4b_vision_block_fp8 submission.py
```
