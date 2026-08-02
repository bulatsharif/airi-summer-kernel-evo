# Task: complete Qwen3.5-4B full-attention block, CuTe FP8

Implement one complete inference-time `full_attention` decoder block from
Qwen3.5-4B in CuTe DSL. The timed entrypoint covers both pre-norm residual
branches:

```text
n1       = rms_norm(hidden, input_norm_weight)
q, gate  = split_heads(linear_fp8(n1, q_gate_weight))
k        = linear_fp8(n1, k_weight)
v        = linear_fp8(n1, v_weight)
q, k     = partial_rope(rms_norm_head(q), rms_norm_head(k), cos, sin)
context  = causal_gqa_softmax(q @ k.T / 16) @ v
attn     = linear_fp8(context * sigmoid(gate), out_weight)
residual = hidden + attn
n2       = rms_norm(residual, post_norm_weight)
mlp      = linear_fp8(silu(linear_fp8(n2, gate_weight))
                      * linear_fp8(n2, up_weight), down_weight)
output   = residual + mlp
```

This is a whole decoder layer, not an isolated projection or activation.
Dropout and KV caching are intentionally absent.

## Published Qwen3.5-4B dimensions

```text
tokens = 128                 hidden = 2560
query heads = 16             key/value heads = 4
head_dim = 256               rotary_dim = 64
intermediate = 9216          attention scale = 1 / 16
```

The evaluator supplies precomputed FP32 `cos` and `sin`. RoPE affects the first
64 dimensions of each head. Each KV head is shared by four query heads. Q and
its output gate are interleaved per head in the `[128,8192]` Q projection.
RMSNorm parameters use Qwen's `1 + weight` convention. `q_norm_weight` and
`k_norm_weight` are shared `[256]` vectors. RoPE uses Hugging Face
`rotate_half`, pairing `d` with `d+32`, not adjacent dimensions.

## Precision contract

- Hidden state and all seven weight matrices are physically FP8 E4M3FN.
- Both block-normalization outputs, Q/gate/K/V projections, normalized rotary
  Q/K, gated attention context, both MLP projections, and the SwiGLU product
  are requantized at the fixed scales in `starter.py`.
- GEMM accumulation, RMSNorm statistics, RoPE, attention scores and softmax,
  sigmoid/SiLU computation, residuals, and final output are FP32.
- Weight layout is `[N,K]`; do not transpose candidate storage.
- The owned Torch reference applies the same intermediate FP8 rounding.

## Candidate ABI

Preserve the `qwen35_full_attention_block` signature. Evaluator-owned workspace
tensors may be used, overwritten, or ignored. The result must be written by
launched CuTe GPU kernels.

The private baseline is deliberately scalar. Tensor-core GEMMs, fused
normalization/projection epilogues, online attention, workspace elimination,
and fusion across the residual branches are allowed without changing the ABI
or precision contract.

Do not define `main()`, call PyTorch, allocate test inputs, or print a PASS
marker. The harness owns compilation, validation, and whole-block timing.

```text
python3 -m cute_harness check model_qwen35_4b_full_attention_block_fp8 submission.py
python3 -m cute_harness run model_qwen35_4b_full_attention_block_fp8 submission.py
```
