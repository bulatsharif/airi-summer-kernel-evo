# Qwen3.5-4B full-attention block reference

This task models a `full_attention` text layer from the official
`Qwen/Qwen3.5-4B` configuration. Qwen3.5-4B alternates three linear-attention
layers with one full-attention layer; this task selects the latter because it
is a tractable first complete Qwen layer for kernel evolution.

Important layout details:

- `q_gate_weight`: `[8192,2560]`; reshape its output to
  `[tokens,16,2,256]`, with Q before gate within each head.
- `k_weight`, `v_weight`: `[1024,2560]` and four KV heads.
- Q/K RMSNorm is independently computed over each 256-wide head, with one
  shared 256-element weight vector for Q and another for K.
- Partial RoPE rotates only dimensions `[0,64)` using `rotate_half`; dimensions
  `[64,256)` pass through unchanged.
- Attention head `h` uses KV head `h // 4` and a causal mask.
- The attention gate is applied after value aggregation and before flattening
  for the `[2560,4096]` output projection.
- The MLP is bias-free SwiGLU with two `[9216,2560]` input projections and one
  `[2560,9216]` down projection.

Start from the dense GEMM template for the projections, then use the
elementwise template and reduction patterns for normalization, gating, and
attention. Optimize the timed block as one program: materializing every public
workspace is not required.
