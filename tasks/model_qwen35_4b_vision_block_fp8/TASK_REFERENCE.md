# Qwen3.5-4B vision block reference

This task follows `Qwen3_5VisionBlock` from the official Transformers
implementation:

- two pre-norm residual branches using LayerNorm with epsilon `1e-6`;
- biased `[3072,1024]` QKV and `[1024,1024]` output projections;
- 16 non-causal attention heads of width 64, scaled by `1/8`;
- full-head rotary embedding on Q and K, using evaluator-supplied embeddings;
- biased `[4096,1024]` and `[1024,4096]` MLP projections;
- tanh-approximate GELU between the MLP projections.

Start with the dense GEMM and elementwise templates in the installed skill.
Treat the block as one optimization unit: public workspaces define a convenient
baseline decomposition, not a required kernel boundary.
