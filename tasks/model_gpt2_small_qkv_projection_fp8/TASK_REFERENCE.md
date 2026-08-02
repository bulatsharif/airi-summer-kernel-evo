# Task reference: GPT-2 Small combined QKV projection

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-dense-gemm-template.py`. Preserve its compile-verified
TMA, pipeline, TMEM, and output-copy flow until the GEMM executes remotely.

For a correctness-first implementation, also use
`references/candidate-elementwise-template.py` and replace its neutral
expression with FP32 bias addition.

## Model mapping

GPT-2 Small uses hidden size 768, 12 attention heads, and head dimension 64.
Its `c_attn` layer produces all three projections with one width-2304 linear
operation:

```text
Q = Y[:,    0: 768]
K = Y[:,  768:1536]
V = Y[:, 1536:2304]
```

GPT-2's `Conv1D` representation convention stores the mathematical weight as
`[K,N]`. This task uses the equivalent inference-packed `[N,K]` representation,
so the supplied `W_qkv_nk` is consumed directly by the CuTe GEMM.

## Known-compatible starting point

```text
M/N/K:           8192 / 2304 / 768
MMA tile:        (128, 256, 128)
threads / CTA:   128
A/B stages:      3
acc stages:      1
accumulator:     Float32 in TMEM
```

The dense template restores `SCALE_X * SCALE_W` in its TMEM-to-GMEM
epilogue. A separate two-dimensional kernel can then add `bias_qkv[column]`.
Once this version passes, bias fusion and tile/stage changes are valid
optimization directions.

After a concrete remote failure, route the first diagnostic through
`references/candidate-error-atlas.md`; do not reconstruct the GEMM from other
API families.
