# Task reference: FP8 square GEMM

Load the `cute-fp8-kernels` skill listed in `task.json.agent_skills` first.
Before coding, read the skill's
`.opencode/skills/cute-fp8-kernels/references/candidate-dense-gemm-template.py`.
It is a complete compile-verified 4.6.1 framework example. Preserve its
TMA/pipeline/TMEM core and adapt only this task's dimensions, scale, public
kernel name, and JIT entrypoint. Do not reconstruct the core from prose.

## Data contract

- `matrix_a`: `[M, K]`, FP8 E4M3FN storage.
- `matrix_b_nk`: `[N, K]`, FP8 E4M3FN storage representing `B.T`.
- `output`: `[M, N]`, FP32.
- Here `M = N = K = 4096`.
- Input values were scaled into FP8 range by the evaluator. Restore the
  mathematical scale in the FP32 epilogue.

## Known-compatible starting design

This is a design point, not a complete implementation:

```text
MMA tile:       (128, 256, 128)
CTA group:      ONE
threads / CTA:  128
A/B stages:     4
acc stages:     1
A major:        derive from tensor (K-major for the supplied input)
B major:        derive from tensor (K-major for the supplied B_nk)
accumulator:    Float32 in TMEM
```

Use:

- `sm100_utils.make_trivial_tiled_mma`;
- `sm100_utils.make_smem_layout_a/b`;
- `cute.nvgpu.make_tiled_tma_atom_A/B`;
- `pipeline.PipelineTmaUmma` and `pipeline.PipelineUmmaAsync`;
- `tiled_mma.make_fragment_A/B/C`;
- `cute.gemm`;
- a TMEM-to-register-to-GMEM epilogue.

The launch grid is the ceiling division of output `(M, N)` by tile `(tile_m,
tile_n)`.

## Numerical note

If both inputs were multiplied by `FP8_MAX` before FP8 conversion, each FP32
accumulator result must be multiplied by:

```text
(1 / FP8_MAX) * (1 / FP8_MAX)
```

Apply that scale once in the output epilogue. Do not scale each K contribution.

If a remote diagnostic occurs, read only
`.opencode/skills/cute-fp8-kernels/references/candidate-error-atlas.md` before
making one targeted correction.
