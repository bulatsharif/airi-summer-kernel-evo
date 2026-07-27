# Blackwell GEMM recipes

Baseline: CUTLASS 4.6.1 commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`. Installed examples decide exact
imports/signatures. Relevant source paths commonly include:

## Contents

[Dense FP8 recipe](#dense-fp8-recipe) ·
[MXFP8 recipe](#mxfp8-recipe) ·
[Starting and adapting](#starting-and-adapting)

```text
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py
examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/
```

Preserve upstream license notices when copying substantial code.

## Dense FP8 recipe

Use when A/B are E4M3 or E5M2 and no block scale tensors enter MMA.

Host/JIT setup:

1. accept A `[M,K,L]`, physical B `[N,K,L]`, C `[M,N,L]`
2. derive major modes and require supported dtype/layout/alignment
3. construct tiled MMA with type, major modes, accumulator, one/two-CTA, and
   MMA M/N tile
4. derive instruction K, CTA tile, cluster, multicast, and grid
5. construct staged SMEM layouts
6. create A/B TMA atoms and total transaction bytes
7. allocate pipeline/shared storage and launch (baseline uses 128 threads)

Compiler-verified CUTLASS 4.6.1 seed:

```python
a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
mma_tiler_mnk = (128, 128, 64)
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a.element_type,
    a_major,
    b_major,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (128, 128),
)
smem_layout_a = sm100_utils.make_smem_layout_a(
    tiled_mma, mma_tiler_mnk, a.element_type, 2
)
smem_layout_b = sm100_utils.make_smem_layout_b(
    tiled_mma, mma_tiler_mnk, b.element_type, 2
)
```

This exact block compiled on the shared B300 and produced MMA shape
`(128,128,32)` for E4M3. Continue with the exact TMA, pipeline, `cute.gemm`,
device-index, and launch signatures in
[api-cutlass-4.6.1.md](api-cutlass-4.6.1.md). Do not replace arguments with
remembered APIs from another release.

Device flow:

```text
initialize cluster/pipelines and allocate SMEM/TMEM
partition A/B/C
for K tiles:
  TMA producer loads A/B stage
  MMA consumer waits, issues tcgen05 MMA, releases stage
wait for MMA
TMEM -> registers -> conversion/epilogue -> C
release TMEM
```

Baseline constraints:

- same A/B type; FP32 accumulation (FP16 may be supported for dense FP8)
- one-CTA MMA M: 64 or 128
- two-CTA MMA M: 128 or 256
- MMA N: 32..256 in steps of 32
- cluster M/N: positive powers of two, product at most 16
- two-CTA requires even cluster M
- contiguous A/B/C modes: at least 16-byte alignment
- no output tails without a compatible predicated/TMA-store epilogue

## MXFP8 recipe

Use only with A/B plus SFA/SFB. Baseline:

```text
A/B: E4M3 or E5M2
SFA/SFB: E8M0
scale vector: 32 K elements
accumulator: FP32
logical SFA [M,ceil(K/32),L], SFB [N,ceil(K/32),L]
```

Add to dense flow:

1. packed/swizzled scale GMEM and SMEM layouts
2. SFA/SFB TMA loads and transaction bytes
3. SMEM-to-TMEM scale copy
4. block-scaled tiled MMA consuming matching scale blocks
5. oracle decoding the exact scale layout/direction

Baseline block-scaled MMA M is 128 or 256; N is 64, 128, 192, or 256. Scale
multicast limits cluster M/N to at most 4. Do not adapt the SM103 FP4 Ultra
recipe by changing dtypes.

## Starting and adapting

Conservative dense seed: MMA `(128,128)`, cluster `(1,1)`, one CTA,
non-persistent, simple epilogue. Conservative MXFP8 seed is likewise
`(128,128)/(1,1)` with release-valid stages. Derive stages from resource
calculations, not another tile.

Adapt as one coupled design. Preserve:

- physical layouts and tiled MMA helper
- SMEM/TMEM size calculations
- TMA bytes/multicast
- warp roles and pipeline transitions
- first-K initialization versus later accumulation
- epilogue completion before TMEM release
- explicit tail restrictions

Two-CTA and persistent variants change ownership/protocol, not just a boolean.
Compile once after `can_implement`, reuse the executor, and validate in the
order in `correctness.md`.
