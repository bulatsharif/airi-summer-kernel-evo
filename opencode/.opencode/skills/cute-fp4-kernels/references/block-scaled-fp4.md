# Native block-scaled FP4 on Blackwell

Use this page only when FP4 values and scale tensors are inputs to a native
Blackwell MMA. A packed scalar decode kernel is not a native FP4 GEMM.

## Contracts that must remain distinct

| Contract | Values | Scale convention |
|---|---|---|
| Packed scalar E2M1 | Two nibbles per byte | Defined entirely by the task |
| MXFP4 | E2M1 blocks | Commonly one E8M0 power-of-two scale per 32 values |
| NVFP4 | E2M1 blocks | Commonly one E4M3 scale per 16 values, with any outer scale defined by the task |

These are mathematical conventions, not permission to guess a physical
layout. The task and installed CUTLASS release determine scale direction,
padding, swizzle, alignment, and operand layout.

## Required task facts

Before adapting a recipe, identify:

- physical A and B storage and logical matrix orientation;
- K elements represented by one scale;
- logical and physical SFA/SFB shapes;
- scale dtype and scale-versus-inverse-scale equation;
- accumulator and output dtype;
- supported M/N/K tiles, alignment, and K-tail behavior;
- one-CTA versus two-CTA ownership;
- whether native FP4 tensor-core evidence is required.

If any item changes the mathematical result, do not silently default it.

## Blackwell dataflow

A native block-scaled implementation normally preserves this coupled flow:

```text
packed FP4 A/B in GMEM
  -> MMA-compatible A/B SMEM layouts
scale A/B in GMEM
  -> scale SMEM layouts
  -> MMA-compatible scale storage, commonly TMEM
block-scaled tcgen05 MMA
  -> FP32 TMEM accumulator
  -> RMEM epilogue
  -> output in GMEM
```

Scale copies participate in transaction-byte accounting and pipeline state.
Ordinary row-major scale tensors cannot be substituted for instruction-specific
packed/swizzled layouts.

## API discipline

The repository verifies generic Blackwell APIs such as:

```text
cute.make_tensor(pointer, layout)
cute.copy(atom, src, dst, **kwargs)
cute.gemm(atom, d, a, b, c)
pipeline.PipelineTmaUmma
pipeline.PipelineUmmaAsync
tcgen05 TMEM allocation/load operations
```

It does not yet contain a remotely verified FP4 tiled-MMA constructor or scale
layout recipe. Therefore:

- start from the installed release's Blackwell FP4 example selected by the
  task;
- copy its operand type, tiled-MMA constructor, scale-layout utilities, stage
  calculations, and scale-copy protocol as one unit;
- do not adapt the MXFP8 recipe by changing only dtypes;
- do not invent `Float4...`, `Mma...`, or scale-layout symbol names;
- treat the first B300 compiler diagnostic as authoritative.

Add exact signatures to this pack only after a neutral probe reaches at least
COMPILE evidence. Label numerical recipes only after an independent oracle
passes.

## Correctness isolation

Validate in this order:

1. one output tile and one K scale block;
2. multiple K scale blocks with distinct scales;
3. multiple output tiles;
4. the exact epilogue and output conversion;
5. staging and cluster variants;
6. tails, only if required.

Use one-hot K inputs to isolate block selection. Use alternating large/small
adjacent scales to expose swapped SFA/SFB modes or an off-by-one scale block.
Compare against dequantized values derived from the actual packed operands and
actual scale tensors.

## Performance evidence

Passing numerics does not prove native FP4 MMA. Before claiming acceleration,
retain codegen/profile evidence for the intended tensor-core instruction
family, exclude compilation and allocation from timing, and compare repeated
measurements against the task baseline.
