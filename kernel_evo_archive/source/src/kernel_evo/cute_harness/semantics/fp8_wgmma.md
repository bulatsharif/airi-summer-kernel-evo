# Genuine Hopper FP8 WGMMA

KernelEvo normally exposes BF16 tensors at the Python/module boundary for an FP8 target. FP8 is an internal storage/compute optimization, not a request to cast the whole model interface.

Exact 4.2.x atom construction:

```python
from cutlass.cute.nvgpu import warpgroup

mma_op = warpgroup.MmaF8Op(
    cutlass.Float8E4M3FN,       # A may independently be E4M3FN or E5M2
    cutlass.Float8E4M3FN,       # B may independently be E4M3FN or E5M2
    cutlass.Float32,
    (64, instruction_n, 32),
    warpgroup.OperandSource.SMEM,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(cute.make_mma_atom(mma_op), atom_layout_mnk)
```

FP8 WGMMA invariants:

- Target exactly `sm_90a`.
- A and B are 8-bit E4M3FN/E5M2; mixed E4M3/E5M2 is legal.
- The instruction K is 32. A/B are K-major in the validated Hopper GEMM.
- Use Float32 accumulation and normally BF16 output.
- The contiguous input extent is a multiple of 16 elements for a 16-byte TMA alignment guarantee, or a separate residue/conversion path is required.
- Quantization scale, saturation, NaN/Inf behavior, and error tolerances are part of correctness. Test numerical extremes, not only random values near zero.

A value round-trip through `Float8E4M3FN` in an elementwise kernel does not invoke FP8 tensor cores and is usually slower. Require `wgmma` in code-generation inspection before calling a candidate FP8-accelerated.

Conversion cost matters. Prefer prepacked immutable weights or fuse BF16-to-FP8 conversion into an already necessary data-movement stage. Include conversion, scale handling, and cache invalidation in the benchmark unless the task contract explicitly provides prequantized operands.

