# BF16 Hopper WGMMA

BF16 is the default storage and module ABI for this harness. Use Float32 accumulation.

The exact 4.2.x atom construction is:

```python
from cutlass.cute.nvgpu import warpgroup

mma_op = warpgroup.MmaF16BF16Op(
    cutlass.BFloat16,
    cutlass.Float32,
    (64, instruction_n, 16),
    warpgroup.OperandSource.SMEM,
    warpgroup.OperandMajorMode.K,
    warpgroup.OperandMajorMode.K,
)
tiled_mma = cute.make_tiled_mma(
    cute.make_mma_atom(mma_op),
    atom_layout_mnk,
)
```

`instruction_n` must be a multiple of 8 in `[8, 256]`; Hopper BF16 WGMMA uses K=16. The full CTA K tile can be a multiple of this instruction K. `cutlass.utils.hopper_helpers.make_trivial_tiled_mma` is a version-matched helper that selects the BF16 or FP8 operation from the operand dtype.

The mainloop protocol is:

1. TMA copies GMEM A/B into a staged, WGMMA-compatible shared-memory layout.
2. The consumer waits for the stage's transaction barrier.
3. Call `cute.nvgpu.warpgroup.fence()` before issuing the group.
4. Issue `cute.gemm(...)`, then `commit_group()` and a bounded `wait_group(...)`.
5. Release a stage only after all WGMMA reads from it are complete.
6. Convert/fuse the epilogue in registers, write a swizzled shared tile, then TMA-store to BF16 output.

Expected code generation includes WGMMA and, for the high-performance mainloop, TMA. A kernel that compiles to scalar FFMA or scalar global loads is not the intended fast path.

