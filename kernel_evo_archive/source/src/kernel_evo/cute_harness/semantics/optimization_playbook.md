# Hopper low-level optimization playbook

Make one hypothesis per candidate and verify the expected instruction/resource change.

Memory movement:

- Prove 16-byte alignment before selecting 128-bit copy atoms.
- Coalesce the contiguous mode across neighboring threads; avoid expensive integer division in the inner address path.
- Use TMA for reusable 2D/3D tiles, `cp.async` for smaller staged transfers, and direct vector GMEM/RMEM copies for bandwidth-bound elementwise work.
- Select shared-memory swizzles (`MN/K` and `SW32/SW64/SW128`) that are legal for both TMA and WGMMA and check bank conflicts.
- Fuse the epilogue before a round trip to global memory.

Compute:

- BF16: `MmaF16BF16Op`, K=16, Float32 accumulator.
- FP8: `MmaF8Op`, K=32, K-major operands, Float32 accumulator.
- Choose CTA M from 64/128 and N from 64/128/256 for the validated example, then measure. Two warp groups can reduce per-thread accumulator pressure for large tiles but add synchronization.
- Use compile-time loops (`range_constexpr` or full unroll only when bounded) around instruction tiles. Avoid exploding code size.

Pipelines and scheduling:

- Tune TMA stages against latency, shared memory, and occupancy.
- Overlap producer TMA with consumer WGMMA without releasing a live stage.
- Consider cluster multicast only with real cross-CTA reuse.
- Persistent tile scheduling helps when launch/tail imbalance dominates, but requires a correct grid-stride tile scheduler and must remain a separate candidate.
- Warp specialization and `warpgroup_reg_alloc`/`warpgroup_reg_dealloc` can redistribute registers; inspect actual registers and spills.

Specialization:

- Specialize dominant aligned shapes behind a correct general path.
- Put dtype, layout, tile, cluster, stage count, and shape (when static) in the compile cache key.
- Do not force `.contiguous()` in every forward unless its cost is included and the layout conversion enables a larger measured gain.

Evidence order:

1. Compile with the exact installed API and `sm_90a`.
2. Correctness across tiny, tile-boundary, ragged, stride/alignment, and numerical-extreme cases.
3. Memcheck; then racecheck/synccheck after synchronization changes.
4. Stable warm benchmark excluding compilation but including required conversions.
5. Inspect SASS for WGMMA/TMA/vector access and local-memory spills.
6. Use a compact NCU section set to decide the next bottleneck.

