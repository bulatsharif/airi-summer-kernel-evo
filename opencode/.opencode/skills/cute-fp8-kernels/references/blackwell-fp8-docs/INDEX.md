# Blackwell FP8 documentation pack

This is the offline entry point for dense FP8 CuTe DSL work on the shared B300.
It compresses the relevant NVIDIA CUTLASS 4.6.1 documentation into a connected
workflow and adds a small set of facts verified against the actual evaluator.

The pack is documentation, not a solution. It does not provide the complete
Level 2 kernel, its launch grid, the BiasAdd + ReLU implementation, or an
optimized schedule.

## Authority and precedence

Use information in this order:

1. `TASK.md` and `task.json` define the numerical and submission contract.
2. `server-api-deltas.md` defines APIs verified on the shared B300 build.
3. The remaining pages summarize NVIDIA CUTLASS 4.6.1 documentation.
4. Compiler/runtime feedback from `python -m cute_harness run` is the final
   authority for an attempted candidate.

The server delta page intentionally overrides upstream documentation when the
installed package behaves differently.

## Reading route

For dense GEMM + epilogue tasks, read in this order:

1. [`exact-api-recipes.md`](exact-api-recipes.md) — copy the verified namespace,
   method, and signature forms instead of guessing CUDA-like conveniences.
2. [`examples/fp8-mma-one-tile.py`](examples/fp8-mma-one-tile.py) — study one
   complete, numerically verified FP8 GMEM-to-GMEM bridge before editing.
3. [`task-adaptation.md`](task-adaptation.md) — map the public task onto tiles,
   scaling, and the two-kernel correctness-first decomposition.
4. [`architecture-and-dataflow.md`](architecture-and-dataflow.md) — understand
   which CuTe objects live in GMEM, SMEM, TMEM, and RMEM.
5. [`tma-and-pipelines.md`](tma-and-pipelines.md) — construct TMA views and the
   producer/consumer barriers without inventing storage objects.
6. [`tmem-and-epilogue.md`](tmem-and-epilogue.md) — bind the accumulator to
   TMEM, issue `cute.gemm`, and store results back through registers.
7. [`server-api-deltas.md`](server-api-deltas.md) — use the exact installed
   spellings and reject known incompatible alternatives.

Read [`SOURCES.md`](SOURCES.md) only when provenance matters. The
[`examples/`](examples/) directory is reserved for small, independently
validated code examples; its README describes the acceptance bar.

## Non-negotiable dense tcgen05 flow

```text
GMEM A/B
  -> local_tile
  -> ThrMma.partition_A/B
  -> cpasync.tma_partition
  -> TMA copy into physically allocated SMEM A/B
  -> TiledMma.make_fragment_A/B
  -> cute.gemm into a TMEM-backed FP32 accumulator
  -> tcgen05 TMEM load into per-thread RMEM
  -> transform/scale if required
  -> store into partitioned GMEM output
```

If a candidate skips an address-space transition, creates a tensor from a
layout and dtype, treats a layout as storage, or attempts to store through the
`partition_C` method object, it is not following the Blackwell dense path.

## Iteration discipline

Before every remote attempt:

```text
python -m cute_harness check <task-id> submission.py
```

Then use the bounded evaluator command supplied by the experiment. Fix the
first concrete compiler/runtime error; do not rewrite working parts of the
dataflow from memory. If an attribute is missing, return to
`exact-api-recipes.md`; never invent a replacement method name.
