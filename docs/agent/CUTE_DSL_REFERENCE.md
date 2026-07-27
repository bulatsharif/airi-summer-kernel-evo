# CuTe DSL reference for eval agents

This is public task context, not a known solution or evaluator source. It
summarizes the supported CUTLASS CuTe DSL patterns available on the remote
Blackwell worker. Read the task-specific reference alongside this file.

Authoritative upstream references:

- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/mma_docs/tcgen05_programming.html
- https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute_nvgpu_tcgen05.html
- https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL/blackwell/tutorial_gemm

The remote compile/run result is authoritative if an installed API differs
from a remembered CUTLASS version.

## Candidate boundary

The candidate owns imports, constants, `@cute.kernel` functions and the
`@cute.jit` entrypoint named by the starter. The harness owns:

- PyTorch input generation and reference computation;
- FP8 storage creation;
- `cute.compile`;
- warmup, CUDA-event timing and numerical validation;
- `main()` and the final PASS line.

Do not add `main()`, input generation, a reference implementation or a fake
PASS print. Preserve the function signatures from `submission.py`.

## Fast iteration

Always start with the local policy check because it does not consume GPU time:

```text
python -m cute_harness check TASK_ID submission.py
```

Then submit only after the candidate parses and contains the required CuTe
calls:

```text
python -m cute_harness run TASK_ID submission.py
```

Use the returned compiler traceback as the API oracle. Fix the first concrete
error rather than searching unrelated repository trees. A successful remote
run already checks correctness and returns a kernel profile.

## CuTe DSL execution model

- `@cute.kernel` defines device code.
- `@cute.jit` defines the host-side JIT entrypoint and launches kernels.
- Runtime tensors use `cute.Tensor`; element types are CUTLASS types such as
  `cutlass.Float8E4M3FN` and `cutlass.Float32`.
- Values known while tracing can drive Python control flow. Prefer
  `cutlass.range_constexpr` for a small compile-time loop and `cutlass.range`
  for DSL loops.
- Obtain coordinates with `cute.arch.thread_idx()`,
  `cute.arch.block_idx()` and `cute.arch.warp_idx()`.
- Launch with `kernel(...).launch(grid=(...), block=(...))`.
- The harness compiles the JIT entrypoint, so the candidate should not call
  `cute.compile` itself.

## Tensor tiling

The common mapping for a GEMM tile is:

```python
global_a = cute.local_tile(
    matrix_a, tile_mnk, (block_m, block_n, None),
    proj=(1, None, 1),
)
global_b = cute.local_tile(
    matrix_b_nk, tile_mnk, (block_m, block_n, None),
    proj=(None, 1, 1),
)
global_c = cute.local_tile(
    output, tile_mnk, (block_m, block_n, None),
    proj=(1, 1, None),
)
```

For this dataset the right operand is already stored as `[N, K]`. Do not add a
runtime transpose.

## Blackwell tcgen05 GEMM checklist

Blackwell `tcgen05` consumes A/B descriptors from SMEM and keeps the FP32
accumulator in TMEM. A practical single-CTA design has these phases:

1. In the JIT entrypoint, derive major modes from the input tensors.
2. Construct a `cute.TiledMma`.
3. Construct staged SMEM layouts for A and B.
4. Construct tiled TMA G2S atoms.
5. Launch a kernel over output tiles.
6. In the kernel, allocate SMEM, barriers and TMEM.
7. Partition global and SMEM tensors for the MMA.
8. TMA-copy K tiles into staged SMEM buffers.
9. Issue `cute.gemm` for every K block.
10. Copy the completed TMEM accumulator through registers to GMEM.

The helper path for FP8 avoids hard-coding a version-specific MMA op class:

```python
a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(b_nk).mma_major_mode()
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a.element_type,
    a_major,
    b_major,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    tile_mnk[:2],
)
```

Build SMEM and TMA metadata in the JIT entrypoint:

```python
a_layout = sm100_utils.make_smem_layout_a(
    tiled_mma, tile_mnk, a.element_type, stages
)
b_layout = sm100_utils.make_smem_layout_b(
    tiled_mma, tile_mnk, b_nk.element_type, stages
)
tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
    tcgen05.CtaGroup.ONE
)
```

Use `cute.nvgpu.make_tiled_tma_atom_A/B` with the one-stage SMEM view,
the MMA tile and `tiled_mma`.

Inside the kernel:

- allocate staged tensors with `utils.SmemAllocator().allocate_tensor`;
- use `pipeline.PipelineTmaUmma` for TMA producer/MMA consumer coordination;
- use `pipeline.PipelineUmmaAsync` to sequence MMA completion and epilogue;
- create fragments with `tiled_mma.make_fragment_A/B/C`;
- set `tcgen05.Field.ACCUMULATE` to false for the first K contribution and
  true for subsequent contributions;
- call `cute.gemm(tiled_mma, accumulator, a_fragment, b_fragment,
  accumulator)`.

`cute.gemm` is asynchronous. Respect the pipeline completion barrier before
reading TMEM. TMEM is explicitly allocated and freed.

For the epilogue, use a `tcgen05.Ld32x32bOp` copy atom,
`tcgen05.make_tmem_copy`, a per-thread register tensor, and
`cute.autovec_copy` into the corresponding output partition.

## Non-GEMM kernels

For an elementwise or reduction kernel:

- use one CTA per independent row or a 1D grid over elements;
- convert FP8 loads to `cutlass.Float32` before arithmetic;
- use `cute.arch.shuffle_sync_bfly` for a warp reduction;
- exchange one value per warp through SMEM for a CTA reduction;
- use `cute.arch.sync_threads()` around SMEM handoffs;
- use `cute.rsqrt` for reciprocal square root;
- write the requested output dtype directly.

## Common failures

- A candidate `main()` conflicts with the harness-owned evaluator.
- Calling PyTorch from candidate code fails policy.
- Using `B` as `[K, N]` is wrong here; the ABI supplies `[N, K]`.
- Reading TMEM before MMA completion produces races or invalid output.
- Forgetting the first-iteration accumulate mode reads an uninitialized
  accumulator.
- Omitting FP8 dequantization scale gives numerically wrong FP32 output.
- A grid computed from input shapes can swap M/N; derive it from output shape.
- Repeating an identical failing tool call is not progress. Read the reported
  error, edit once, then rerun the check.
