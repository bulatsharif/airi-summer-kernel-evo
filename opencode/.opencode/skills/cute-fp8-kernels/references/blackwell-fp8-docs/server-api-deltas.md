# Shared B300 API deltas and verified contracts

This page overrides upstream documentation for the evaluator's installed
CUTLASS DSL build. Evidence was collected with neutral probes on 2026-07-27.

Evidence levels:

- `SIGNATURE`: runtime introspection only.
- `COMPILE`: traced successfully to MLIR.
- `LAUNCH`: compiled and executed on B300.
- `NUMERICAL`: output compared against an independent reference.

## Imports

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
```

## Dense FP8 TiledMma constructor — COMPILE

Use exactly six positional arguments:

```python
a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a.element_type,
    a_major,
    b_major,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (128, 128),
)
```

For E4M3 inputs the observed hardware MMA shape was `(128,128,32)`. The
upstream guide shows a different generic parameter list; do not use it on this
server. Keyword forms such as `a_major=`, `b_dtype=`, `cta_group=`, or
`atom_mn=` were rejected or mis-bound.

## SMEM layouts and TMA factories — COMPILE/LAUNCH

```text
make_smem_layout_a(tiled_mma, mma_tiler_mnk, dtype, stages)
make_smem_layout_b(tiled_mma, mma_tiler_mnk, dtype, stages)
cute.nvgpu.make_tiled_tma_atom_A(...) -> TmaInfo
cute.nvgpu.make_tiled_tma_atom_B(...) -> TmaInfo
```

`TmaInfo` has `.atom`, `.tma_tensor`, and `.smem_layout`. It is not a tuple.

An FP8 TMA copy using `cpasync.tma_partition` launched successfully. A manual
equal-shape composition compiled but raised a CUDA illegal instruction.

## One-tile dense bridge — LAUNCH + NUMERICAL

A neutral `A[128,64] @ B[128,64].T -> C[128,128]` kernel completed through:

```text
TMA GMEM->SMEM
SMEM descriptors -> cute.gemm
FP32 TMEM accumulator
TMEM->RMEM->FP32 GMEM epilogue
```

With integer inputs representable exactly in E4M3FN:

```text
max_abs_error=0.0
device_time_ms=0.13721599999791942
profile_id=1bf96cbb-9f27-4888-9539-95ff03883227
```

This validates the protocol and address spaces, not the full Level 2 schedule.

## Exact pipeline group syntax — source + compiler feedback

```python
pipeline.CooperativeGroup(pipeline.Agent.Thread)
pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
```

The first argument is the `pipeline.Agent` enum. Integer substitutes caused
repeated `number of threads ... must be more than 0` errors because they are
not recognized agent values.

## Launch syntax

Bind every declared kernel argument, then provide an explicit grid and block:

```python
kernel(arg0, arg1).launch(
    grid=(grid_x, grid_y, 1),
    block=(128, 1, 1),
)
```

Do not use `kernel[grid](...)`, `kernel.launch(...)`, or a bound `.launch()`
with no `grid`/`block`.

## Core installed signatures

```text
cute.make_layout(shape, *, stride=None)
cute.make_tensor(pointer, layout)
cute.make_rmem_tensor(layout_or_shape, dtype)
cute.gemm(atom, d, a, b, c)
cute.copy(atom, src, dst, **kwargs)
cute.size(tensor, mode=None)
cute.size_in_bytes(dtype, layout)
```

`cute.full(shape, fill_value, dtype)` returns TensorSSA, not a `cute.Tensor`
that can be bound to a tensor-typed kernel argument.

## Known unavailable or incompatible spellings

Do not use:

- `cute.LayoutEnum`; use `utils.LayoutEnum`.
- `cute.pointer` or `cute.raw_pointer_as_ptr`.
- `cute.make_stride`; pass a plain tuple as `stride=`.
- `cute.make_smem_tensor` or `TiledMma.make_smem_A`.
- `cute.partition_S/D` as top-level functions.
- `cute.pipeline`, `cute.PipelineTmaUmma`, or
  `cute.arch.PipelineTmaUmma`.
- `sm100_utils.make_tiled_tma_atom_A/B`; factories live in `cute.nvgpu`.
- `sm100_utils.make_trivial_pipeline`, `stage_input_A/B`, `wait_pipeline`, or
  `commit_pipeline`.
- `cutlass.float32`, `cute.float32`, `cute.Shape(...)`, `cute._range`,
  `.numel()`, or `layout.cosize()`.
- early `return` inside `@cute.kernel`.

## Raw TMA barrier ordering — LAUNCH

For the neutral non-multicast copy:

```text
elect one: mbarrier_init + mbarrier_expect_tx
mbarrier_init_fence
CTA barrier
TMA cute.copy outside elect_one
elect one: mbarrier_arrive
mbarrier_wait
```

The TMA copy already elects its issuing thread. Double election can deadlock.
