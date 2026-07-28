<!--
Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: BSD-3-Clause
-->

# Candidate-mode dense FP8 GEMM API

This chapter is the version-pinned API reference for candidate-only eval
workspaces. It targets the CUTLASS CuTe DSL 4.6.1 API installed on the B300
worker. It is an API skeleton, not a complete kernel: the candidate still owns
tiling, pipeline flow, synchronization, accumulator state, and epilogue.

Read [candidate-kernel-patterns.md](candidate-kernel-patterns.md) for the exact
participant/token transitions. Its examples deliberately omit the
task-specific layouts, partitions, epilogue, and launch design.

The remote harness compiler remains authoritative if an error contradicts this
chapter. Fix its first diagnostic before changing another part of the design.

## Candidate boundary

Preserve the task's `@cute.jit` entrypoint signature. It is valid to change the
device-kernel signature so the JIT entrypoint can pass compile-time objects such
as the tiled MMA, TMA atoms, and SMEM layouts.

Do not add `main()`, input allocation, Torch reference code, compilation,
timing, or PASS output. The harness owns those.

## Imports and spellings

```python
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
```

Use CUTLASS scalar types:

```python
AB_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
C_DTYPE = cutlass.Float32
```

These remembered spellings are wrong for this worker:

```text
cute.constexpr
cute.Float32
cute.StructuredMmaDType
cute.ceildiv
cute.struct.SharedStorage
cute.arch.cluster_barrier
cute.arch.cluster_arrive
cute.arch.cluster_arrive_relaxed
cute.arch.cluster_wait
tcgen05.make_tiled_tma_atom_A
tcgen05.make_tiled_tma_atom_B
cpasync.make_tiled_tma_atom_A
cpasync.make_tiled_tma_atom_B
cpasync.TmaOperandMajorMode
tiled_mma.get_slice_in_stage
mma_slice.partition_D
```

Use plain trace-time Python values, `cutlass.Constexpr` where a type annotation
is required, `cutlass.range_constexpr` for fixed unrolled loops, and
`cute.ceil_div` for launch grids. A tensor shape is a tuple-like value; do not
call it as `tensor.shape(0)`.

For `tcgen05.CtaGroup.ONE`, do not invent a cluster-barrier protocol. Use the
pipeline participants below plus `pipeline.sync` for the final CTA-wide
synchronization. TMA atoms are constructed in the JIT entrypoint with
`cute.nvgpu.make_tiled_tma_atom_A/B`; never construct them inside the kernel or
from the `cpasync` module.

## JIT-side tiled MMA

Derive operand major modes from the actual tensor layouts. Do not pass a second
FP8 dtype in place of a major mode and do not invent `m=`, `n=`, or `k=`
keywords.

```python
a_major = utils.LayoutEnum.from_tensor(matrix_a).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(matrix_b_nk).mma_major_mode()

tiled_mma = sm100_utils.make_trivial_tiled_mma(
    matrix_a.element_type,
    a_major,
    b_major,
    ACC_DTYPE,
    tcgen05.CtaGroup.ONE,
    MMA_TILER_MNK[:2],
)
```

For the task-specific `(128, 256, 128)` tile, `MMA_TILER_MNK[:2]` is the
required `mma_tiler_mn` argument. Instruction K is derived by the helper; it is
not passed as a `k=` keyword.

Create staged SMEM layouts with these positional signatures:

```python
a_smem_layout = sm100_utils.make_smem_layout_a(
    tiled_mma,
    MMA_TILER_MNK,
    matrix_a.element_type,
    AB_STAGES,
)
b_smem_layout = sm100_utils.make_smem_layout_b(
    tiled_mma,
    MMA_TILER_MNK,
    matrix_b_nk.element_type,
    AB_STAGES,
)
```

Create one-stage TMA views and atoms in the JIT entrypoint:

```python
a_smem_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
b_smem_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])
tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
    tcgen05.CtaGroup.ONE
)

tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    tma_op,
    matrix_a,
    a_smem_one_stage,
    MMA_TILER_MNK,
    tiled_mma,
)
tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
    tma_op,
    matrix_b_nk,
    b_smem_one_stage,
    MMA_TILER_MNK,
    tiled_mma,
)
```

Pass the returned TMA tensor views—not the original tensors—to a kernel whose
signature accepts the constructed objects:

```python
@cute.kernel
def gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    output: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    ...
```

Bind those arguments before `.launch()`:

```python
gemm_kernel(
    tiled_mma,
    tma_atom_a,
    tma_tensor_a,
    tma_atom_b,
    tma_tensor_b,
    output,
    a_smem_layout,
    b_smem_layout,
).launch(
    grid=cute.ceil_div((*output.layout.shape, 1), MMA_TILER_MNK[:2]),
    block=(THREADS_PER_CTA, 1, 1),
)
```

Do not write `gemm_kernel().launch(...)` when the kernel declares arguments.

## Kernel coordinates and storage

Use architecture accessors:

```python
thread_idx, _, _ = cute.arch.thread_idx()
warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
block_m, block_n, _ = cute.arch.block_idx()
mma_coord = (block_m, block_n, None)
```

A compatible shared-storage declaration for A/B and accumulator pipelines is:

```python
@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32
```

Allocate it and the staged tensors with `utils.SmemAllocator`:

```python
smem = utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
smem_a = smem.allocate_tensor(
    element_type=AB_DTYPE,
    layout=a_smem_layout.outer,
    byte_alignment=128,
    swizzle=a_smem_layout.inner,
)
smem_b = smem.allocate_tensor(
    element_type=AB_DTYPE,
    layout=b_smem_layout.outer,
    byte_alignment=128,
    swizzle=b_smem_layout.inner,
)
```

`cute.empty_like`, `cute.Tensor(storage, layout)`, and `tcgen05.barrier(...)`
are not replacements for this allocation/pipeline API.

## Pipeline construction

Compute the promised TMA transaction bytes from a single stage:

```python
tma_copy_bytes = cute.size_in_bytes(
    AB_DTYPE,
    cute.select(a_smem_layout, mode=[0, 1, 2]),
) + cute.size_in_bytes(
    AB_DTYPE,
    cute.select(b_smem_layout, mode=[0, 1, 2]),
)
```

Create participants with:

```python
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=AB_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=tma_copy_bytes,
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
).make_participants()

acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
    num_stages=ACC_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(
        pipeline.Agent.Thread,
        THREADS_PER_CTA,
    ),
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
).make_participants()
```

Do not construct these as `tcgen05.PipelineTmaUmma` or replace them with an
invented manual barrier. Do not substitute `PipelineUmmaAsync` for the A/B TMA
pipeline.

## Global tiles, partitions, and fragments

For physical A `[M,K]`, B `[N,K]`, and C `[M,N]`:

```python
global_a = cute.local_tile(
    tma_tensor_a, MMA_TILER_MNK, mma_coord, proj=(1, None, 1)
)
global_b = cute.local_tile(
    tma_tensor_b, MMA_TILER_MNK, mma_coord, proj=(None, 1, 1)
)
global_c = cute.local_tile(
    output, MMA_TILER_MNK, mma_coord, proj=(1, 1, None)
)

mma_slice = tiled_mma.get_slice(0)
mma_global_a = mma_slice.partition_A(global_a)
mma_global_b = mma_slice.partition_B(global_b)
mma_global_c = mma_slice.partition_C(global_c)
mma_smem_a = tiled_mma.make_fragment_A(smem_a)
mma_smem_b = tiled_mma.make_fragment_B(smem_b)

acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
tmem_accumulator = tiled_mma.make_fragment_C(acc_shape)
```

Partition each TMA copy with:

```python
tma_smem_a, tma_global_a = cute.nvgpu.cpasync.tma_partition(
    tma_atom_a,
    0,
    cute.make_layout(1),
    cute.group_modes(smem_a, 0, 3),
    cute.group_modes(mma_global_a, 0, 3),
)
tma_smem_b, tma_global_b = cute.nvgpu.cpasync.tma_partition(
    tma_atom_b,
    0,
    cute.make_layout(1),
    cute.group_modes(smem_b, 0, 3),
    cute.group_modes(mma_global_b, 0, 3),
)
```

## TMA copy and MMA signatures

After acquiring an empty A/B stage, issue TMA copies with:

```python
empty_ab = ab_producer.acquire_and_advance()

cute.copy(
    tma_atom_a,
    tma_global_a[(None, empty_ab.count)],
    tma_smem_a[(None, empty_ab.index)],
    tma_bar_ptr=empty_ab.barrier,
)
cute.copy(
    tma_atom_b,
    tma_global_b[(None, empty_ab.count)],
    tma_smem_b[(None, empty_ab.index)],
    tma_bar_ptr=empty_ab.barrier,
)
```

TMA completion is recorded by the configured transaction bytes arriving at
`empty_ab.barrier`; there is no `ab_producer.commit()` call. Wait and release
the corresponding consumer token:

```python
full_ab = ab_consumer.wait_and_advance()
# Consume staged A/B fragments selected by full_ab.index.
full_ab.release()
```

The accumulator pipeline has a different producer transition:

```python
empty_accumulator = acc_producer.acquire_and_advance()
# Issue all UMMA contributions into this accumulator.
empty_accumulator.commit()

full_accumulator = acc_consumer.wait_and_advance()
# Read TMEM and store the task-specific epilogue.
full_accumulator.release()
```

The `commit()` and `release()` calls above belong to returned tokens, not to
`acc_producer` or `acc_consumer`. See the linked kernel patterns for the
complete nesting order and explicit invalid spellings.

`cute.gemm` is a fragment-level MMA operation. It does not create TMA atoms,
pipelines, SMEM, or the epilogue. Its five-argument order is:

```python
cute.gemm(
    tiled_mma,
    tmem_accumulator,
    a_fragment,
    b_fragment,
    tmem_accumulator,
)
```

For a staged SMEM fragment:

```python
num_k_blocks = cute.size(mma_smem_a, mode=[2])
for k_block in cutlass.range_constexpr(num_k_blocks):
    k_coord = (None, None, k_block, full_ab.index)
    cute.gemm(
        tiled_mma,
        tmem_accumulator,
        mma_smem_a[k_coord],
        mma_smem_b[k_coord],
        tmem_accumulator,
    )
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

The first contribution must use overwrite/initialize mode. The tiled MMA
starts with accumulation disabled; set `ACCUMULATE` only after issuing the
first contribution.

## TMEM and epilogue API

Allocate TMEM with a named allocation barrier:

```python
allocation_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=THREADS_PER_CTA,
)
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=allocation_barrier,
)
tmem.allocate(512)
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
tmem_accumulator = cute.make_tensor(tmem_ptr, tmem_accumulator.layout)
```

After the MMA pipeline completes, load TMEM through a matching copy atom. The
following two-subtile pattern is known to match the `(128, 256, 128)` candidate
tile on this worker:

```python
subtile_count = 2
epilogue_tiler = (
    (
        cute.size(tmem_accumulator, mode=[0, 0]),
        cute.size(tmem_accumulator, mode=[0, 1]) // subtile_count,
    ),
)
tmem_acc_epilogue = cute.zipped_divide(
    tmem_accumulator, epilogue_tiler
)
global_c_epilogue = cute.zipped_divide(
    mma_global_c, epilogue_tiler
)

tmem_atom = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    ACC_DTYPE,
)
tmem_copy = tcgen05.make_tmem_copy(
    tmem_atom, tmem_acc_epilogue[None, 0]
)
tmem_thread_copy = tmem_copy.get_slice(thread_idx)
tmem_source = tmem_thread_copy.partition_S(tmem_acc_epilogue)
global_destination = tmem_thread_copy.partition_D(global_c_epilogue)
register_accumulator = cute.make_rmem_tensor(
    global_destination[None, None, 0].shape,
    ACC_DTYPE,
)
```

`partition_S` and `partition_D` belong to the thread slice returned by
`make_tmem_copy`; they do not belong to `ThrMma`, `TiledMma`, or the tensor
returned by `partition_C`. After the accumulator consumer becomes full:

```python
for tile_idx in cutlass.range(cute.size(tmem_source, mode=[2])):
    cute.copy(
        tmem_copy,
        tmem_source[None, None, tile_idx],
        register_accumulator,
    )
    register_accumulator.store(
        register_accumulator.load() * OUTPUT_SCALE
    )
    cute.autovec_copy(
        register_accumulator,
        global_destination[None, None, tile_idx],
    )
```

Wait for the accumulator consumer before reading TMEM. Release its pipeline
stage, synchronize participating threads, and call `tmem.free(tmem_ptr)` only
after all output stores have been issued.

Keep `mma_global_a`, `mma_global_b`, and `mma_global_c` construction before any
`tma_partition` call. Do not replace the destination partition above with
`global_destination_full[None, None, 0]`, scalar `(i, j)` stores, or an
invented `get_slice_in_stage`/`partition_D` method.

## Debugging order

1. Make local `check` parse and find the required CuTe calls.
2. Run the remote harness without pipes or filters.
3. Fix only the first remote traceback.
4. Do not reintroduce a symbol already rejected by the worker.
5. Reach correctness before changing tile, stages, cluster, or epilogue.
