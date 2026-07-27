# TMA staging, shared storage, and pipelines

This page combines the relevant Blackwell TMA tutorial and CUTLASS pipeline
helper contracts. All SMEM allocation happens inside `@cute.kernel`.

## Shared storage is a module-level CuTe struct

For a two-stage A/B pipeline and a one-stage accumulator pipeline:

```python
AB_STAGES = 2
ACC_STAGES = 1

@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32
```

Inside the kernel:

```python
smem = utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
```

Do not define a Python storage class inside `@cute.jit`, allocate shared memory
from host/JIT code, or pass a Python storage instance as a kernel argument.

Pointer access depends on the field:

```text
storage.ab_mbar_ptr.data_ptr()       pipeline barrier array pointer
storage.acc_mbar_ptr.data_ptr()      accumulator barrier array pointer
storage.tmem_holding_buf.ptr         one Int32 holding location for TMEM address
```

An allocated tensor does not expose `.ptr` merely because it is backed by
memory.

## Allocate staged A/B tensors

The Blackwell helpers may return a composed layout. Split its outer layout and
swizzle during allocation:

```python
sA = smem.allocate_tensor(
    element_type=fp8_dtype,
    layout=smem_layout_a.outer,
    byte_alignment=128,
    swizzle=smem_layout_a.inner,
)
sB = smem.allocate_tensor(
    element_type=fp8_dtype,
    layout=smem_layout_b.outer,
    byte_alignment=128,
    swizzle=smem_layout_b.inner,
)
```

## TMA descriptor objects

The installed A/B factories return one `TmaInfo` each. Construct them in the
`@cute.jit` function from the evaluator-owned GMEM tensors and a one-stage SMEM
layout:

```python
tma_a = cute.nvgpu.make_tiled_tma_atom_A(
    sm100_utils.CopyBulkTensorTileG2SOp(),
    a,
    cute.select(smem_layout_a, mode=[0, 1, 2]),
    mma_tiler_mnk,
    tiled_mma,
)
```

Use `tma_a.atom` and `tma_a.tma_tensor`; do not tuple-unpack the result.

## TMA partitioning

First partition the local GMEM tile with `ThrMma.partition_A/B`. Then transform
both the staged SMEM tensor and that GMEM partition into the TMA atom's internal
layout:

```python
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a,
    0,
    cute.make_layout(1),
    cute.group_modes(sA, 0, 3),
    cute.group_modes(tCgA, 0, 3),
)
```

Repeat symmetrically for B. A manual equal-shape composition is not an
equivalent replacement for `tma_partition`.

## CooperativeGroup semantics

The constructor is:

```text
pipeline.CooperativeGroup(agent, size=1, alignment=None)
```

The first argument is an enum, never an integer thread count:

```python
one_thread = pipeline.CooperativeGroup(pipeline.Agent.Thread)
all_threads = pipeline.CooperativeGroup(pipeline.Agent.Thread, 128)
```

Do not write `CooperativeGroup(128)`, `CooperativeGroup(1, 128)`, or
`CooperativeGroup(agent=128, size=1)`.

## Pipeline creation

Compute transaction bytes from one A stage plus one B stage:

```python
tx_bytes = (
    cute.size_in_bytes(fp8_dtype, cute.select(smem_layout_a, mode=[0, 1, 2]))
    + cute.size_in_bytes(fp8_dtype, cute.select(smem_layout_b, mode=[0, 1, 2]))
)
```

Then create both participant pairs:

```python
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=AB_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=tx_bytes,
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
).make_participants()

acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
    num_stages=ACC_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
    barrier_storage=storage.acc_mbar_ptr.data_ptr(),
).make_participants()
```

`barrier_storage=None` is not valid on the shared B300 build.

## Producer/consumer lifecycle

The A/B producer obtains an empty stage, issues both TMA copies against its
barrier, and the consumer waits for the full stage:

```text
ab_empty = ab_producer.acquire_and_advance()
copy A into stage ab_empty.index with barrier ab_empty.barrier
copy B into stage ab_empty.index with the same barrier
ab_full = ab_consumer.wait_and_advance()
issue every MMA K block reading stage ab_full.index
ab_full.release()
```

The accumulator pipeline wraps the full K loop:

```text
acc_empty = acc_producer.acquire_and_advance()
issue all K tiles and K blocks
acc_empty.commit()

acc_full = acc_consumer.wait_and_advance()
read TMEM in the epilogue
acc_full.release()
```

## TMA election rule

Non-multicast `CopyBulkTensorTileG2SOp` already performs its own issue
election. Do not wrap the TMA `cute.copy(..., tma_bar_ptr=...)` in
`with cute.arch.elect_one()`.

For a raw one-shot barrier (outside the pipeline helper), initialize and set
expected bytes under `elect_one`, synchronize the CTA, issue the TMA copy
outside the election scope, then arrive/wait according to the verified barrier
protocol in `server-api-deltas.md`.
