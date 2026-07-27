<!--
Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
SPDX-License-Identifier: BSD-3-Clause
-->

# Candidate-mode CuTe kernel patterns

These are version-pinned API patterns for CUTLASS CuTe DSL 4.6.1 on the B300
worker. They deliberately are **not complete task solutions**:

- they do not define a task entrypoint;
- they do not choose a tile, stage count, layout, or warp assignment;
- they do not construct TMA atoms or tensor partitions;
- they do not define the output epilogue, scaling, tails, or validation;
- placeholder names are intentionally undefined.

Copy the API transitions you need, not the whole block. The candidate must
derive and implement every omitted task-specific part.

## Pattern 1: typed kernel binding

A CuTe kernel with arguments is bound before launch. Kernel arguments that
carry trace-time CuTe objects need CuTe type annotations.

```python
@cute.kernel
def typed_kernel_pattern(
    tiled_mma: cute.TiledMma,
    tma_atom: cute.CopyAtom,
    tma_tensor: cute.Tensor,
    output: cute.Tensor,
    smem_layout: cute.ComposedLayout,
):
    # Deliberately omitted: coordinates, partitions, computation, and stores.
    pass


@cute.jit
def launch_pattern(
    source: cute.Tensor,
    output: cute.Tensor,
):
    # Deliberately omitted: derive these objects from source/output.
    tiled_mma = BUILD_TILED_MMA(...)
    tma_atom, tma_tensor = BUILD_TMA_VIEW(...)
    smem_layout = BUILD_SMEM_LAYOUT(...)

    typed_kernel_pattern(
        tiled_mma,
        tma_atom,
        tma_tensor,
        output,
        smem_layout,
    ).launch(
        grid=DERIVE_GRID_FROM_OUTPUT_AND_TILE(...),
        block=(DERIVE_THREADS_PER_CTA(...), 1, 1),
    )
```

Do not call `typed_kernel_pattern().launch(...)` without binding its declared
arguments.

## Pattern 2: construct distinct A/B and accumulator pipelines

The A/B pipeline tracks TMA bytes arriving in staged SMEM. The accumulator
pipeline tracks completion of asynchronous UMMA work in TMEM. They are
different pipeline types with different producer completion rules.

```python
ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
    num_stages=AB_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=DERIVED_BYTES_FOR_ONE_A_AND_B_STAGE,
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

`PipelineUmmaAsync` is not a replacement for the TMA A/B pipeline. Do not
manually index its barrier storage or construct a second barrier protocol
around either participant pair.

## Pattern 3: complete A/B stage lifecycle

This is the complete participant transition sequence for one staged A/B tile.
The undefined tensors and fragments must be constructed from the candidate's
own layouts and partitions.

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

# The TMA barrier becomes full when the tx_count bytes arrive. There is no
# ab_producer.commit() call in this PipelineTmaUmma participant protocol.
full_ab = ab_consumer.wait_and_advance()

for k_block in cutlass.range_constexpr(
    cute.size(mma_smem_a, mode=[2])
):
    k_coord = (None, None, k_block, full_ab.index)
    cute.gemm(
        tiled_mma,
        tmem_accumulator,
        mma_smem_a[k_coord],
        mma_smem_b[k_coord],
        tmem_accumulator,
    )
    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)

full_ab.release()
```

The allowed transition owners are:

```text
ab_producer: acquire_and_advance() -> token with count/index/barrier
TMA copies:  signal completion by delivering the configured tx_count bytes
ab_consumer: wait_and_advance() -> full token -> token.release()
```

These spellings are wrong for this worker:

```python
ab_producer.acquire()
ab_producer.commit()
ab_producer.release(...)
ab_consumer.acquire()
ab_consumer.commit()
```

## Pattern 4: complete accumulator lifecycle

Acquire one empty accumulator stage before issuing its UMMA contributions.
Unlike the TMA producer token, the accumulator producer token is explicitly
committed after the final contribution.

```python
empty_accumulator = acc_producer.acquire_and_advance()

# Deliberately omitted: one or more A/B stage lifecycles that issue cute.gemm
# contributions into this accumulator.
ISSUE_ALL_MMA_CONTRIBUTIONS(...)

empty_accumulator.commit()

full_accumulator = acc_consumer.wait_and_advance()

# Deliberately omitted: derive a TMEM copy atom, partition source/destination,
# load registers, apply the task's epilogue, and store the result.
STORE_TASK_SPECIFIC_EPILOGUE(...)

full_accumulator.release()
```

The allowed transition owners are:

```text
acc_producer: acquire_and_advance() -> empty token -> token.commit()
acc_consumer: wait_and_advance() -> full token -> token.release()
```

Do not call `acc_producer.commit()` or `acc_consumer.release()` directly; those
transitions belong to the tokens returned by the participant methods.

## Pattern 5: nesting the two lifecycles

This control-flow-only pattern shows where the two protocols nest. It still
omits all task-defining work.

```python
empty_accumulator = acc_producer.acquire_and_advance()

for _ in cutlass.range(NUM_K_TILES, prefetch_stages=AB_STAGES - 2):
    empty_ab = ab_producer.acquire_and_advance()
    ISSUE_TMA_A_AND_B_WITH_BARRIER(empty_ab)

    full_ab = ab_consumer.wait_and_advance()
    ISSUE_MMA_FROM_STAGE(full_ab.index)
    full_ab.release()

empty_accumulator.commit()

full_accumulator = acc_consumer.wait_and_advance()
STORE_TASK_SPECIFIC_EPILOGUE(...)
full_accumulator.release()
```

To turn this into a candidate, the agent must still solve tensor layouts,
TMA partitioning, TMEM allocation, accumulation initialization, synchronization,
output partitioning, numerical scaling, boundary behavior, and launch geometry.
