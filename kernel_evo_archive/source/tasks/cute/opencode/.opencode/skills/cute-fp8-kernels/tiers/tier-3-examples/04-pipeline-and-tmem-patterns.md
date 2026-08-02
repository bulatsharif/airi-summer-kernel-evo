# Pipeline and tensor-memory patterns

These fragments demonstrate object lifecycles. They omit the operation,
resource values, role predicates, iteration space, and tensor layouts.

## Cooperative groups

```python
producer_group = pipeline.CooperativeGroup(
    pipeline.Agent.Thread,
    PRODUCER_PARTICIPANTS,
)
consumer_group = pipeline.CooperativeGroup(
    pipeline.Agent.Thread,
    CONSUMER_PARTICIPANTS,
)
```

The counts must equal the actual participating threads. A group with the wrong
agent enum or count can construct an invalid pipeline.

## TMA/UMMA pipeline factory

```python
created_pipeline = pipeline.PipelineTmaUmma.create(
    num_stages=NUMBER_OF_STAGES,
    producer_group=producer_group,
    consumer_group=consumer_group,
    tx_count=TRANSACTION_BYTES_PER_STAGE,
    barrier_storage=BARRIER_STORAGE_POINTER,
    cta_layout_vmnk=CTA_LAYOUT_VMNK,
    mcast_mode_mn=MULTICAST_MODES,
)

producer, consumer = created_pipeline.make_participants()
```

Every uppercase value belongs to the resource and ownership model.

## TMA participant lifecycle

```python
producer_token = producer.acquire_and_advance()

ISSUE_TMA_WORK(
    stage_barrier=producer_token.barrier,
    work_coordinate=work_coordinate,
)

consumer_token = consumer.wait_and_advance()
CONSUME_COMPLETED_STAGE(consumer_token)
consumer_token.release()
```

TMA byte arrival publishes producer completion. This fragment intentionally
does not call a generic producer commit.

## UMMA accumulator pipeline

```python
created_accumulator_pipeline = pipeline.PipelineUmmaAsync.create(
    num_stages=ACCUMULATOR_STAGES,
    producer_group=mma_group,
    consumer_group=epilogue_group,
    barrier_storage=ACCUMULATOR_BARRIER_STORAGE,
    cta_layout_vmnk=CTA_LAYOUT_VMNK,
)

mma_producer, epilogue_consumer = (
    created_accumulator_pipeline.make_participants()
)
```

The accumulator stage count and groups are distinct from the operand pipeline.

## UMMA participant lifecycle

```python
empty_accumulator = mma_producer.acquire_and_advance()
ISSUE_ALL_MMA_CONTRIBUTIONS(empty_accumulator)
empty_accumulator.commit()

full_accumulator = epilogue_consumer.wait_and_advance()
READ_AND_STORE_ACCUMULATOR(full_accumulator)
full_accumulator.release()
```

Commit belongs to the UMMA producer token in this interface. The uppercase
operations hide task-specific MMA and output work.

## Pipeline loop skeleton

```python
for work_index in cutlass.range(runtime_work_count):
    producer_token = producer.acquire_and_advance()
    ISSUE_ASYNCHRONOUS_STAGE(work_index, producer_token)

    consumer_token = consumer.wait_and_advance()
    CONSUME_STAGE(work_index, consumer_token)
    consumer_token.release()
```

This does not show a correct prologue or drain for any particular ring. Derive
those regions from work count and pipeline semantics.

## Named synchronization

```python
role_barrier = pipeline.NamedBarrier(
    barrier_id=BARRIER_ID,
    num_threads=PARTICIPATING_THREADS,
)

pipeline.sync(barrier_id=BARRIER_ID)
```

The exact call surface depends on whether the named barrier object or namespace
helper is used. Do not mix IDs or participant counts with another protocol.

## TMEM allocation

```python
tmem_allocation_barrier = pipeline.NamedBarrier(
    barrier_id=TMEM_BARRIER_ID,
    num_threads=TMEM_BARRIER_PARTICIPANTS,
)
tmem_allocator = utils.TmemAllocator(
    TMEM_BASE_ADDRESS_STORAGE,
    barrier_for_retrieve=tmem_allocation_barrier,
    allocator_warp_id=ALLOCATOR_WARP_ID,
    is_two_cta=USES_TWO_CTAS,
)
tmem_allocator.allocate(TMEM_COLUMNS)
tmem_allocator.wait_for_alloc()
tmem_pointer = tmem_allocator.retrieve_ptr(dtype=ACCUMULATOR_TYPE)
```

`TMEM_COLUMNS` follows from the accumulator and epilogue fragment layouts. The
fragment leaves every resource value undefined and does not show release.

## Binding a fragment layout to TMEM

```python
accumulator_metadata = tiled_mma.make_fragment_C(
    tiled_mma.partition_shape_C(OUTPUT_TILE_SHAPE)
)
accumulator = cute.make_tensor(
    tmem_pointer,
    accumulator_metadata.layout,
)
```

The metadata layout must be compatible with the allocated TMEM region. This
fragment does not establish the correct output tile shape.

## TMEM-to-register copy

```python
tmem_copy = tcgen05.make_tmem_copy(
    TMEM_LOAD_OPERATION,
    accumulator,
)
participant_copy = tmem_copy.get_slice(participant_index)

tmem_partition = participant_copy.partition_S(accumulator)
register_fragment = cute.make_rmem_tensor_like(
    participant_copy.partition_D(DESTINATION_VIEW),
    dtype=COMPUTE_TYPE,
)
register_partition = participant_copy.partition_D(register_fragment)

cute.copy(tmem_copy, tmem_partition, register_partition)
```

The load operation, destination view, and participant mapping must be selected
together.

## Lifetime audit

```text
barrier storage initialized by:
operand stage acquired by:
TMA issued by:
stage consumed by:
stage released by:
TMEM allocated by:
MMA committed by:
TMEM read after:
TMEM released after:
```

No line can be filled from syntax alone.
