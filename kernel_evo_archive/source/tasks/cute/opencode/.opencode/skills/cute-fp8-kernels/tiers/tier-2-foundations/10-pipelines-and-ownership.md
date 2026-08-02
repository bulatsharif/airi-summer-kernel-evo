# Asynchronous pipelines and ownership

A pipeline is a ring of storage stages plus synchronization state. The stage
index identifies a storage slot; the phase distinguishes successive uses of
that slot.

## Pipeline vocabulary

Conceptual producer:

```text
wait until a stage is reusable
issue asynchronous work associated with the stage
publish or commit completion responsibility
advance to the next stage
```

Conceptual consumer:

```text
wait until a stage is complete
consume its data
release the stage for reuse
advance to the next stage
```

These are roles, not universal method names. Different pipeline classes model
different completion engines.

## Pipeline namespace

Pipeline classes live under:

```python
import cutlass.pipeline as pipeline
```

Core participant types:

```text
pipeline.Agent.Thread
pipeline.Agent.Warp
pipeline.Agent.ThreadBlock
pipeline.Agent.ThreadBlockCluster
pipeline.CooperativeGroup(agent, size=1)
```

`CooperativeGroup` receives an `Agent` enum first and an optional participant
count. An integer alone is not an agent kind.

## TMA-to-UMMA pipeline

```text
pipeline.PipelineTmaUmma.create(
    *,
    num_stages,
    producer_group,
    consumer_group,
    tx_count,
    barrier_storage=None,
    cta_layout_vmnk=None,
    mcast_mode_mn=(1, 1),
    enable_multicast_signaling=False,
    defer_sync=False,
    name="",
) -> pipeline
```

| Parameter | Meaning |
| --- | --- |
| `num_stages` | ring length |
| `producer_group` | participants issuing TMA work |
| `consumer_group` | participants consuming staged operands |
| `tx_count` | expected asynchronous bytes per stage |
| `barrier_storage` | optional shared-memory backing; omitted storage uses the reserved allocator |
| `cta_layout_vmnk` | optional cluster participant layout |
| `mcast_mode_mn` | multicast modes for operand sharing |
| `enable_multicast_signaling` | enable consumer signaling across multicast recipients |
| `defer_sync` | postpone factory-owned initial synchronization |
| `name` | optional profiling/debug label |

This class couples TMA byte arrival to readiness for UMMA consumption. A
producer does not add an unrelated commit if transaction completion already
publishes the stage.

## UMMA asynchronous pipeline

```text
pipeline.PipelineUmmaAsync.create(
    *,
    num_stages,
    producer_group,
    consumer_group,
    barrier_storage=None,
    cta_layout_vmnk=None,
    defer_sync=False,
    name="",
) -> pipeline
```

This class coordinates asynchronous tensor-core output with its consumer.
Completion is based on the UMMA commit/wait protocol rather than TMA byte
arrival.

## Participant interface

On the study worker, a created pipeline can expose paired participants:

```text
producer, consumer = created_pipeline.make_participants()
```

Participant lifecycle:

```text
producer.acquire_and_advance() -> producer token
consumer.wait_and_advance() -> consumer token
producer_token.commit()
consumer_token.release()
```

Token fields are pipeline-specific. A TMA producer token commonly provides the
barrier associated with the acquired stage. Do not call a method merely because
its name appears on another pipeline or on the internal pipeline object.

Lower-level pipeline state APIs may expose:

```text
pipeline.make_pipeline_state(role, stages)
state.advance()
state.clone()
pipeline_object.producer_acquire(state, try_acquire_token=None)
pipeline_object.producer_get_barrier(state)
pipeline_object.producer_commit(state)
pipeline_object.producer_tail(state)
pipeline_object.consumer_wait(state, try_wait_token=None)
pipeline_object.consumer_get_barrier(state)
pipeline_object.consumer_release(state)
```

Use one coherent interface style. Do not combine participant tokens from one
style with manually advanced state from another.

## Named barriers

```text
pipeline.NamedBarrier(
    barrier_id,
    num_threads,
)
```

A named barrier coordinates a fixed CTA participant count. Hardware IDs are
in `[0, 15]`; ID 0 is also used by CTA synchronization. Barrier ID and count
must not collide with other protocols in the same kernel.

```text
pipeline.sync(barrier_id=...)
```

is a pipeline-namespace synchronization helper when available. Its signature
is not interchangeable with `cute.arch.barrier`.

## Deriving stage count

More stages can hide latency but increase:

- operand shared memory;
- barrier storage;
- live descriptors and state;
- register pressure;
- prologue and drain work.

Stage count follows from a resource and latency model. It is not supplied here
because no single value is correct across operations or shapes.

## Prologue, steady state, and drain

Any multistage loop has three logical regions:

1. **Prologue:** fill enough stages before the first consumer use.
2. **Steady state:** issue future work while consuming completed work.
3. **Drain:** finish outstanding consumers and release or tail the pipeline.

Count events for a finite number of work items:

```text
producer acquisitions == stages actually written
consumer waits == stages actually consumed
consumer releases == reusable stages returned
```

If an acquired stage is never published, or a consumed stage is never
released, later reuse can stall.

## Phase discipline

A circular slot is reused, so waiting on only its numeric stage index is
ambiguous. Pipeline state carries phase/parity. Every acquire, wait, release,
and advance must refer to the same logical generation of the slot.

Copying a stage index while dropping phase state can read stale completion from
a prior ring traversal.

## Warp and CTA participation

For each pipeline:

- define producer and consumer warps;
- derive cooperative-group sizes from those roles;
- initialize barrier storage exactly once;
- ensure all required participants reach initialization synchronization;
- ensure role branches remain compatible with named/CTA/cluster barriers;
- release stages even when a participant has no arithmetic work.

Predicates over data coordinates must not accidentally predicate required
pipeline state transitions.

## Persistent scheduling

Persistent kernels reuse CTAs across multiple work tiles. Before a CTA moves to
another tile, all stages belonging to the prior tile must be drained or
reinitialized according to the pipeline contract. Work-tile scheduling and
pipeline lifetime are coupled.

## Resource ledger

Maintain a local ledger:

| Resource | Formula source |
| --- | --- |
| staged operand bytes | exact typed SMEM layout × stage count |
| barrier bytes | pipeline factory/storage type |
| transaction bytes | physical transfers per stage |
| producer participants | role assignment |
| consumer participants | role assignment |
| prologue iterations | pipeline and work count |
| drain operations | outstanding stages/tokens |

The ledger is general; its values must be derived from the current
implementation.
