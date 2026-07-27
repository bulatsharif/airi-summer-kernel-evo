# Blackwell memory movement and pipelines

Use this reference when implementing GMEM/SMEM/TMEM movement, TMA descriptors,
pipeline stages, barriers, warp specialization, accumulator movement, or
persistent scheduling.

## Contents

- [Memory spaces](#memory-spaces)
- [Dense GEMM dataflow](#dense-gemm-dataflow)
- [Warp and CTA roles](#warp-and-cta-roles)
- [TMA descriptors and copies](#tma-descriptors-and-copies)
- [Shared-memory allocation](#shared-memory-allocation)
- [Pipeline model](#pipeline-model)
- [Pipeline creation](#pipeline-creation)
- [Producer protocol](#producer-protocol)
- [Consumer protocol](#consumer-protocol)
- [TMEM allocation and MMA](#tmem-allocation-and-mma)
- [TMEM-to-register epilogue](#tmem-to-register-epilogue)
- [One-CTA and two-CTA mode](#one-cta-and-two-cta-mode)
- [Persistent scheduling](#persistent-scheduling)
- [Deadlock diagnosis](#deadlock-diagnosis)
- [Invariant checklist](#invariant-checklist)

## Memory spaces

Blackwell CuTe kernels use four relevant memory spaces:

| Space | Typical use | Lifetime/ownership |
|---|---|---|
| GMEM | Inputs, outputs, scale factors | Grid-wide allocation |
| SMEM | Staged A/B/scales/output tiles and barriers | CTA/cluster |
| TMEM | tcgen05 accumulators and block scales | CTA group |
| RMEM | Thread-local epilogue values and scalars | Thread |

The normal dense path is:

```text
A/B: GMEM --TMA--> SMEM --tcgen05.mma--> TMEM accumulator
C:   TMEM --tcgen05.ld--> RMEM --convert/epilogue--> GMEM
```

An optional epilogue uses:

```text
TMEM -> RMEM -> SMEM --TMA store--> GMEM
```

Block-scaled MMA adds:

```text
SFA/SFB: GMEM --TMA--> SMEM --tcgen05.cp--> TMEM
```

## Dense GEMM dataflow

For each output tile:

1. Select the work tile `(M_tile,N_tile,L)`.
2. Build A/B global tile views across K.
3. A producer warp acquires an empty SMEM stage.
4. The producer issues TMA A/B copies associated with that stage's full
   transaction barrier.
5. The MMA warp waits for the stage to become full.
6. The MMA warp selects the stage's A/B SMEM descriptors.
7. Set `ACCUMULATE=False` for the first K tile and `True` afterward.
8. Issue asynchronous tcgen05 MMA into TMEM.
9. Release the SMEM stage when its operand data is no longer needed.
10. After the final K tile, make accumulator completion visible to epilogue
    warps.
11. Load TMEM accumulator tiles to per-thread registers.
12. Convert, apply epilogue, and store.
13. Release or deallocate TMEM before CTA exit.

The stages overlap step 4 for future K tiles with steps 5-9 for current K tiles.

## Warp and CTA roles

A typical 128-thread dense kernel assigns four warps:

- one TMA/load warp
- one MMA/control warp
- two epilogue warps

Exact warp counts differ across kernels. What matters is that every role has a
single, explicit participation predicate and all barriers are initialized for
the participating group.

Common role values:

```python
warp_idx = cute.arch.warp_idx()
warp_idx = cute.arch.make_warp_uniform(warp_idx)
lane_idx = cute.arch.lane_idx()
```

Use compile-time constants for role IDs. Make branch predicates warp-uniform.
Do not let one lane issue a warp-collective operation while siblings take a
different path.

CTA clusters introduce leader and peer roles. Multicast TMA and two-CTA MMA
require cluster-rank-aware masks. Reuse the masks derived by pipeline/helper
utilities.

## TMA descriptors and copies

TMA setup happens in the host JIT function because descriptors depend on tensor
layout and static tile configuration.

For A/B in the baseline dense kernel:

```python
tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
    a_op,
    a,
    a_smem_layout,
    mma_tiler,
    tiled_mma,
    cluster_layout_vmnk.shape,
)

tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
    b_op,
    b,
    b_smem_layout,
    mma_tiler,
    tiled_mma,
    cluster_layout_vmnk.shape,
)
```

The helper returns:

- a copy atom/descriptor
- a tensor view shaped for tiled TMA coordinates

The kernel creates per-CTA source partitions and a shared destination view, then
issues:

```python
cute.copy(
    tma_atom,
    source_partition[None, k_tile],
    shared_destination[None, pipeline_stage],
    tma_bar_ptr=full_barrier,
    mcast_mask=mask,
)
```

Exact argument names vary by copy atom. Preserve these invariants:

- TMA source layout matches the actual global tensor.
- TMA box/tile matches the shared destination.
- descriptor element type matches the physical data.
- multicast mask matches the cluster layout and operand sharing direction.
- all bytes expected by the transaction barrier are actually issued.

For FP8, TMA copy width and tile origins must preserve at least the promised
16-byte alignment.

## Shared-memory allocation

Host/JIT setup computes staged composed layouts. Kernel allocation binds them:

```python
smem = cutlass.utils.SmemAllocator()
sA = smem.allocate_tensor(
    element_type=a_dtype,
    layout=a_smem_layout_staged.outer,
    byte_alignment=128,
    swizzle=a_smem_layout_staged.inner,
)
```

Shared storage also contains:

- pipeline full/empty mbarriers, unless reserved allocation is used
- optional accumulator/epilogue buffers
- optional SFA/SFB stages
- scheduler response or CLC storage in persistent kernels

Calculate capacity from all staged tensors and barriers. Do not tune stage count
without recomputing shared-memory use and occupancy.

## Pipeline model

A circular pipeline has:

- `num_stages`
- producer state: count, index, phase
- consumer state: count, index, phase
- one full barrier per stage
- one empty barrier per stage

At a conceptual level:

```text
empty(stage) -> producer owns stage
producer issues async transaction
full(stage) -> consumer owns stage
consumer finishes using stage
empty(stage) -> reusable
```

`PipelineState.advance()` increments count/index and flips phase when wrapping
the ring. Do not manually change index without phase.

For a producer start at stage 0, create the producer state for an initially
empty ring. Consumer initialization can be offset or phase-reversed depending
on the pipeline utility. Use `make_pipeline_state(...)` or the pattern from the
installed example instead of inventing phase values.

## Pipeline creation

For TMA producer and tcgen05 consumer, use
`pipeline.PipelineTmaUmma.create(...)`.

Conceptual parameters:

```python
ab_pipeline = pipeline.PipelineTmaUmma.create(
    num_stages=num_ab_stages,
    producer_group=producer_group,
    consumer_group=consumer_group,
    tx_count=num_tma_load_bytes,
    barrier_storage=barrier_ptr,
    cta_layout_vmnk=cluster_layout,
)
```

The exact signature may include:

- cluster layout
- multicast signaling flag
- defer-sync option
- pipeline name/profiling metadata

`tx_count` is the bytes expected for one full stage, not the allocation size for
all stages. For dense two-operand load:

```text
tx_count = bytes(A stage) + bytes(B stage)
```

For block scaling, include SFA and SFB bytes.

Call the initialization arrive/wait protocol required by the example before
role-specific mainloops. Cluster kernels require the initialization fence and
cluster synchronization to make barriers and shared memory visible.

## Producer protocol

The explicit producer sequence is:

```python
producer_try = pipe.producer_try_acquire(producer_state)
pipe.producer_acquire(producer_state, producer_try)

barrier = pipe.producer_get_barrier(producer_state)
# Issue all TMA copies for this stage using barrier.

pipe.producer_commit(producer_state)  # often a no-op for TMA
producer_state.advance()
```

Some helpers return a combined token with index/count/barrier and advance
automatically. Follow one API style consistently.

For TMA pipelines, hardware completes the full barrier by decrementing expected
transaction bytes. A missing copy leaves the barrier permanently incomplete.

At loop end, call the producer tail method when required so no delayed barrier
signal survives CTA exit.

## Consumer protocol

The explicit consumer sequence is:

```python
consumer_try = pipe.consumer_try_wait(consumer_state)
pipe.consumer_wait(consumer_state, consumer_try)

# Read the stage and issue MMA.

pipe.consumer_release(consumer_state)
consumer_state.advance()
```

Some examples use:

```python
full_token = pipe.consumer_wait_and_advance()
# ...
full_token.release()
```

Release only after the async consumer no longer reads the SMEM stage. For
tcgen05, pipeline utilities encode the appropriate MMA/barrier signaling;
do not replace this with a thread fence guessed from CUDA C++ patterns.

## TMEM allocation and MMA

TMEM is addressed in columns. Compute required columns from the tiled MMA,
accumulator type, tile shape, and any number of accumulator stages.

Conceptual allocation:

```python
tmem = cutlass.utils.TmemAllocator(...)
tmem.allocate(num_cols=num_tmem_cols)
tmem_ptr = tmem.retrieve_ptr(acc_dtype)

acc_layout = tiled_mma.make_fragment_C(acc_shape_staged).layout
accumulators = cute.make_tensor(tmem_ptr, acc_layout)
```

For each K tile:

```python
tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile_idx != 0)
cute.gemm(
    tiled_mma,
    accumulators,
    a_fragment_for_stage,
    b_fragment_for_stage,
    accumulators,
)
```

tcgen05 MMA is asynchronous. A host-side `torch.cuda.synchronize()` is not a
replacement for the in-kernel ordering needed before TMEM loads.

For block scaling, SFA/SFB occupy additional TMEM and the MMA operands are:

```python
[a_fragment, sfa_tmem]
[b_fragment, sfb_tmem]
```

## TMEM-to-register epilogue

The epilogue needs a tiled TMEM load compatible with the accumulator layout:

```python
copy_atom_t2r = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    acc_dtype,
)
tiled_copy_t2r = tcgen05.make_tmem_copy(copy_atom_t2r, accumulator_tile)
thread_copy = tiled_copy_t2r.get_slice(epilogue_thread_idx)
tmem_src = thread_copy.partition_S(accumulator_tensor)
gmem_dst = thread_copy.partition_D(global_c_tile)
rmem = cute.make_rmem_tensor(gmem_dst_slice.shape, acc_dtype)
cute.copy(tiled_copy_t2r, tmem_src_slice, rmem)
```

Then:

1. apply alpha/beta or activation in accumulator precision when specified
2. convert to output dtype
3. predicate output coordinates when supported
4. store directly or through a staged SMEM/TMA epilogue

Do not recast accumulator bits as output type. Use an actual numeric conversion.

## One-CTA and two-CTA mode

One-CTA MMA:

- one CTA owns the instruction tile
- simpler cluster/barrier behavior
- useful first correctness target

Two-CTA MMA:

- two CTAs cooperate on one MMA
- the pair divides the instruction's M ownership
- B can be shared/multicast across the pair
- cluster M must accommodate pairs
- TMEM allocation and remote barrier masks differ
- both CTAs must execute compatible MMA/control paths

Do not enable `CtaGroup.TWO` by changing only the enum. Recompute:

- CTA tile M
- cluster layout
- multicast counts/masks
- grid
- SMEM partitioning
- TMEM allocation
- epilogue ownership

## Persistent scheduling

A persistent kernel launches a bounded number of CTAs/clusters and assigns
multiple output tiles to each.

Benefits:

- amortized setup
- improved occupancy control
- overlap between mainloop and epilogue across work tiles
- better handling of large tile grids

Costs:

- scheduler state and termination protocol
- accumulator ping-pong
- more pipeline tail hazards
- harder debugging

Start with static tile scheduling. Move to persistent only after:

- the non-persistent kernel is correct
- the task has enough tiles to benefit
- the remote hardware occupancy is known

For every new work tile, reset accumulate mode and any per-tile phase/state that
is not intentionally continuous.

## Deadlock diagnosis

If the kernel hangs, do not tune. Check in this order:

1. Every CTA reaches pipeline initialization.
2. Cluster shape at launch matches the compiled cluster layout.
3. Producer/consumer group sizes match actual participating warps.
4. Every acquired stage receives every expected TMA copy.
5. `tx_count` equals issued bytes for one stage.
6. Every consumed stage is released exactly once.
7. Pipeline state advances with correct phase on wrap.
8. Multicast masks target valid cluster ranks.
9. Two-CTA peers follow the same number of MMA iterations.
10. Producer/consumer tail calls occur before exit.
11. K tile count is identical between load and MMA roles.
12. Early out does not skip a barrier for only some CTAs.

Reduce to:

- one output tile
- one batch
- one K tile
- one CTA group when possible
- one or two stages

Then reintroduce concurrency.

## Invariant checklist

- [ ] Memory-space path is explicit for A, B, scales, accumulator, and C.
- [ ] Each warp/CTA has one role predicate.
- [ ] SMEM allocation matches host-computed composed layouts.
- [ ] TMA descriptors match global and shared layouts.
- [ ] Transaction bytes equal bytes issued per stage.
- [ ] Producer and consumer states use matching stage count.
- [ ] Every acquire/wait has one corresponding release/advance.
- [ ] Accumulate mode resets on the first K tile of every output tile.
- [ ] MMA completion is ordered before TMEM load.
- [ ] TMEM columns cover accumulator plus scale/overlap allocations.
- [ ] One-CTA/two-CTA cluster, grid, masks, and partitions agree.
- [ ] Pipeline tail and TMEM release happen before CTA exit.
