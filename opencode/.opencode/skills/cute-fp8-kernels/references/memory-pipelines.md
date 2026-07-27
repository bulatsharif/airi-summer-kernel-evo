# Blackwell memory and pipelines

## Contents

[Dataflow and ownership](#dataflow-and-ownership) ·
[TMA contract](#tma-contract) · [Pipeline protocol](#pipeline-protocol) ·
[TMEM and MMA](#tmem-and-mma) · [Epilogue](#epilogue) ·
[Clusters, two-CTA, persistence](#clusters-two-cta-persistence) ·
[Deadlock checklist](#deadlock-checklist)

## Dataflow and ownership

```text
GMEM A/B[/SFA/SFB]
  --TMA producer--> staged SMEM
  --tcgen05 MMA[/scale copy]--> TMEM accumulator/scales
  --TMEM load--> registers
  --convert/epilogue--> GMEM C (direct or SMEM/TMA)
```

Assign each operation to a warp/warp group:

- TMA producer: acquire a stage token and issue asynchronous copies against
  its barrier; byte arrival completes the producer phase
- MMA consumer: wait for full stage, issue MMA, release stage
- epilogue: wait for accumulator, load TMEM fragments, convert/store

Leader election and collective participation must match the installed recipe.
Inactive warps still need compatible CTA/cluster synchronization where required.

## TMA contract

A TMA transfer couples:

- GMEM tensor/layout and alignment
- SMEM destination layout and tile
- descriptor and copy atom
- transaction byte count
- barrier/stage
- cluster multicast mask/participants

For dense A/B, transaction bytes cover both staged tiles. MXFP8 adds SFA/SFB
bytes. The barrier becomes ready only when the promised bytes arrive; wrong
bytes can hang or expose incomplete data.

Multicast sends a tile to selected CTAs. Derive masks from cluster coordinates
and ensure receivers use the same stage protocol. Do not enable multicast
without a compatible cluster layout.

## Pipeline protocol

A staged pipeline is a ring of storage plus barrier phases, not an ordinary
index. The conceptual protocol is:

```text
producer: acquire -> issue async work -> commit -> advance
consumer: wait -> consume -> release -> advance
```

Do not translate those conceptual verbs into guessed participant methods.
Candidate-mode `PipelineTmaUmma` folds advance into
`acquire_and_advance()`, and TMA byte arrival completes the producer phase:

```text
ab_producer.acquire_and_advance()
  -> issue TMA copies using token.barrier
ab_consumer.wait_and_advance()
  -> consume
  -> consumer_token.release()
```

There is no `ab_producer.commit()` in that protocol. By contrast, the
`PipelineUmmaAsync` accumulator producer returns a token that is explicitly
committed:

```text
acc_producer.acquire_and_advance()
  -> issue all UMMA contributions
  -> producer_token.commit()
acc_consumer.wait_and_advance()
  -> read/store accumulator
  -> consumer_token.release()
```

Exact candidate-mode code is in
[candidate-kernel-patterns.md](candidate-kernel-patterns.md).

Create pipeline storage/state with the installed helpers. Preserve:

- stage count and initial phase
- producer/consumer arrival counts
- producer/consumer role assignment
- transaction bytes
- prologue fill and drain/tail behavior
- identical iteration counts for paired acquire/release

More stages overlap TMA and MMA but multiply SMEM and barriers. Recompute storage
and launch resources whenever stages or tile change.

## TMEM and MMA

Blackwell tcgen05 MMA is asynchronous. A/B are normally consumed from SMEM;
accumulators live in TMEM. Allocate TMEM columns using the tiled MMA and
epilogue fragment requirements.

For the K loop:

```text
first K tile: initialize/overwrite accumulator
later tiles: accumulate
```

Wait for required MMA completion before TMEM loads. Load TMEM fragments to
registers with the matching tiled copy, then convert/apply the epilogue.
Release TMEM only after every consumer is finished.

MXFP8 additionally copies staged scales to their MMA-compatible TMEM layout and
uses the matching scale block for each K tile.

## Epilogue

Simple path:

```text
TMEM -> registers -> conversion -> predicated GMEM store
```

TMA-store path:

```text
TMEM -> registers -> epilogue SMEM -> TMA GMEM store
```

The latter adds SMEM layout, synchronization, and store pipeline state. Start
with the simplest path compatible with required shapes. Tail support belongs to
the selected epilogue; do not assume it.

## Clusters, two-CTA, persistence

One-CTA MMA is the debugging baseline. Two-CTA changes instruction tiling,
cluster constraints, TMEM ownership, and epilogue participation. Cluster
dimensions also affect multicast and grid mapping.

Persistent kernels add a work-tile scheduler. Every CTA/cluster must agree on
work ownership and cleanly drain pipelines before advancing or exiting. Add
persistence only after a non-persistent kernel is correct.

## Deadlock checklist

On a tiny timeout, reduce to one CTA/K tile, minimal stages, one-CTA,
non-persistent, simple epilogue. Compare:

- acquired, committed, waited, and released stage counts
- expected arrivals and TMA bytes
- stage indices/phases
- warp/CTA/cluster participation
- multicast sender/receiver masks
- prologue and final drain
- TMEM allocation/release ordering

Never “fix” a hang by changing only timeout or barrier counts without deriving
the protocol.
