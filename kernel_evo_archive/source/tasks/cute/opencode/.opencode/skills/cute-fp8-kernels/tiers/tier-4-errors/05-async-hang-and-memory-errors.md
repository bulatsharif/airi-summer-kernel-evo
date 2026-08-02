# Asynchronous, hang, and memory errors

## Timeout on a small workload

A small deterministic timeout usually indicates a synchronization protocol
error, not insufficient timeout.

Audit one stage:

```text
initial phase
producer acquisition
expected bytes
asynchronous issue
explicit arrivals
consumer wait
consumer release
phase advance
```

Then compare counts across prologue, steady state, and drain.

## Producer/consumer count mismatch

Symptoms:

- hang only after ring wraparound;
- one stage count works and another does not;
- final work item never completes.

Count:

```text
acquired stages
published/committed stages
waited stages
released stages
tail/drain operations
```

Every logical stage generation must complete its lifecycle.

## Wrong transaction byte count

If expected bytes exceed issued bytes, the barrier never completes. If expected
bytes are too small, consumers can observe incomplete data.

Recompute from physical transfers, including:

- every operand;
- scale-factor transfers;
- packed/internal element width;
- multicast behavior;
- predicates and skipped transfers.

Do not modify bytes solely until a timeout disappears; validate loaded data.

## Double or missing election

Single-thread state operations need election. TMA copy issue already performs
its own election.

Potential failures:

- every lane initializes one barrier;
- no lane registers expected bytes;
- a TMA copy is nested under an extra election and does not follow the expected
  issue protocol;
- tcgen commit is issued by an incompatible participant set.

Classify each operation as single-lane, warp collective, CTA collective, or
instruction-owned election.

## Divergent collective control flow

All required participants must reach:

- CTA/warp/cluster barriers;
- pipeline initialization;
- stage waits and releases;
- TMEM allocation waits;
- instruction commits.

A data predicate may suppress arithmetic or a store, but it must not suppress a
required collective transition.

## Unspecified launch failure

This often surfaces at a later synchronization. Suspect:

- illegal address;
- invalid asynchronous instruction operand;
- use-before-completion;
- use-after-release;
- broken cluster/distributed-SMEM participation.

Insert synchronization after major asynchronous boundaries to identify the
first failing region. Remove those synchronizations after diagnosis.

## Illegal memory access

Audit:

1. logical coordinate;
2. physical stride and byte offset;
3. allocation span;
4. source/destination address space;
5. alignment;
6. boundary predicate;
7. object lifetime.

For a tiled/partitioned tensor, perform this audit after every view
transformation, not only on the whole tensor.

## Stale or partially updated output

Possible mechanisms:

- missing store for some participants;
- accumulator consumer read before completion;
- stage reused before release;
- output store completed asynchronously after lifetime ended;
- predicate excludes valid coordinates;
- grid does not cover all output tiles.

Sentinel initialization and repeated runs can distinguish missing writes from
wrong arithmetic.

## Stage-dependent numerical error

If correctness changes with stage count, prioritize:

- omitted stage mode in SMEM;
- wrong stage pointer;
- phase mismatch;
- premature reuse;
- barrier bytes shared across stages;
- prologue/drain off by one.

The mathematical operation should not depend on the number of storage stages.

## Cluster-only hang

Check:

- all CTAs in the cluster launch;
- multicast sender and receiver masks;
- remote barrier address/rank;
- cluster arrival/wait participation;
- multi-CTA instruction ownership;
- output responsibility.

Reduce cluster complexity only to identify which cross-CTA contract is broken.

## Lifetime ordering

General order:

```text
allocate
publish initialization
write asynchronously
wait for completion
read/consume
wait for consumers if necessary
release
```

Python lexical scope does not enforce device lifetime.
