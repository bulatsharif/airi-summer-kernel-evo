# Layout, copy, and MMA errors

## Operation creation failed

This usually means every named Python object exists, but their combination is
illegal. Candidate dimensions:

- operand dtype;
- accumulator dtype;
- address space;
- major mode;
- atom/tile shape;
- CTA-group mode;
- operand fragment layout;
- alignment.

Reduce to construction of the failing operation. Compare each parameter role
against Tier II before adding TMA, pipeline, or epilogue logic.

## IR verification failed

IR verification is later than Python type construction. Common mechanisms:

- an operation received a value from the wrong address space;
- a region yields incompatible types;
- a layout/instruction constraint was not enforced by the Python wrapper;
- a runtime value entered a type-forming position;
- unsupported sub-byte scalar access reached lowering.

The first verifier message usually names the invalid operation or operand.
Inspect that boundary instead of rewriting the whole kernel.

## Layout versus tensor passed to partitioning

Symptoms:

- partition method rejects an object;
- generated operation expects a tensor value;
- shape prints correctly but lowering fails.

`ThrMma.partition_A/B/C` and `ThrCopy.partition_S/D` consume tensor views.
Pass the allocated or pointer-backed tensor, not its `.layout`.

## Partition method on the wrong owner

Observed mistakes include:

```text
tensor.partition_D(...)
top-level cute.partition_D(...)
TiledMma.partition_D(...)
```

Partition ownership:

```text
ThrCopy.partition_S / partition_D
ThrMma.partition_A / partition_B / partition_C
```

Obtain the participant slice from the matching tiled object first.

## Shape matches but copy is illegal

Check:

- source and destination address spaces;
- source/destination partition value layouts;
- vector direction and stride-one mode;
- element/internal types;
- base and stride alignment;
- participant count;
- predicate layout.

Manual reshaping can preserve extents while losing instruction-compatible
ownership.

## Transposed or repeated output

Pattern-based hypotheses:

| Pattern | First layout hypothesis |
| --- | --- |
| exact transpose | mathematical and physical mode conventions differ |
| repeated rows/columns | a mode was omitted or fixed incorrectly |
| periodic stripes | vector width, stride, or participant mapping mismatch |
| correct first tile only | work-tile coordinate not advanced |
| correct interior, wrong edge | boundary predicate or tail policy |

Confirm with coordinate-to-address calculations. Do not compensate with an
ad hoc output transpose until the physical contract is explicit.

## TMA partition mismatch

Symptoms:

- descriptor construction succeeds but launch fails;
- an equal-shape manual partition causes illegal instruction;
- only multicast/cluster cases fail.

Verify the TMA atom, TMA coordinate tensor, SMEM layout, CTA layout, and
coordinate all came from one construction. Preserve the descriptor-returned
tensor and use `cpasync.tma_partition`.

## Shared-memory overlap

Symptoms:

- stage-dependent corruption;
- changing stage count changes numerical pattern;
- one operand appears to overwrite another;
- launch succeeds only with smaller storage.

Check:

- full staged layout `cosize`;
- element width;
- allocator offsets and alignments;
- whether a stage mode was omitted or added twice;
- barrier storage versus operand storage;
- mutually exclusive branch allocation policy.

## Accumulator initialization error

Patterns:

- error grows with reduction length;
- only the first reduction tile contributes;
- output changes between repeated runs;
- NaN or stale values appear before epilogue arithmetic.

Confirm:

- first contribution initializes/overwrites;
- later contributions accumulate;
- TMEM is allocated before use;
- source accumulator does not read uninitialized storage;
- each logical output has one accumulator owner.

## Incomplete fragment store

Symptoms:

- periodic untouched values;
- sentinel remains in a subset;
- only one participant's output appears.

Trace:

```text
TMEM fragment
-> participant TMEM partition
-> register destination partition
-> output coordinate mapping
-> store predicate
```

Compare value counts at each boundary.
