# Reasoning about layouts, tensors, and ownership

The core API chapter lists layout functions. This chapter explains how to
reason with them.

## Four views of every tensor

Record four distinct descriptions:

1. **Mathematical view:** the modes used by the operation.
2. **Physical view:** the order and strides in memory.
3. **Tiled view:** the region assigned to one CTA or cluster.
4. **Participant view:** the values owned by a thread, warp, or instruction.

A layout error often comes from silently replacing one view with another.
Equal element counts do not imply equal coordinate meaning.

## Shape and stride are a mapping

For coordinate `(i, j)` under shape `(m, n)` and stride `(s_m, s_n)`:

```text
linear_coordinate = i * s_m + j * s_n
```

The stride-one mode is physically contiguous. Names such as row-major and
column-major are shorthand; the actual shape/stride pair is authoritative.

For hierarchical shapes:

```text
((tile_mode, within_tile_mode), other_mode)
```

the matching stride hierarchy explains how each nested coordinate contributes
to the address. Preserve hierarchy while another object uses it to identify
instruction, stage, thread, or value modes.

## Logical size, codomain size, and allocation

Three quantities answer different questions:

- `cute.size(layout)` counts logical coordinates.
- `cute.cosize(layout)` describes the required linear codomain span.
- `cute.size_in_bytes(dtype, layout)` describes typed storage bytes.

A strided layout may have `cosize > size`. A composed or swizzled layout may
also address a nontrivial codomain. Allocation must cover the codomain, not just
the logical count.

## Static versus dynamic structure

Static facts:

- rank and tuple hierarchy;
- element and address-space types;
- instruction atoms;
- most thread/value mappings;
- specialization choices.

Potentially dynamic facts:

- problem extents;
- coordinates;
- selected strides;
- loop trip counts;
- predicates.

Making everything static produces many specializations. Making structure
dynamic prevents Python from constructing types and layouts. The useful middle
ground keeps hierarchy and divisibility static while representing only varying
extent leaves as runtime values.

## Tiling versus partitioning

Tiling answers:

```text
Which region of the whole tensor belongs to this CTA or work item?
```

Partitioning answers:

```text
Which values in that region belong to this participant or hardware atom?
```

Typical object flow:

```text
whole tensor
  -> CTA tile through local_tile
  -> operation-specific partition through ThrCopy or ThrMma
  -> fragment compatible with an instruction
```

Do not substitute a manually reshaped tensor for an operation-specific
partition merely because its shape is similar. The operation object may encode
address-space, ownership, and vectorization constraints that are invisible in
the printed extent.

## Copy congruence

For a tiled copy, verify:

1. the participant slice came from the same `TiledCopy`;
2. `partition_S` receives the source tensor;
3. `partition_D` receives the destination tensor;
4. per-participant value counts agree;
5. vector modes map to physically compatible strides;
6. all active participants have valid coordinates;
7. predicates cover every possibly invalid source and destination coordinate.

A predicated synchronous copy and a transaction-counted asynchronous copy have
different tail requirements. Do not promise bytes for accesses that a
predicate suppresses.

## MMA congruence

An MMA connects:

- logical M/N/K modes;
- the tiled instruction shape;
- A, B, and accumulator layouts;
- participant ownership;
- operand memory spaces and formats.

The A partition consumes an M/K view, B consumes an N/K view under the chosen
physical convention, and C/D consume an M/N view. The operation's partition
methods establish the exact mapping.

The physical B tensor may expose N/K even when the mathematical notation writes
K/N. That representation is a layout decision, not an implicit transpose.
Always derive the oracle and indexing from the declared physical view.

## Shared-memory layouts and swizzles

A shared-memory layout must satisfy all of:

- storage fits;
- pointer alignment is sufficient;
- the producer copy can write it;
- the consumer copy or MMA can read it;
- stage modes do not overlap;
- bank mapping is legal and useful.

Swizzling changes address bits to alter bank access. Reusing a swizzle with a
different element width, tile, or consumer can invalidate the mapping. Prefer a
layout constructed for the selected operation instead of treating swizzle
parameters as independent tuning values.

## Tensor memory is a separate address space

Blackwell tensor memory is not register memory or shared memory. A TMEM tensor
requires:

- a TMEM allocation;
- a compatible TMEM layout;
- an instruction that writes it;
- completion before reads;
- a matching TMEM-to-register copy;
- release after the final consumer.

A fragment initially carrying only layout metadata is not usable storage until
it is bound to an allocated pointer in the correct address space.

## Bounds and tails

For each logical mode, state whether the implementation:

- requires exact divisibility;
- pads a physical allocation;
- predicates a compatible load/store;
- uses a separate tail path.

Predicating a write does not make an out-of-bounds read safe. Predicating a TMA
transaction may also require changing transaction bytes and pipeline
participation. Derive safety at every memory boundary.

## Coordinate audit

For a small hypothetical coordinate:

1. expand the whole-tensor layout;
2. calculate the selected CTA tile coordinate;
3. identify retained and selected modes;
4. expand the participant partition;
5. compute the physical address;
6. check bounds and alignment;
7. confirm the consumer interprets the modes identically.

This audit is more reliable than changing index arithmetic based only on an
output pattern.
