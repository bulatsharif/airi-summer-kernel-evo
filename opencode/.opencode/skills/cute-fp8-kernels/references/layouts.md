# Layouts and partitioning

A layout maps a logical coordinate to a linear storage coordinate. Always track:

```text
logical shape + hierarchical modes + stride/layout + element type + alignment
```

## Contents

[Shape, stride, hierarchy](#shape-stride-hierarchy) ·
[GEMM convention](#gemm-convention) ·
[Tiling and partitioning](#tiling-and-partitioning) ·
[Shared-memory layouts and swizzles](#shared-memory-layouts-and-swizzles) ·
[Tails and alignment](#tails-and-alignment) ·
[Debug layout failures](#debug-layout-failures)

## Shape, stride, hierarchy

For shape `(d0,d1)` and stride `(s0,s1)`, offset is `i0*s0+i1*s1`.
Stride-1 identifies the contiguous mode; shape alone does not define major
order. Nested modes preserve tile/thread/value structure and affect TMA/MMA
congruence—flatten only with proof.

Use:

- `cute.size`: logical elements
- `cute.cosize`: backing span required by a layout
- `cute.size_in_bytes`: allocation/transaction bytes

Dynamic extents can share a static layout type when structure/stride pattern is
fixed. Tile, MMA, copy, and swizzle layouts are usually compile-time.

## GEMM convention

State mathematical and physical tensors separately:

```text
math: A[M,K,L], B[K,N,L], C[M,N,L]
common CuTe physical view: A[M,K,L], B[N,K,L], C[M,N,L]
```

The B view lets the tiled MMA interpret its K and N modes without physically
transposing storage. The oracle must follow the declared mapping.

## Tiling and partitioning

Typical sequence:

1. create whole-problem tensor views
2. use block/cluster coordinate with `local_tile`
3. create tiled MMA/copy
4. obtain a thread or warp-group slice
5. partition source/destination through the same object
6. slice K tile and pipeline stage

`local_tile` answers which logical region a CTA owns. `partition_S/D` answers
which values each participant moves/consumes. A tensor can have correct shape
yet be incompatible with an atom because modes are ordered or grouped wrongly.

Check:

- source/destination partitions have equal value count
- vector dimension maps to stride-1 storage
- MMA A/B/C partitions agree with instruction modes
- batch and cluster modes are not dropped
- grid/cluster coordinates map to distinct promised output tiles

## Shared-memory layouts and swizzles

SMEM layouts must simultaneously:

- fit staged storage
- support TMA destination requirements
- avoid harmful bank conflicts
- match MMA/copy consumption
- keep stage modes distinct

Use architecture utilities such as `make_smem_layout_a/b`; do not invent
swizzles or apply a row-major mental model to them. Allocate using `cosize`/byte
calculations from the exact layout.

Block-scaled SFA/SFB have separate logical, GMEM, SMEM, and TMEM layouts. The
logical coordinate chooses row/column and K block; physical layout is
instruction-specific and must come from block-scaled utilities.

## Tails and alignment

Alignment is a contract over base pointer and strides. The dense FP8 baseline
requires at least 16 contiguous bytes. Verify actual pointer alignment and that
leading/batch strides preserve it.

An exactly tiled mainloop does not support tails automatically. For each mode,
choose one:

- require divisibility and reject otherwise
- pad storage/shape under an explicit contract
- predicate with a copy/epilogue path designed for tails

Predicating memory access without updating transaction/barrier accounting can
deadlock.

## Debug layout failures

Print trace-time tensor/layout objects, shape/stride, `size`, `cosize`, tile
coordinate, and partitions. Reduce to one tile and initialize output with a
sentinel.

| Symptom | Likely issue |
|---|---|
| transpose | math/physical B mismatch |
| repeated rows/columns | omitted mode or bad partition |
| periodic stripes | vector width/alignment |
| stage-dependent error | wrong stage slice/phase |
| edges only | unsupported tail |
| block-scale boundaries | scale layout/index |

Do not fix layouts with ad hoc index arithmetic until the intended coordinate
mapping is written explicitly.
