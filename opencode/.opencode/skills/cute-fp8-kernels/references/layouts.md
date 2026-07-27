# CuTe layouts and tensors

This reference explains the layout reasoning needed for CuTe DSL kernels. Read it
before changing tensor shapes, strides, tiling, partitioning, swizzles, or
framework conversion.

## Contents

- [Mental model](#mental-model)
- [Shape and stride notation](#shape-and-stride-notation)
- [Hierarchical modes](#hierarchical-modes)
- [Static and dynamic layouts](#static-and-dynamic-layouts)
- [Framework tensors and DLPack](#framework-tensors-and-dlpack)
- [Layout operations](#layout-operations)
- [Tiling and partitioning](#tiling-and-partitioning)
- [Shared-memory swizzles](#shared-memory-swizzles)
- [GEMM tensor conventions](#gemm-tensor-conventions)
- [Alignment and divisibility](#alignment-and-divisibility)
- [Debugging layouts](#debugging-layouts)
- [Checklist](#checklist)

## Mental model

A layout maps a logical coordinate to a storage offset:

```text
layout(coord) -> linear offset
tensor(coord) = *(engine + layout(coord))
```

A CuTe tensor is therefore:

```text
Tensor = Engine composed with Layout
```

The engine is commonly a pointer in GMEM, SMEM, TMEM, or generic memory. The
layout owns the indexing rule. Two tensors can share the same logical shape and
still have different physical storage because their strides or composed
swizzles differ.

Never infer row/column major from a shape alone.

## Shape and stride notation

CuTe prints layouts as:

```text
shape:stride
```

Examples:

```text
(4,8):(8,1)    # second mode contiguous; conventional row-major 4x8
(4,8):(1,4)    # first mode contiguous; conventional column-major 4x8
(4,8):(0,1)    # first mode broadcast; every row aliases the same storage
```

For coordinate `(i,j)`, a flat stride maps:

```text
offset = i * stride[0] + j * stride[1]
```

Use `cute.crd2idx(coord, layout)` when the mapping needs to be inspected inside
a JIT function.

Important size terms:

- `cute.size(layout)` is the number of logical coordinates.
- `cute.cosize(layout)` is the storage span required by its codomain.
- `cute.size_in_bytes(dtype, layout)` converts the layout storage requirement to
  bytes.

`size` and `cosize` can differ for strided, broadcast, sparse, or composed
layouts.

## Hierarchical modes

CuTe shapes and strides can be nested:

```python
layout = cute.make_layout(
    ((8, 4), (16, 2)),
    stride=((1, 8), (32, 512)),
)
```

Each top-level mode can contain submodes. Hierarchy expresses how a logical
dimension is factored into instruction, warp, CTA, cluster, or pipeline
components without losing the relationship between those factors.

Common indexing idioms:

- `None` keeps an entire mode.
- An integer selects one coordinate in a mode.
- A tuple selects coordinates recursively.
- `cute.slice_(tensor_or_layout, coord)` selects modes or a pipeline stage.
- `cute.flatten` removes hierarchy; use only when the consumer expects it.
- `cute.coalesce` merges compatible adjacent modes while preserving mapping.

Do not flatten just to make shapes look simpler. Tiled MMA and copy operations
use hierarchy to decide which threads and descriptors own each subspace.

## Static and dynamic layouts

A static layout has compile-time-known shape and stride. It enables:

- compile-time bounds and branch elimination
- unrolling
- exact descriptor construction
- stronger vectorization and alignment proofs

It also specializes the compiled executor. Calling an executor compiled for
static `(3):(1)` with a `(5):(1)` tensor does not change the executor's expected
layout and can silently process only the compiled extent.

A dynamic layout contains runtime values, printed as `?`, optionally with
divisibility constraints:

```text
(?,?):(?,1)
(?{div=16},?):(?,1)
```

Use dynamic layouts for reusable shape-polymorphic kernels, but retain static
tile sizes, leading unit stride, alignment, and divisibility where possible.

Choose deliberately:

- Fixed benchmark shapes: prefer static dimensions when compilation count is
  acceptable.
- Reusable operator over many M/N values: make only those problem modes dynamic.
- MMA instruction, stage count, tile, cluster shape, and datatype: keep static.

CuTe layout algebra currently uses 32-bit shapes/strides internally. DLPack
conversion can retain 64-bit dynamic strides for address safety; opt into
32-bit strides only when the storage span is proven below `INT32_MAX`.

## Framework tensors and DLPack

Passing a Torch tensor directly across a JIT boundary performs implicit DLPack
conversion and usually creates a fully dynamic layout except for a deduced
unit-stride leading dimension.

Explicit conversion:

```python
from cutlass.cute.runtime import from_dlpack

packed = from_dlpack(
    torch_tensor,
    assumed_align=16,
    use_32bit_stride=True,
)
```

Properties:

- conversion is zero-copy
- the CuTe tensor shares the source allocation
- the source tensor must remain alive
- explicit conversion initially produces a static layout
- `assumed_align` becomes part of the pointer type and JIT identity
- conversion overhead can matter for launch-bound kernels

Use:

```python
packed.mark_layout_dynamic(leading_dim=-1)
```

to make shape/stride modes dynamic while retaining the selected unit stride and
broadcast strides. Use:

```python
packed.mark_compact_shape_dynamic(
    mode=0,
    stride_order=(0, 1),
    divisibility=16,
)
```

for finer control over one compact dynamic dimension.

Beware that DLPack canonicalizes some size-one strides to one. A tensor with
several size-one dimensions can therefore have several apparent unit strides,
making automatic leading-dimension deduction ambiguous. Specify it explicitly.

## Layout operations

Frequently used operations:

| Operation | Role |
|---|---|
| `cute.make_shape` | Create a hierarchical shape |
| `cute.make_layout` | Create shape/stride mapping |
| `cute.make_tensor` | Bind engine/pointer to a layout |
| `cute.composition` | Compose mappings |
| `cute.complement` | Build non-overlapping complementary layout |
| `cute.logical_divide` | Factor a layout by a tile while preserving modes |
| `cute.tiled_divide` | Divide into repeated tile coordinates |
| `cute.zipped_divide` | Group tile and rest modes |
| `cute.tile_to_shape` | Repeat a layout atom over a target shape |
| `cute.local_tile` | Select a problem tile for a CTA/work unit |
| `cute.dice` | Project selected modes |
| `cute.filter_zeros` | Remove zero-stride/broadcast modes for copy setup |
| `cute.recast_layout` | Reinterpret layout for a different element width |
| `cute.make_fragment_like` | Create a compatible register fragment |

Verify preconditions before applying algebra:

- tile divides or correctly predicates the target
- composition codomain matches the next mapping domain
- recast preserves byte span and alignment
- removed zero-stride modes truly represent replicated data

## Tiling and partitioning

Tiling selects a logical block. Partitioning assigns pieces of that block to
threads, warps, CTAs, or descriptors.

Typical sequence:

```text
global tensor
  -> local_tile by CTA/work coordinate
  -> tiled-copy thread slice
  -> partition_S / partition_D
  -> staged shared tensor
  -> tiled-MMA fragment/descriptor
```

Copy partition:

```python
thread_copy = tiled_copy.get_slice(thread_idx)
thread_src = thread_copy.partition_S(source)
thread_dst = thread_copy.partition_D(destination)
```

MMA partition/fragment APIs include:

```python
tiled_mma.partition_shape_A(...)
tiled_mma.partition_shape_B(...)
tiled_mma.partition_shape_C(...)
tiled_mma.make_fragment_A(...)
tiled_mma.make_fragment_B(...)
tiled_mma.make_fragment_C(...)
```

For tcgen05, A/B SMEM fragments are descriptor tensors rather than ordinary
per-thread register fragments. The C fragment is a layout over TMEM.

## Shared-memory swizzles

A swizzle XOR-permutes address bits to reduce bank conflicts while preserving a
logical tensor view. `ComposedLayout` commonly represents:

```text
swizzle composed with offset composed with outer layout
```

Do not invent a shared layout from a conventional row-major array. The TMA
descriptor and MMA descriptor must agree with the same swizzled layout.

Blackwell helpers select among common contiguous swizzle widths:

- 128-byte span
- 64-byte span
- 32-byte span
- 16-byte interleave

The selected span must divide the operand's contiguous major mode in bits.
Staged layouts append a pipeline mode after the MMA-compatible operand modes.

When allocating:

```python
smem.allocate_tensor(
    element_type=dtype,
    layout=composed.outer,
    byte_alignment=128,
    swizzle=composed.inner,
)
```

use the exact outer layout and swizzle produced during host/JIT setup.

## GEMM tensor conventions

This project follows the common CuTe Blackwell convention:

```text
A: (M,K,L)
B: (N,K,L) physically
C: (M,N,L)
```

Mathematically the operation is still:

```text
C[M,N,L] = A[M,K,L] @ B[K,N,L]
```

The physical `B[N,K,L]` view lets its major mode and descriptor express the
mathematical transpose without materializing another matrix.

Always state:

- logical mathematical shape
- physical tensor shape
- stride tuple
- major mode (`K` or `MN`)
- batch mode

Scale-factor tensors have logical shapes but specialized physical layouts:

```text
SFA logical: (M, ceil_div(K, sf_vec_size), L)
SFB logical: (N, ceil_div(K, sf_vec_size), L)
```

Use block-scaled layout helpers to create their physical swizzle and packing.

## Alignment and divisibility

Alignment is a property of:

- base pointer
- tensor stride/layout
- tile coordinate
- vector width
- TMA box

Checking only the allocation address is insufficient. Every tile origin must
preserve the alignment promised to the compiler.

For baseline dense FP8 Blackwell GEMM, require at least 16-byte alignment in the
contiguous A/B/C dimensions; FP8 contiguous element counts are therefore
multiples of 16.

Use:

- `from_dlpack(..., assumed_align=16)`
- pointer `.align(16)` only when the address is genuinely aligned
- dynamic shape divisibility constraints
- `cute.assume(value, divisor)` only for a true runtime invariant

False alignment assumptions can produce incorrect code rather than a friendly
error.

## Debugging layouts

Inside JIT tracing:

```python
print(tensor.layout)
print(tensor.shape)
print(tensor.stride)
print(cute.size(tensor))
print(cute.cosize(tensor.layout))
```

These Python prints occur at compile time. For runtime coordinates or values,
use `cute.printf` temporarily.

When a layout fails:

1. Print logical and physical input layouts.
2. Print the CTA tile returned by `local_tile`.
3. Print copy partitions.
4. Print MMA partition shapes and fragment layouts.
5. Check that pipeline stage is the final intended mode.
6. Check byte spans and alignment.
7. Check that no flattened mode destroyed descriptor hierarchy.

## Checklist

- [ ] Logical and physical shapes are both documented.
- [ ] Shape/stride mapping matches the intended memory order.
- [ ] Static versus dynamic modes are deliberate.
- [ ] DLPack leading dimension and alignment are explicit.
- [ ] CTA tile covers or predicates the required problem.
- [ ] Copy source/destination partitions are congruent.
- [ ] MMA fragment layouts come from the selected tiled MMA.
- [ ] SMEM swizzle is shared by allocation and descriptors.
- [ ] Pipeline stage mode is indexed consistently.
- [ ] Pointer, stride, tile origin, and vector width preserve alignment.
