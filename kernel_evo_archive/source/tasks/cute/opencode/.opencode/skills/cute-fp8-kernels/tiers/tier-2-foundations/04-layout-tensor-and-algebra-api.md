# Layout, tensor-view, and algebra API

A CuTe layout maps a logical coordinate to a linear coordinate. It is not a
memory allocation. A tensor combines that mapping with an engine such as a
pointer.

## Shape, stride, and rank

For flat shape `(d0, d1)` and stride `(s0, s1)`:

```text
offset(i0, i1) = i0 * s0 + i1 * s1
```

Shapes and strides may be hierarchical tuples. Hierarchy carries semantic
structure such as tile, thread, value, pipeline-stage, or instruction modes.
Two layouts with equal flattened extents can still express different
ownership.

## `make_layout`

```text
cute.make_layout(shape, *, stride=None) -> Layout
```

| Parameter | Meaning |
| --- | --- |
| `shape` | integer or nested integer tuple describing logical extents |
| `stride` | optional integer or nested tuple describing coordinate weights |

When `stride` is omitted, CuTe constructs a compact layout following its
left-major convention. State stride explicitly whenever physical order is part
of the external tensor contract.

Layout properties:

```text
layout.shape
layout.stride
layout.max_alignment
```

## Tensor construction and indexing

```text
cute.make_tensor(engine, layout) -> Tensor
```

The engine is commonly a pointer or fragment storage. The layout codomain
addresses the engine.

```python
value = tensor[coordinate]
tensor[coordinate] = value
subview = tensor[(None, fixed_coordinate)]
```

`None` retains a mode instead of selecting one coordinate from it. The returned
object may be a scalar value or another tensor view depending on the retained
modes.

## Extent and structure queries

| API | Result |
| --- | --- |
| `cute.rank(object)` | number of top-level modes |
| `cute.depth(object)` | hierarchy nesting depth |
| `cute.size(object, mode=None)` | logical element count, optionally for selected modes |
| `cute.cosize(layout)` | minimum codomain span addressed by a layout |
| `cute.size_in_bytes(dtype, layout)` | byte storage needed for a typed layout |
| `cute.is_static(object)` | whether structure is known at compile time |
| `cute.is_congruent(a, b)` | exact structural congruence |
| `cute.is_weakly_congruent(a, b)` | weaker compatible structure test |
| `cute.pretty_str(object)` | readable static representation |

`size` and `cosize` answer different questions. A strided or swizzled layout
can address a larger codomain than its logical number of elements.

## Coordinate conversion

```text
cute.crd2idx(coordinate, layout) -> index
cute.idx2crd(index, shape) -> coordinate
```

`crd2idx` applies the layout mapping. `idx2crd` reconstructs a coordinate under
the supplied shape/layout convention. These functions reason about logical
coordinates; they do not validate that a resulting pointer access lies within
an allocation.

## Slicing and selecting modes

```text
cute.slice_(object, coordinate) -> sliced object
cute.select(object, mode) -> selected modes
cute.front(object) -> first leaf or mode
cute.get_leaves(object) -> flattened leaves
cute.group_modes(object, begin, end=None) -> regrouped object
cute.append(object, value, up_to_rank=None) -> extended object
cute.prepend(object, value, up_to_rank=None) -> extended object
```

The same structural operation may accept a shape, stride, layout, coordinate,
or tensor. The result preserves the kind of the input where the API defines
that transformation.

`group_modes` changes hierarchy but not storage. Regroup only when the consumer
expects the resulting mode structure.

## Layout transformations

Representative public operations:

| API | Purpose |
| --- | --- |
| `cute.coalesce(layout, target_profile=None)` | combine compatible adjacent modes |
| `cute.composition(outer, inner)` | compose coordinate mappings |
| `cute.complement(layout, codomain)` | construct uncovered codomain mapping |
| `cute.logical_divide(object, tiler)` | divide into tile and rest modes |
| `cute.zipped_divide(object, tiler)` | divide and group tile/rest modes |
| `cute.tiled_divide(object, tiler)` | divide by a hierarchical tiler |
| `cute.logical_product(a, b)` | build a logical product layout |
| `cute.blocked_product(a, b)` | build a blocked product |
| `cute.raked_product(a, b)` | build a raked product |

These operations transform coordinate structure. They do not copy data.
Composition order matters: `composition(a, b)` applies the inner mapping and
then the outer mapping.

## Tiling

```text
cute.local_tile(
    tensor,
    tiler,
    coordinate,
    proj=None,
) -> Tensor
```

| Parameter | Meaning |
| --- | --- |
| `tensor` | source tensor view |
| `tiler` | tile shape or hierarchical tiler |
| `coordinate` | coordinate selecting a tile in the quotient modes |
| `proj` | optional mode projection |

The result is a view. It does not allocate or transfer storage.

```text
cute.local_partition(
    target,
    tiler,
    index,
    proj=1,
) -> Tensor
```

`local_partition` assigns a tiler's partition selected by `index`. It is a
general layout operation; it is not interchangeable with copy- or MMA-specific
partition methods.

## Layout comparison questions

Before passing a layout or tensor to another API, establish:

1. rank and hierarchy;
2. logical shape;
3. stride-one mode;
4. codomain span;
5. element type and memory space;
6. static versus runtime modes;
7. ownership expected by the consumer.

Matching only the printed shape is insufficient.

## Swizzles

`cute.make_swizzle(b, m, s)` constructs a bitwise address permutation. `b` is
the number of swizzle bits, `m` is the number of unchanged low bits, and `s`
is the shift distance used by the XOR mapping.

Swizzles are normally composed with an existing layout. They change address
mapping, not logical shape. The three parameters must follow from the memory
operation and bank-layout contract; they are not general tuning knobs.

## Static and dynamic layouts

A static layout contains Python-known extents and strides and participates in
specialization. A dynamic layout contains runtime integer leaves with known
constraints. Rank and tuple nesting remain static even when selected extents
are dynamic.

Dynamic layout values can participate in generated arithmetic but cannot
change Python structure during execution. Preserve static hierarchy and make
only genuinely variable leaves dynamic.

## Deriving an operand's major mode

The major mode is read from the tensor, not chosen from a constant:

```text
utils.LayoutEnum.from_tensor(tensor).mma_major_mode()
```

There is no `LayoutEnum.RowMajor` and no `LayoutEnum.MMA_MAJOR_A`, and a layout
object has no `leading_dim` attribute.

## Integer arithmetic and composed layouts

Ceiling division lives on `cute`, not on `cutlass`:

```text
cute.ceil_div(value, divisor)
```

It accepts a tuple as well as a scalar, which is how a launch grid is derived
from a shape and a tile shape. `cutlass.ceil_div` does not exist; reaching for
it is a common carry-over from the C++ interface.

A layout produced by composing a swizzle with an underlying layout has its own
type, and appears in annotations where a shared-memory layout is passed:

```text
cute.ComposedLayout
```
