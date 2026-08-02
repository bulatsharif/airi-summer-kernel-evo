# Layout and tensor-view patterns

These fragments illustrate coordinate mappings and view operations. The small
extents are neutral teaching values, not launch or kernel recommendations.

## Flat physical layouts

```python
left_major = cute.make_layout((4, 8))
right_major = cute.make_layout((4, 8), stride=(8, 1))

print(left_major.shape, left_major.stride)
print(right_major.shape, right_major.stride)
```

The layouts have equal logical shape and different address mappings.

## Coordinate-to-index check

```python
layout = cute.make_layout((4, 8), stride=(8, 1))
coordinate = (2, 3)
linear = cute.crd2idx(coordinate, layout)
```

For this static example, `linear` is the sum of each coordinate multiplied by
its stride. Use such checks to verify coordinate order before adding tiling.

## Hierarchical layout

```python
hierarchical = cute.make_layout(
    ((OUTER_MODE, INNER_MODE), OTHER_MODE),
    stride=((OUTER_STRIDE, INNER_STRIDE), OTHER_STRIDE),
)
```

Every uppercase leaf must be derived from a storage contract. Hierarchy is
preserved in `.shape` and `.stride`.

## Pointer plus layout

```python
typed_pointer = make_ptr(
    ELEMENT_TYPE,
    raw_address,
    cute.AddressSpace.gmem,
    assumed_align=ASSUMED_ALIGNMENT,
)
tensor = cute.make_tensor(typed_pointer, physical_layout)
```

`ASSUMED_ALIGNMENT` promises an existing property of the allocation. It does
not change the address.

## Runtime framework view

```python
runtime_tensor = from_dlpack(
    framework_tensor,
    assumed_align=ASSUMED_ALIGNMENT,
    use_32bit_stride=USE_32BIT_STRIDE,
)
```

The conversion is zero-copy. Keep `framework_tensor` alive through completion.

## Dynamic compact mode

```python
runtime_tensor = runtime_tensor.mark_compact_shape_dynamic(
    mode=DYNAMIC_MODE,
    stride_order=PHYSICAL_STRIDE_ORDER,
    divisibility=EXTENT_DIVISIBILITY,
)
```

The mode, ordering, and divisibility are part of the specialization contract.

## Local tile

```python
cta_tile = cute.local_tile(
    whole_tensor,
    (TILE_MODE_0, TILE_MODE_1),
    (tile_coordinate_0, tile_coordinate_1),
)
```

The result is a view of the same storage. The tile coordinate lives in quotient
modes created by division.

## Retaining and selecting modes

```python
all_first_mode_at_second_coordinate = tensor[(None, second_coordinate)]
selected_modes = cute.select(tensor, SELECTED_MODE_PROFILE)
grouped = cute.group_modes(tensor, BEGIN_MODE, END_MODE)
```

`None` retains a mode. `select` chooses modes. `group_modes` changes hierarchy.
None of these operations copies data.

## Logical versus codomain storage

```python
logical_elements = cute.size(layout)
storage_elements = cute.cosize(layout)
storage_bytes = cute.size_in_bytes(ELEMENT_TYPE, layout)
```

Allocate according to `storage_bytes` or the exact allocator contract, not
only `logical_elements`.

## Shared-memory tensor

```python
smem_allocator = utils.SmemAllocator()
smem_tensor = smem_allocator.allocate_tensor(
    ELEMENT_TYPE,
    SHARED_LAYOUT,
    byte_alignment=BYTE_ALIGNMENT,
)
```

The layout, element type, and byte alignment are a coupled contract. This
fragment does not construct `SHARED_LAYOUT`.

## Low-level shared-memory path

```python
element_count = cute.cosize(SHARED_LAYOUT)
smem_pointer = cute.arch.alloc_smem(
    ELEMENT_TYPE,
    element_count,
    alignment=BYTE_ALIGNMENT,
)
smem_tensor = cute.make_tensor(smem_pointer, SHARED_LAYOUT)
```

Use either a compatible allocator or the low-level pointer path; do not
allocate twice for one tensor.

## Participant ownership table

Before indexing a view, record:

```text
whole tensor modes:
CTA tile modes:
participant modes:
retained modes after slicing:
physical stride-one mode:
valid coordinate predicate:
```

This table is intentionally blank because its contents are task- and
implementation-specific.
