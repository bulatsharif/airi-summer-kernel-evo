# Types, pointers, tensors, and runtime conversion

CuTe types describe both values and compile-time structure. A correct program
must distinguish Python-static values, runtime scalar proxies, layouts,
pointers, and tensors.

## Scalar type families

Representative public scalar types live under `cutlass`:

| Family | Representative types |
| --- | --- |
| Boolean | `Boolean` |
| Signed integer | `Int8`, `Int16`, `Int32`, `Int64` |
| Unsigned integer | `Uint8`, `Uint16`, `Uint32`, `Uint64` |
| Binary floating point | `Float16`, `BFloat16`, `TFloat32`, `Float32`, `Float64` |
| FP8 | `Float8E4M3FN`, `Float8E5M2` |
| Narrow types | release-supported FP6/FP4 types |
| Compile-time value | `Constexpr` |

A scalar type object acts as a conversion constructor:

```python
runtime_i = cutlass.Int32(value)
runtime_x = cutlass.Float32(value)
```

The conversion becomes generated code when `value` is a runtime proxy and is
folded when both type and value are static. Use the public type constructor;
there is no general requirement for a `cute.cast` or `cute.convert` helper.

## Static and runtime integers

Python `int` is a static metaprogramming value. CuTe integer proxies carry a
runtime IR value and may also carry divisibility information used by layout
algebra.

Runtime integers support arithmetic such as addition, subtraction,
multiplication, floor division, and remainder. Divisibility constraints
propagate through compatible arithmetic. They cannot be used wherever the
Python interpreter itself requires an integer, for example:

- indexing a Python tuple or list;
- selecting a Python class;
- setting a static array length;
- iterating `cutlass.range_constexpr`;
- changing the rank or nesting of a static layout.

Use `cute.is_static(value)` to distinguish a static value and
`cute.get_divisibility(value)` to query known integer divisibility when
available.

## `cute.Pointer`

A pointer combines an element type, address space, alignment, and address.

Important properties:

| Property | Meaning |
| --- | --- |
| `dtype` | pointed-to scalar type |
| `memspace` | global, shared, register, tensor, or generic address space |
| `alignment` / `max_alignment` | known byte alignment |

Common operations:

```text
pointer + element_offset
pointer - element_offset
pointer.align(minimum_byte_alignment)
pointer.toint()
```

Pointer offsets count elements of the pointer's dtype, not bytes. Alignment is
an assumption and a contract; declaring stronger alignment does not realign the
underlying allocation.

## `make_ptr`

```text
cutlass.cute.runtime.make_ptr(
    dtype,
    value,
    mem_space=cute.AddressSpace.generic,
    assumed_align=None,
) -> cute.Pointer
```

| Parameter | Meaning |
| --- | --- |
| `dtype` | scalar type of each pointed-to element |
| `value` | integer address or compatible pointer object |
| `mem_space` | address-space annotation |
| `assumed_align` | assumed byte alignment |

Use `nullptr(dtype, mem_space=..., assumed_align=...)` only when a null pointer
is part of a compilation or optional-argument contract.

## `cute.Tensor`

A tensor is an engine, usually a pointer, composed with a layout:

```text
tensor(coordinate) = engine[layout(coordinate)]
```

Core properties include:

| Property | Meaning |
| --- | --- |
| `element_type` | scalar element type |
| `layout` | coordinate-to-address mapping |
| `shape` | logical extents |
| `stride` | logical strides |
| `memspace` | storage address space |
| `iterator` | underlying engine |

Tensor indexing may use a scalar coordinate, a coordinate tuple, or a tuple
containing `None` to retain modes. The legality of a load or store also depends
on element type, address space, alignment, and whether the coordinate is in
bounds.

## Constructing a tensor from a pointer

```text
cute.make_tensor(pointer, layout) -> cute.Tensor
```

`pointer` and `layout` must describe the same allocation. The layout's codomain
must fit in the backing storage. A layout alone has no storage and cannot be
loaded, stored, copied, or partitioned as a tensor.

## DLPack conversion

```text
cutlass.cute.runtime.from_dlpack(
    tensor_dlpack,
    assumed_align=None,
    use_32bit_stride=False,
    *,
    enable_tvm_ffi=False,
    force_tf32=False,
) -> cute.Tensor
```

| Parameter | Meaning |
| --- | --- |
| `tensor_dlpack` | object implementing the DLPack producer protocol |
| `assumed_align` | byte alignment promised by the caller; defaults to element size |
| `use_32bit_stride` | use 32-bit dynamic strides when the addressable span permits it |
| `enable_tvm_ffi` | produce a TVM-FFI-compatible runtime argument |
| `force_tf32` | represent Float32 storage through a TF32 compute type where supported |

Conversion is a zero-copy view. The original storage owner must remain alive,
and producer and consumer must obey compatible stream ordering.

## Dynamic runtime tensor layouts

Runtime DLPack tensors initially carry concrete shape and stride metadata.
Methods can replace selected facts by dynamic constraints:

```text
tensor.mark_layout_dynamic(leading_dim=None) -> tensor
tensor.mark_compact_shape_dynamic(
    mode,
    stride_order=None,
    divisibility=1,
) -> tensor
```

`mark_layout_dynamic` keeps the identified leading dimension at stride one and
makes other supported layout facts dynamic. If `leading_dim` is omitted, the
method attempts to infer a unique stride-one dimension. The current API is
limited to flat layouts.

`mark_compact_shape_dynamic` marks one compact shape mode dynamic and
propagates the constraint into strides.

| Parameter | Meaning |
| --- | --- |
| `mode` | shape mode to make dynamic |
| `stride_order` | outermost-to-innermost mode ordering |
| `divisibility` | promised divisibility of the runtime extent |

Omitting `stride_order` is valid only when the compact ordering is
unambiguous.

## Shared storage structures

`@cute.struct` defines compile-time shared-storage records:

```text
cute.struct.MemRange[dtype, count]
cute.struct.Align[field_type, byte_alignment]
```

`MemRange` describes a typed fixed-size range. Its data object provides
`data_ptr()` and `get_tensor(layout, swizzle=None, dtype=None)`.
`Align` assigns explicit alignment to a scalar, range, or nested struct.

The resulting struct class provides static size and alignment information.
Allocation belongs to an allocator such as `cutlass.utils.SmemAllocator`; the
struct decorator itself is not an allocator.

## Memory-space discipline

| Space | Typical owner and lifetime |
| --- | --- |
| Global memory | application-visible allocation, survives the kernel |
| Shared memory | one CTA or cluster, allocated for a launch |
| Register memory | one thread's generated values |
| Tensor memory | Blackwell tensor-core storage with explicit protocol |
| Generic | pointer whose concrete space is not encoded strongly |

An equal dtype and shape do not make pointers from different address spaces
interchangeable. Copy and MMA operations constrain the source and destination
spaces they accept.

## Elementwise math intrinsics

Reciprocal square root is a `cute` intrinsic rather than a Python or NumPy call:

```text
cute.rsqrt(x)
```

Use it where a normalization needs `1 / sqrt(...)`; computing it as `x ** -0.5`
inside a kernel is not equivalent and may not lower.
