# Copy, MMA, memory, and synchronization API

CuTe expresses data movement and matrix operations through atoms and tiled
operations. An atom describes one hardware or logical operation. A tiled
operation composes atoms with participant and value layouts.

## Copy object model

```text
CopyOp -> CopyAtom -> TiledCopy -> ThrCopy -> tensor partitions
```

- `CopyOp` identifies an instruction or logical copy operation.
- `CopyAtom` combines that operation with an element/value layout.
- `TiledCopy` maps the copy across a thread layout.
- `ThrCopy` is the slice owned by one participant.

Representative constructors:

```text
cute.make_copy_atom(copy_operation, copy_internal_type, **operation_parameters)
cute.make_tiled_copy(copy_atom, thread_value_layout, tiler)
cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)
```

| Parameter | Meaning |
| --- | --- |
| `copy_operation` | instruction or logical copy operation |
| `copy_internal_type` | scalar type used to express the atom's source/destination layouts |
| operation parameters | operation-specific options such as `num_bits_per_copy` for a universal copy |
| `thread_value_layout` | combined participant/value mapping |
| `tiler` | logical tile covered by the tiled copy |
| `thread_layout` | mapping from participant coordinate to participant ID |
| `value_layout` | values owned by each participant |

## Copy partitioning

```text
tiled_copy.get_slice(thread_index) -> ThrCopy
thread_copy.partition_S(source_tensor) -> Tensor
thread_copy.partition_D(destination_tensor) -> Tensor
```

`partition_S` and `partition_D` are methods on the participant slice. The
source and destination must be tensors, not layouts. Their per-participant
value counts and ordering must match the copy atom.

## Issuing a copy

```text
cute.copy(
    atom_or_tiled_copy,
    source,
    destination,
    *,
    pred=None,
    unroll_factor=None,
    **operation_parameters,
) -> None
```

| Parameter | Meaning |
| --- | --- |
| `atom_or_tiled_copy` | copy operation defining transfer semantics |
| `source` | compatible source tensor or partition |
| `destination` | compatible destination tensor or partition |
| `pred` | optional predicate tensor/value for guarded transfers |
| `unroll_factor` | optional generated-loop unroll control |
| operation parameters | instruction-specific barriers, masks, or cache controls |

`cute.autovec_copy(source, destination)` requests a compiler-selected vectorized
copy when the source/destination layout and alignment permit one. It does not
replace an instruction-specific asynchronous protocol.

## Register fragments

```text
cute.make_rmem_tensor(layout_or_shape, dtype) -> Tensor
cute.make_rmem_tensor_like(source, dtype=None) -> Tensor
cute.full(shape, fill_value, dtype) -> TensorSSA
cute.full_like(source, fill_value, dtype=None) -> TensorSSA
```

`make_rmem_tensor` creates thread-local fragment storage. `full` creates an SSA
tensor value and requires shape, fill value, and dtype. A tensor SSA value and
a pointer-backed `cute.Tensor` are different object kinds.

## MMA object model

```text
MmaOp -> MmaAtom -> TiledMma -> ThrMma -> operand/accumulator partitions
```

- `MmaOp` identifies an instruction family and its operand contract.
- `MmaAtom` wraps the instruction with layouts.
- `TiledMma` maps it over an instruction or CTA tile.
- `ThrMma` is the participant-specific slice.

Representative construction:

```text
cute.make_tiled_mma(
    mma_operation_or_atom,
    atom_layout_mnk=(1, 1, 1),
    permutation_mnk=None,
) -> TiledMma
```

The first argument may be an MMA operation or an already constructed atom.
The `atom_layout_mnk` maps atoms over logical M/N/K modes.
`permutation_mnk` optionally changes the operand/accumulator mode order.

Common `TiledMma` surface:

```text
tiled_mma.get_slice(participant_index) -> ThrMma
tiled_mma.get_tile_size(mode_index) -> shape
tiled_mma.partition_shape_A(shape_mk)
tiled_mma.partition_shape_B(shape_nk)
tiled_mma.partition_shape_C(shape_mn)
tiled_mma.make_fragment_A(tensor_or_partition)
tiled_mma.make_fragment_B(tensor_or_partition)
tiled_mma.make_fragment_C(tensor_or_partition)
tiled_mma.shape_mnk
tiled_mma.size
```

Common `ThrMma` surface:

```text
thread_mma.partition_A(tensor_mk) -> Tensor
thread_mma.partition_B(tensor_nk) -> Tensor
thread_mma.partition_C(tensor_mn) -> Tensor
```

The operand suffix expresses the logical role. It is not a generic source or
destination marker.

## Issuing an MMA

```text
cute.gemm(atom, destination, operand_a, operand_b, source_accumulator) -> None
```

The five roles are:

1. MMA atom or tiled MMA;
2. destination accumulator;
3. A fragment;
4. B fragment;
5. prior/source accumulator.

The selected instruction determines whether the destination aliases the source
accumulator, whether the operation overwrites or accumulates, and where the
fragments reside.

## Shared-memory allocation

High-level allocator:

```text
cutlass.utils.SmemAllocator.allocate_tensor(
    element_type,
    layout,
    byte_alignment=1,
    swizzle=None,
) -> Tensor
```

| Parameter | Meaning |
| --- | --- |
| `element_type` | scalar type stored in shared memory |
| `layout` | exact shared-memory layout |
| `byte_alignment` | required start alignment |
| `swizzle` | optional address swizzle when not already composed into layout |

Low-level path:

```text
cute.arch.alloc_smem(dtype, count, alignment) -> Pointer
cute.make_tensor(pointer, layout) -> Tensor
```

`count` is an element count; derive it from the exact layout codomain. The
pointer, layout, and declared alignment must agree.

## Device index functions

All dimensional index functions return three values and take no axis argument:

```text
cute.arch.thread_idx() -> (x, y, z)
cute.arch.block_dim() -> (x, y, z)
cute.arch.block_idx() -> (x, y, z)
cute.arch.grid_dim() -> (x, y, z)
cute.arch.cluster_idx() -> (x, y, z)
cute.arch.cluster_dim() -> (x, y, z)
```

Additional scalar queries:

```text
cute.arch.lane_idx()
cute.arch.warp_idx()
cute.arch.physical_warp_id()
cute.arch.block_idx_in_cluster()
cute.arch.dynamic_smem_size()
```

## Synchronization functions

| API | Scope and role |
| --- | --- |
| `cute.arch.sync_warp(mask=None)` | synchronize participating warp lanes |
| `cute.arch.sync_threads()` | synchronize threads in a CTA |
| `cute.arch.barrier(barrier_id, number_of_threads)` | named CTA barrier |
| `cute.arch.barrier_arrive(barrier_id, number_of_threads)` | split-phase arrival |
| `cute.arch.elect_one()` | context selecting one lane in each warp |
| `cute.arch.make_warp_uniform(value)` | assert/hint that an integer is warp-uniform |

Barrier IDs, participant counts, and scope are semantic parameters, not
arbitrary constants. Every required participant must execute compatible
barrier control flow.

Transaction-barrier and Blackwell-specific functions are described in Tier II.

## TMA atom and pipeline constructors

```text
cute.nvgpu.make_tiled_tma_atom_A(...)
cute.nvgpu.make_tiled_tma_atom_B(...)

pipeline.PipelineTmaUmma.create(...)
pipeline.PipelineUmmaAsync.create(...)
pipeline.CooperativeGroup(...)
pipeline.Agent.Thread
pipeline.NamedBarrier
```

## Cross-lane exchange

A warp-level butterfly shuffle exchanges a value between lanes at a given
offset:

```text
cute.arch.shuffle_sync_bfly(value, offset=<int>)
```

It returns the partner lane's value; combining the returned value with the local
one, over a descending sequence of offsets, is the standard way to reduce across
a warp without shared memory. The offsets and the combining operation are yours
to choose for the reduction you need.
