# Layout, thread/value partitioning, copy, and predication

`cute.Layout` is a coordinate-to-index map. For shape `(4, 8)` and stride `(8, 1)`, coordinate `(2, 3)` maps to `19`. Confirm unfamiliar mappings with:

```bash
kernel-evo cute probe-layout --shape 4,8 --stride 8,1 --coord 2,3
```

When composition or division is the question, compile the same probe through the installed DSL and inspect the hierarchical result:

```bash
kernel-evo cute probe-layout --dsl --shape 4,8 --stride 8,1 --coord 2,3 --tile 2,4
```

The default probe is fast deterministic arithmetic. `--dsl` is slower because it traces with the pinned package; use it to verify an unfamiliar CuTe transformation, not for every edit.

Exact 4.2.x APIs:

```python
layout = cute.make_layout((4, 8), stride=(8, 1))
row_major = cute.make_ordered_layout((4, 8), order=(1, 0))
tiler_mn, layout_tv = cute.make_layout_tv(thr_layout, val_layout)
tiled = cute.zipped_divide(tensor, tiler_mn)
copy_atom = cute.make_copy_atom(
    cute.nvgpu.CopyUniversalOp(),
    tensor.element_type,
    num_bits_per_copy=128,  # only after proving per-thread alignment
)
tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
thread_slice = tiled_copy.get_slice(thread_idx)
source = thread_slice.partition_S(tile)
destination = thread_slice.partition_D(tile)
```

Invariants:

- `thr_layout` must be compact. Its codomain is the participating thread IDs.
- `val_layout` describes values owned per thread; it is not a second thread layout.
- Use `partition_S` for a copy source and `partition_D` for its destination. Equal shapes do not make the roles interchangeable.
- A 128-bit copy needs 16-byte pointer alignment and a contiguous vector of `128 // dtype.width` elements.
- A wide TV layout is not proof of a wide instruction. In 4.2.x, pass `num_bits_per_copy=128` to make the contract explicit, then require `LDG/STG .128` in retained SASS. The predicated residue example demonstrates that per-element predicates can scalarize an otherwise wide ownership layout.
- Runtime scalar indexing such as `tensor[thread_idx]` is not a general global-memory access idiom in the DSL. Partition a tensor and copy through fragments.
- `make_fragment_like` preserves the partition shape and element type. Allocate a separate Float32 fragment when widening compute.

Boundary pattern:

```python
identity = cute.make_identity_tensor(output.shape)
coordinates = cute.zipped_divide(identity, tiler=tiler_mn)
thread_coordinates = thread_slice.partition_S(coordinates[((None, None), block_idx)])
predicate = cute.make_fragment(thread_coordinates.shape, cutlass.Boolean)
for i in range(cute.size(predicate)):
    predicate[i] = cute.elem_less(thread_coordinates[i], output.shape)
cute.copy(load_atom, thread_source, registers, pred=predicate)
cute.copy(store_atom, registers, thread_destination, pred=predicate)
```

Test exact tile multiples and `tile-1` / `tile+1` dimensions. Alignment and bounds are separate proofs: a predicate does not make a misaligned vector access legal.
