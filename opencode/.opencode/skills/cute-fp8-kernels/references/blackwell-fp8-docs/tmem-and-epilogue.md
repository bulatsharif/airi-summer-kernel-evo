# TMEM accumulator and output epilogue

tcgen05 keeps its accumulator in Tensor Memory. A correct dense kernel must
allocate TMEM, bind the accumulator layout to that pointer, wait for async MMA
completion, load results into registers, and only then write GMEM.

## Allocate TMEM inside the kernel

```python
tmem_alloc_barrier = pipeline.NamedBarrier(
    barrier_id=1,
    num_threads=128,
)
tmem = utils.TmemAllocator(
    storage.tmem_holding_buf.ptr,
    barrier_for_retrieve=tmem_alloc_barrier,
)
tmem.allocate(512)
```

The column count must be a power of two and a multiple of 32. The neutral
single-tile bridge used 512 columns for simplicity.

Before any warp consumes the allocated address:

```python
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
```

## Bind the MMA C layout to the TMEM pointer

```python
acc_fragment = tiled_mma.make_fragment_C(
    tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
)
tCtAcc = cute.make_tensor(tmem_ptr, acc_fragment.layout)
```

The first argument is a pointer and the second is a layout. Do not call
`cute.make_tensor(cute.make_layout(...), dtype)`.

The resulting tensor is TMEM-backed. It is not a normal register tensor and
does not support direct `.fill()`.

## Issue the MMA

For one loaded A/B stage and one hardware K block:

```python
coord = (None, None, k_block, ab_full.index)
cute.gemm(
    tiled_mma,
    tCtAcc,
    tCrA[coord],
    tCrB[coord],
    tCtAcc,
)
tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

The first operation initializes/overwrites the accumulator. Set ACCUMULATE for
subsequent operations, not before the first MMA of a new output tile.

## Partition the output destination

The GMEM output path is independent of the TMEM accumulator path:

```text
output tensor -> local_tile -> thr_mma.partition_C -> epilogue partition_D
```

`partition_C` returns a tensor view. It is not a store function.

## TMEM to RMEM to GMEM

For the verified `(128,128)` Float32 output tile, the shared B300 bridge used:

- two output subtiles;
- `tcgen05.Ld32x32bOp(tcgen05.Repetition.x64)`;
- `tcgen05.make_tmem_copy`;
- a per-thread Float32 RMEM tensor;
- `cute.autovec_copy` from RMEM to the partitioned GMEM destination.

The structural sequence is:

```python
tmem_atom = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    cutlass.Float32,
)
tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tmem_epi[None, 0])
thread_copy = tmem_copy.get_slice(thread_idx)
tmem_source = thread_copy.partition_S(tmem_epi)
gmem_destination = thread_copy.partition_D(gmem_epi)
registers = cute.make_rmem_tensor(
    gmem_destination[None, None, 0].shape,
    cutlass.Float32,
)
```

After the accumulator consumer wait:

```python
cute.copy(tmem_copy, tmem_source[None, None, tile_idx], registers)
cute.autovec_copy(registers, gmem_destination[None, None, tile_idx])
```

For a fused epilogue, transform `registers.load()` and store the transformed
value in an appropriately typed RMEM tensor before the GMEM copy. A
correctness-first two-kernel solution may instead write unscaled FP32 GEMM
results and apply scale + BiasAdd + ReLU in a second CuTe kernel.

## Cleanup order

Before epilogue reads, relinquish the allocation permit and wait for the async
accumulator pipeline. After all threads finish the epilogue:

```python
tmem.relinquish_alloc_permit()
acc_full = acc_consumer.wait_and_advance()
# TMEM -> RMEM -> GMEM
acc_full.release()
pipeline.sync(barrier_id=1)
tmem.free(tmem_ptr)
```

Do not free TMEM before the CTA-wide epilogue is complete.
