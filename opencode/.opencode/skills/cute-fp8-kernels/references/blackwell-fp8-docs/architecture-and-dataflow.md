# tcgen05 architecture and dense GEMM dataflow

This page condenses NVIDIA's tcgen05 programming guide for the common Blackwell
dense GEMM case: A and B begin in global memory, are staged in shared memory,
and an FP32 accumulator lives in tensor memory.

## Why the object graph is different from ordinary CUDA

Blackwell tcgen05 instructions do not consume arbitrary per-thread register
fragments in the usual CUDA sense:

- A is read from SMEM or, for selected operations, TMEM.
- B is read from SMEM.
- C/D, the accumulator input/output, always lives in TMEM.
- one thread issues a tcgen05 MMA instruction on behalf of the participating
  CTA or CTA pair.
- the completed accumulator is loaded from TMEM into registers before it can be
  transformed and written to GMEM.

Therefore `cute.gemm(atom, d, a, b, c)` is the middle of a protocol, not a
standalone matrix-multiplication launch.

## Memory-space roles

| Space | Objects | Role |
|---|---|---|
| GMEM | input/output `cute.Tensor` values | Persistent matrices passed by the evaluator |
| SMEM | tensors allocated by `SmemAllocator` inside `@cute.kernel` | TMA destination and MMA A/B source |
| TMEM | tensor backed by `TmemAllocator.retrieve_ptr` | tcgen05 FP32 accumulator |
| RMEM | `cute.make_rmem_tensor(...)` | Per-thread epilogue values loaded from TMEM |

A layout describes an index mapping. It is not memory. `cute.make_tensor`
requires a real pointer followed by a layout.

## Naming map used by NVIDIA examples

```text
mA, mB, mC     full GMEM tensors
gA, gB, gC     CTA-local GMEM tiles from local_tile
tCgA/B/C       those tiles partitioned for the MMA/CTA
sA, sB         physically allocated SMEM tensors
tCrA, tCrB     MMA-readable descriptors/fragments made from sA/sB
tCtAcc          TMEM-backed accumulator tensor
tAsA/tBsB      TMA-partitioned SMEM views
tAgA/tBgB      TMA-partitioned GMEM views
tTR_*           TMEM-to-register epilogue partitions
```

Keeping these categories distinct is more important than keeping the exact
variable names.

## Construction order

### 1. Build one TiledMma

Derive A/B major modes from the input tensors, then use the installed trivial
MMA constructor in `server-api-deltas.md`. Its output defines compatible SMEM
layouts, partition shapes, and fragments.

### 2. Construct staged SMEM layouts on the host side

`make_smem_layout_a/b(tiled_mma, mma_tiler_mnk, dtype, stages)` describes the
whole staged buffer. TMA descriptor construction uses a one-stage selection;
the kernel receives the full staged layout and physically allocates it.

### 3. Tile the global tensors for one output CTA

For block coordinates `(block_m, block_n)`:

```python
gA = cute.local_tile(mA, mma_tiler_mnk, (block_m, None, None),
                     proj=(1, None, 1))
gB = cute.local_tile(mB, mma_tiler_mnk, (None, block_n, None),
                     proj=(None, 1, 1))
gC = cute.local_tile(mC, mma_tiler_mnk, (block_m, block_n, None),
                     proj=(1, 1, None))
```

The remaining A/B mode enumerates K tiles.

### 4. Partition actual tensors

For a single-CTA collective MMA, obtain the collective slice and partition the
GMEM tensor views:

```python
thr_mma = tiled_mma.get_slice(0)
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)
tCgC = thr_mma.partition_C(gC)
```

`partition_A/B/C` accept tensors. They do not accept a bare layout and are not
objects on which to call `.load` or `.store`.

### 5. Make MMA operands from allocated SMEM

The same SMEM allocations written by TMA become the MMA descriptors:

```python
tCrA = tiled_mma.make_fragment_A(sA)
tCrB = tiled_mma.make_fragment_B(sB)
```

Do not make these fragments from GMEM tiles or synthetic tensors.

### 6. Bind the accumulator layout to TMEM

Create the C partition shape, create its fragment/layout, allocate TMEM, and
rebuild the tensor with `(tmem_ptr, fragment.layout)`. See
`tmem-and-epilogue.md`.

## K-loop state

The TMA pipeline advances in units of `BK` tiles. Within one loaded stage,
`cute.size(tCrA, mode=[2])` gives the number of hardware K blocks. The first
MMA overwrites/initializes the accumulator; subsequent MMAs set
`tcgen05.Field.ACCUMULATE` to true.

The A/B stage cannot be released until all MMA instructions that read it have
been issued. The accumulator cannot be read by the epilogue until the async
UMMA completion pipeline signals it is full.

## FP8 is storage and multiplication format, not the accumulator format

For this task family, A and B are E4M3FN storage. The dense tcgen05 operation
accumulates into Float32 in TMEM. Any task-level dequantization scale is applied
to the FP32 result before or during the epilogue; it is not represented by
inventing extra FP8 tensor fragments.
