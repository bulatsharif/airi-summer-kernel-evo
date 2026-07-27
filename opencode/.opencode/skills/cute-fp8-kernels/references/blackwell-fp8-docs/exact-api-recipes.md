# Exact B300 API recipes

Use these spellings literally. Every positive recipe below is taken from the
numerically verified one-tile bridge in `examples/fp8-mma-one-tile.py` unless
the section says otherwise. Do not translate CUDA built-ins into guessed CuTe
names and do not invent convenience methods after a compiler error.

## Device coordinates

All coordinate helpers live under `cute.arch`, take no arguments, and return an
`(x, y, z)` tuple:

```python
thread_x, thread_y, thread_z = cute.arch.thread_idx()
block_x, block_y, block_z = cute.arch.block_idx()
block_x_size, block_y_size, block_z_size = cute.arch.block_dim()
```

For a one-dimensional elementwise kernel:

```python
thread_x, _, _ = cute.arch.thread_idx()
block_x, _, _ = cute.arch.block_idx()
block_x_size, _, _ = cute.arch.block_dim()
linear = block_x * block_x_size + thread_x
```

Never use `cute.thread_id`, `cute.block_dim`, `cute.arch.thread_id`, or pass an
axis argument such as `cute.arch.thread_idx(0)`.

## Shared struct storage

`cute.struct.MemRange[...]` is an annotation, not a constructor:

```python
@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32
```

Instantiate it only through the allocator inside `@cute.kernel`:

```python
smem = utils.SmemAllocator()
storage = smem.allocate(SharedStorage)
barrier_storage = storage.ab_mbar_ptr.data_ptr()
tmem_holding_pointer = storage.tmem_holding_buf.ptr
```

Do not call `cute.struct.MemRange(...)` and do not define a Python storage class
inside `@cute.jit`.

## TiledMma and ThrMma

Construct the dense FP8 MMA with exactly six positional arguments:

```python
tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a.element_type,
    utils.LayoutEnum.from_tensor(a).mma_major_mode(),
    utils.LayoutEnum.from_tensor(b).mma_major_mode(),
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (128, 128),
)
```

Obtain the single-CTA collective slice with one required index:

```python
thr_mma = tiled_mma.get_slice(0)
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)
tCgC = thr_mma.partition_C(gC)
```

The verified `ThrMma` operations used by this flow are `partition_A`,
`partition_B`, and `partition_C`. There is no verified
`ThrMma.get_thread_slice`, `ThrMma.get_coord`, or zero-argument
`TiledMma.get_slice()` recipe. CTA coordinates come from
`cute.arch.block_idx()`, not from `ThrMma`.

For a matrix larger than one `(128,128)` output tile, expand the neutral example
only by replacing its literal zero tile coordinates with the CTA coordinates:

```python
block_m, block_n, _ = cute.arch.block_idx()
gA = cute.local_tile(
    tma_tensor_a,
    MMA_TILER_MNK,
    (block_m, None, None),
    proj=(1, None, 1),
)
gB = cute.local_tile(
    tma_tensor_b,
    MMA_TILER_MNK,
    (None, block_n, None),
    proj=(None, 1, 1),
)
gC = cute.local_tile(
    output,
    MMA_TILER_MNK,
    (block_m, block_n, None),
    proj=(1, 1, None),
)
```

For static dimensions divisible by 128, launch:

```python
grid=(M // 128, N // 128, 1)
```

The remaining A/B K-tile mode is already handled by the complete example's
`num_k_tiles = cute.size(global_a, mode=[2])` loop. Do not derive K tiles from
`ThrMma`.

## SMEM fragments versus GMEM partitions

Partition actual GMEM tensor tiles through `ThrMma`, but create MMA operand
fragments from the physically allocated SMEM tensors through `TiledMma`:

```python
tCgA = thr_mma.partition_A(gA)
tCgB = thr_mma.partition_B(gB)
tCrA = tiled_mma.make_fragment_A(sA)
tCrB = tiled_mma.make_fragment_B(sB)
acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
tCtAcc = tiled_mma.make_fragment_C(acc_shape)
```

These names describe roles, not additional API objects. Do not call `.load`,
`.store`, `.ptr`, or coordinate methods on the `ThrMma` object.

## TMA descriptors and partitions

The host/JIT factories return one `TmaInfo` object:

```python
tma_a = cute.nvgpu.make_tiled_tma_atom_A(
    sm100_utils.CopyBulkTensorTileG2SOp(),
    a,
    cute.select(smem_layout_a, mode=[0, 1, 2]),
    MMA_TILER_MNK,
    tiled_mma,
)
```

Pass `tma_a.atom` and `tma_a.tma_tensor` to the kernel. Inside the kernel:

```python
tAsA, tAgA = cpasync.tma_partition(
    tma_atom_a,
    0,
    cute.make_layout(1),
    cute.group_modes(sA, 0, 3),
    cute.group_modes(tCgA, 0, 3),
)
```

The first `tma_partition` thread argument is the integer `0` in the verified
single-CTA recipe.

## Pipeline groups

The first `CooperativeGroup` argument is always a `pipeline.Agent` enum:

```python
producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread)
consumer_group=pipeline.CooperativeGroup(
    pipeline.Agent.Thread,
    THREADS_PER_CTA,
)
```

Create pipeline objects with real shared barrier storage and then call
`.make_participants()`:

```python
producer, consumer = pipeline.PipelineTmaUmma.create(
    num_stages=AB_STAGES,
    producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
    tx_count=transaction_bytes,
    barrier_storage=storage.ab_mbar_ptr.data_ptr(),
).make_participants()
```

## MMA issue

Bind the allocated TMEM pointer to the accumulator layout before issuing MMA:

```python
tmem.wait_for_alloc()
tmem_ptr = tmem.retrieve_ptr(cutlass.Float32)
tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc.layout)
```

Issue exactly five operands and set accumulation only after the first issue:

```python
cute.gemm(tiled_mma, tCtAcc, tCrA[coord], tCrB[coord], tCtAcc)
tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
```

## TMEM to RMEM to GMEM

`get_slice` here belongs to the tiled copy and requires the actual thread
index, not `0` and not a `ThrMma` helper:

```python
thread_x, _, _ = cute.arch.thread_idx()
tmem_atom = cute.make_copy_atom(
    tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
    cutlass.Float32,
)
tmem_copy = tcgen05.make_tmem_copy(tmem_atom, tmem_epi[None, 0])
thread_copy = tmem_copy.get_slice(thread_x)
tmem_source = thread_copy.partition_S(tmem_epi)
gmem_destination = thread_copy.partition_D(gmem_epi)
registers = cute.make_rmem_tensor(
    gmem_destination[None, None, 0].shape,
    cutlass.Float32,
)
```

Then `cute.copy` loads TMEM into registers and `cute.autovec_copy` stores the
register tensor to the partitioned GMEM destination.

## Scalar scale + bias + ReLU epilogue

This exact indexing and scalar arithmetic recipe passed numerical validation on
the shared B300. Give one CTA to each row and let 128 threads stream across the
columns:

```python
@cute.kernel
def scale_bias_relu_kernel(
    output: cute.Tensor,
    bias: cute.Tensor,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(N // THREADS_PER_CTA):
        column = iteration * THREADS_PER_CTA + thread_idx
        value = output[row_idx, column].to(cutlass.Float32)
        value = value * (SCALE_A * SCALE_B)
        value = value + bias[column].to(cutlass.Float32)
        output[row_idx, column] = value * (value > 0.0)
```

Launch it after the GEMM launch in the same `@cute.jit` function:

```python
scale_bias_relu_kernel(output, bias).launch(
    grid=(M, 1, 1),
    block=(THREADS_PER_CTA, 1, 1),
)
```

The independent neutral probe reported `max_abs_error=0.0`, device time
`0.06582400000351481 ms`, and profile ID
`8b73adb4-5b02-4906-9715-ba4ee7910382`.

## Kernel launch

Bind every kernel argument first, then launch with explicit keywords:

```python
fp8_kernel(arg0, arg1, arg2).launch(
    grid=(grid_x, grid_y, 1),
    block=(128, 1, 1),
)
```

## Error recovery rule

If the compiler reports a missing method or attribute, do not substitute a
plausible name. Search this page and `examples/fp8-mma-one-tile.py`, restore the
closest exact verified sequence, and change only the object or coordinate that
must vary for the public task.
