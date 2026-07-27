# Verified CUTLASS 4.6.1 API

This page is the compiler-facing compatibility manifest for the shared B300.
The snippets below were compiled on the remote service on 2026-07-27. Prefer
them over remembered CuTe APIs or examples from another release.

Evidence labels used on this page:

- **SIGNATURE**: the symbol and Python signature were introspected remotely.
- **COMPILE**: a neutral CuTe trace compiled to MLIR on the shared B300 stack.
- **LAUNCH**: a compiled neutral kernel executed and synchronized successfully.

Higher levels include the lower ones. A signature is not proof that an address
space, layout, pipeline protocol, or numerical operation is valid. Never turn a
COMPILE-only snippet into a correctness claim.

## Dense FP8 MMA construction

The following six-argument form is verified for dense E4M3/E5M2 operands:

```python
import cutlass
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05

a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
mma_tiler_mnk = (128, 128, 64)

tiled_mma = sm100_utils.make_trivial_tiled_mma(
    a.element_type,
    a_major,
    b_major,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (128, 128),
)
```

For E4M3 inputs this produced MMA shape `(128, 128, 32)` on the shared B300.
Do not pass `m=`, `n=`, or `k=` keyword arguments. Do not put numeric tile
sizes in the first two positions: those positions are the operand dtype and A
major mode.

## Verified TiledMma and register-fragment surface

The installed `TiledMma` exposes `get_slice`, `get_tile_size`,
`make_fragment_A/B/C`, `partition_shape_A/B/C`, `shape_mnk`, and `size`.
It does not expose `.threads`, `.mma_shape`, or `.shape_stages`.

```text
get_slice(thr_idx) -> ThrMma
get_tile_size(mode_idx) -> Shape
make_fragment_A(input) -> OpResult
make_fragment_B(input) -> OpResult
make_fragment_C(input) -> OpResult
partition_shape_A(shape_mk) -> Any
partition_shape_B(shape_nk) -> Any
partition_shape_C(shape_mn) -> Any
```

Generic register constructors are also exposed:

```text
cute.full(shape, fill_value, dtype) -> TensorSSA
cute.full_like(tensor, fill_value, dtype=None) -> TensorSSA
cute.make_rmem_tensor(layout_or_shape, dtype) -> Tensor
cute.make_rmem_tensor_like(src, dtype=None) -> Tensor
```

`cute.full` requires all three logical arguments; passing only a shape and
fill value is invalid. These signatures do not determine which fragment form
is correct for a particular MMA protocol.

## Verified staged SMEM layouts

This block compiled with the tiled MMA above:

```python
smem_layout_a = sm100_utils.make_smem_layout_a(
    tiled_mma,
    mma_tiler_mnk,
    a.element_type,
    2,  # num_stages
)
smem_layout_b = sm100_utils.make_smem_layout_b(
    tiled_mma,
    mma_tiler_mnk,
    b.element_type,
    2,  # num_stages
)
```

The installed signatures are:

```text
make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, num_stages,
                   *, is_k_major=None)
make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, num_stages,
                   *, is_k_major=None)
```

## Verified TMA atom construction

TMA atom factories live in `cute.nvgpu`, not in `blackwell_helpers`:

```python
tma_a = cute.nvgpu.make_tiled_tma_atom_A(
    sm100_utils.CopyBulkTensorTileG2SOp(),
    a,
    smem_layout_a,
    mma_tiler_mnk,
    tiled_mma,
)
tma_b = cute.nvgpu.make_tiled_tma_atom_B(
    sm100_utils.CopyBulkTensorTileG2SOp(),
    b,
    smem_layout_b,
    mma_tiler_mnk,
    tiled_mma,
)
```

The installed signatures are:

```text
make_tiled_tma_atom_A(op, gmem_tensor, smem_layout, mma_tiler_mnk,
                      tiled_mma, cluster_shape_vmnk=None,
                      *, internal_type=None) -> TmaInfo
make_tiled_tma_atom_B(op, gmem_tensor, smem_layout, mma_tiler_mnk,
                      tiled_mma, cluster_shape_vmnk=None,
                      *, internal_type=None) -> TmaInfo
```

Both calls compiled on the shared B300. Their returned `TmaInfo` exposes
exactly the public fields `atom`, `smem_layout`, and `tma_tensor`.
`sm100_utils.make_tiled_tma_atom_A/B` does not exist in this release.

## Verified typed SMEM allocation

The concise public allocation path is:

```python
smem_allocator = utils.SmemAllocator()
smem_a = smem_allocator.allocate_tensor(
    a.element_type,
    smem_layout_a,
    byte_alignment=128,
)
smem_b = smem_allocator.allocate_tensor(
    b.element_type,
    smem_layout_b,
    byte_alignment=128,
)
```

Its installed signature is:

```text
SmemAllocator.allocate_tensor(self, element_type, layout,
                              byte_alignment=1, swizzle=None) -> Tensor
```

This path was compiled and launched in a real kernel which allocated both FP8
staged layouts and wrote a sentinel result. The lower-level equivalent was also
compiled and launched:

```python
ptr = cute.arch.alloc_smem(dtype, cute.cosize(layout), alignment=128)
tensor = cute.make_tensor(ptr, layout)
```

Use the free function `cute.cosize(layout)`; layout objects do not provide a
`.cosize()` method. Do not submit `make_smem_tensor[_A/_B]`,
`TiledMma.make_smem_A`, or private `cute._cute` helpers; those paths are not
exposed by the installed release.

These probes establish construction and allocation only. The TMA partition,
pipeline protocol, TMEM accumulation, and epilogue still require separate
compile-and-launch verification; do not infer those calls from this page.

## TMA issue election and barrier ordering

`CopyBulkTensorTileG2SOp` is implicitly wrapped in a one-thread election by
the DSL. Put barrier initialization and expected-byte registration under
`elect_one`, partition the tensors for the TMA atom, then call the copy outside
that context:

```python
gmem_tiled = cute.local_tile(tma_tensor, tile_shape, (None, None))
smem_part, gmem_part = cute.nvgpu.cpasync.tma_partition(
    tma_atom,
    0,
    cute.make_layout(1),
    cute.group_modes(smem_tensor, 0, 2),
    cute.group_modes(gmem_tiled, 0, 2),
)
gmem_tile = gmem_part[(None, block_m, block_n)]
with cute.arch.elect_one():
    cute.arch.mbarrier_init(mbar_ptr, 1)
    cute.arch.mbarrier_expect_tx(mbar_ptr, tx_bytes)
cute.arch.mbarrier_init_fence()
cute.arch.barrier()
cute.copy(tma_atom, gmem_tile, smem_part, tma_bar_ptr=mbar_ptr)
with cute.arch.elect_one():
    cute.arch.mbarrier_arrive(mbar_ptr)
cute.arch.mbarrier_wait(mbar_ptr, phase)
```

Do not wrap the TMA `cute.copy` itself in `elect_one`; double election can
deadlock. Do not substitute a manually composed equal-shape view for
`tma_partition`; it can compile and still execute an illegal instruction. This
sequence was launched successfully on the B300 with FP8 storage.

## Verified copy and partition signatures

The installed copy entry point is:

```text
cute.copy(atom, src, dst, *, pred=None, unroll_factor=None, **kwargs) -> None
```

The only top-level CuTe name containing `partition` is `local_partition`:

```text
cute.local_partition(target, tiler, index, proj=1) -> Tensor
```

`cute.partition_S` and `cute.partition_D` are not exposed. Do not mechanically
replace either old name with `local_partition`: first check that the target,
tiler, index, and projection represent the intended partition. TMA atoms also
expose `get(field)` and `with_(**kwargs)`, but these generic object operations
are not a substitute for a verified copy protocol.

## Core call signatures and launch rules

The installed `cute.gemm` signature is:

```python
cute.gemm(atom, d, a, b, c)
```

Keyword spelling is allowed only when it preserves those five required
operands. `cute.gemm()` with no operands is not a launch helper.

Device indices are tuple-valued:

```python
grid_x, grid_y, grid_z = cute.arch.grid_dim()
block_x, block_y, block_z = cute.arch.block_idx()
thread_x, thread_y, thread_z = cute.arch.thread_idx()
block_dim_x, block_dim_y, block_dim_z = cute.arch.block_dim()
```

These functions take no positional axis argument. There is no CUDA-style
`cute.arch.blockDim().x`. Tensor shape is tuple-like as well: use
`tensor.shape[0]`, not `tensor.shape(0)`. Use `cute.size(tensor)` rather than a
`.numel()` method; `cute.size(obj, mode=())` is the installed free function.

There is no `cute.arch.grid_dim_x()`, `cute.launch_config()`, or
`cute.launch_kernel()` in this release. Launch a decorated kernel from JIT
code with the bound kernel object:

```python
kernel(a, b, c).launch(
    grid=(grid_x, grid_y, grid_z),
    block=(128, 1, 1),
)
```

Add `cluster=` and `stream=` only when the kernel contract requires them and
the corresponding objects are already available.

Use plain tuples for grid/block shapes. `cute.Shape` is a typing union in this
release, so `cute.Shape[x, y, z]` raises `TypeError`. There is no `cute.div()`
or `cute.cdiv()` helper; for static integer tiles use `(x + y - 1) // y`.
Use `cutlass.range(...)` for DSL loops; private `cute._range` is not part of the
public API.

Do not import `OperandSource` or `OperandMajorMode` from
`cutlass.cute.nvgpu[.tcgen05]`. For the verified dense path, derive major modes
with `utils.LayoutEnum.from_tensor(...).mma_major_mode()` and use the six-arg
`make_trivial_tiled_mma` constructor above.

The helper is introspected as a generic `*args/**kwargs` wrapper, but the
installed path is verified only with six positional arguments:

```python
sm100_utils.make_trivial_tiled_mma(
    a_dtype,
    a_major,
    b_major,
    cutlass.Float32,
    tcgen05.CtaGroup.ONE,
    (atom_m, atom_n),
)
```

Do not infer keyword names from internal parameter labels. In particular,
`a_major=`, `b_dtype=`, `cta_group=`, and `atom_mn=` were rejected or bound to
the wrong internal position.

## Pipeline constructors

There is no `sm100_utils.make_trivial_pipeline` helper. Construct the TMA/UMMA
pipeline only through the installed `pipeline.PipelineTmaUmma.create` factory
shown below.

```python
import cutlass.pipeline as pipeline

mma_pipeline = pipeline.PipelineTmaUmma.create(
    num_stages=num_stages,
    producer_group=producer_group,
    consumer_group=consumer_group,
    tx_count=tx_count,
    barrier_storage=barrier_storage,
)
```

The factory is not exposed as `cute.PipelineTmaUmma` or
`cute.arch.PipelineTmaUmma`, and `cute.pipeline` is not a public module alias.
Import `cutlass.pipeline` explicitly.

Although the introspected annotation permits `barrier_storage=None`, a neutral
kernel trace rejected `None`. Pass an explicit shared-memory pointer.

The installed `blackwell_helpers` namespace also does not expose convenience
calls named `stage_input_A`, `stage_input_B`, `wait_pipeline`, or
`commit_pipeline`. Use the verified pipeline object methods listed below and
`cute.copy`; do not synthesize wrapper names.

Relevant installed keyword-only constructors:

```text
PipelineTmaUmma.create(num_stages, producer_group, consumer_group, tx_count,
                       barrier_storage=None, cta_layout_vmnk=None,
                       mcast_mode_mn=(1,1), defer_sync=False)
PipelineUmmaAsync.create(num_stages, producer_group, consumer_group,
                         barrier_storage=None, cta_layout_vmnk=None,
                         defer_sync=False)
```

The installed `PipelineTmaUmma` state operations are:

```text
producer_acquire(state, try_acquire_token=None) -> None
producer_get_barrier(state) -> Pointer
producer_commit(state) -> None
producer_tail(state) -> None
consumer_wait(state, try_wait_token=None) -> None
consumer_get_barrier(state) -> Pointer
consumer_release(state) -> None
PipelineState.advance() -> None
PipelineState.clone() -> PipelineState
make_pipeline_state(type, stages) -> PipelineState
```

## TMEM accumulator and epilogue surface

`tiled_mma.get_slice(thread_idx)` returns a `ThrMma`. The following operations
are **SIGNATURE** verified:

```text
ThrMma.partition_A(input_mk) -> Tensor
ThrMma.partition_B(input_nk) -> Tensor
ThrMma.partition_C(input_mn) -> Tensor
TmemAllocator.reserve(num_columns) -> TmemBufferPool
TmemAllocator.retrieve_ptr(dtype=Float32) -> Pointer
TmemBufferPool.allocate_tensor(layout, dtype) -> Tensor
```

The neutral sequence `ThrMma.partition_C` followed by
`ThrMma.make_fragment_C` is **LAUNCH** verified. Its result was a TMEM-backed
tensor, not a normal register tensor; direct `.fill()` was rejected. Do not
derive an accumulator size from nonexistent `TiledMma.threads`.

Epilogue layout/load/store helpers live in `blackwell_helpers`, while the three
partition helpers live in `cutlass.utils`:

```text
sm100_utils.make_smem_layout_epi(...)
sm100_utils.get_tmem_load_op(...)
sm100_utils.get_smem_store_op(...)
utils.epilog_tmem_copy_and_partition(...)
utils.epilog_smem_copy_and_partition(...)
utils.epilog_gmem_copy_and_partition(...)
```

These generic epilogue helper entries are **SIGNATURE** verified only.

A direct one-tile path is now **LAUNCH + NUMERICAL** verified for FP8 E4M3FN
`A[128,64]`, `B[128,64]`, and FP32 `C[128,128]`:

```text
local_tile(GMEM) -> ThrMma.partition_A/B/C
TiledMma.make_fragment_A/B(SMEM)
TiledMma.make_fragment_C(partition_shape_C(...))
TmemAllocator.allocate/wait_for_alloc/retrieve_ptr
cute.make_tensor(tmem_ptr, accumulator_fragment.layout)
cpasync.tma_partition -> PipelineTmaUmma -> cute.gemm
tcgen05.make_tmem_copy -> RMEM -> cute.autovec_copy(GMEM)
```

The launch used a `(128,128,64)` MMA tiler, two A/B stages, 128 threads, a
one-stage `PipelineUmmaAsync` accumulator barrier, 512 allocated TMEM columns,
and a two-subtile `Ld32x32bOp(Repetition.x64)` FP32 epilogue. It produced exact
results for integer-valued FP8 inputs. See the task-selected Level 2 API context
for the verified object-by-object micro-example.

Critical type contracts established by that launch:

- `partition_A/B/C` consume tiled tensors, not layouts;
- `make_fragment_A/B` consume allocated SMEM tensors;
- `cute.make_tensor` consumes `(pointer, layout)`;
- the accumulator fragment layout becomes usable only after its pointer is
  replaced with an allocated TMEM pointer;
- `TmaInfo` is accessed through `.atom` and `.tma_tensor`;
- `PipelineTmaUmma.create(...).make_participants()` and the
  `acquire_and_advance` / `wait_and_advance` protocol are available.

These signatures prove only API compatibility. A compile plus real launch is
still required to validate address spaces, resource use, pipeline protocol,
and numerical correctness.
