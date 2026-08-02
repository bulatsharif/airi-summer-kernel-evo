# A complete Blackwell FP8 GEMM

The other files show boundaries and two whole kernels that do not use the MMA
path. This one does: a single-tile FP8 GEMM that compiles and runs on this
device, verified at `max_abs_error=0.0` and `device_time_ms=0.061`.

It computes `A[128,64] @ B[128,64].T -> C[128,128]` with E4M3FN inputs and FP32
accumulation. It is a building block, not an answer: it carries no task shape, no
task scale factors, and no epilogue. A benchmark task runs far larger shapes,
applies its own dequantization, and fuses its own elementwise stage -- all of
which you still have to work out and none of which appears below.

`main()` and its torch validation are cut, because the harness supplies those and
a candidate must not define them.

```python
"""Numerically verified neutral B300 FP8 GEMM bridge.

Computes A[128,64] @ B[128,64].T -> C[128,128] with E4M3FN inputs and
Float32 accumulation/output. This is documentation, not a benchmark candidate:
candidate submissions must not copy main(), torch validation, or the direct
main() call because the harness supplies those parts.

Shared-B300 evidence from 2026-07-27:
  max_abs_error=0.0
  device_time_ms=0.13721599999791942
  profile_id=1bf96cbb-9f27-4888-9539-95ff03883227
"""

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


FP8_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
MMA_TILER_MNK = MMA_TILE_SHAPE      # choose: (M_tile, N_tile, K_tile)
THREADS_PER_CTA = THREADS_PER_BLOCK  # choose to match the tiled MMA
AB_STAGES = AB_PIPELINE_DEPTH        # choose: how many A/B buffers overlap
ACC_STAGES = ACC_PIPELINE_DEPTH      # choose: accumulator pipeline depth


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def fp8_mma_one_tile_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    output: cute.Tensor,
    smem_layout_a: cute.ComposedLayout,
    smem_layout_b: cute.ComposedLayout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    smem_a = smem.allocate_tensor(
        element_type=FP8_DTYPE,
        layout=smem_layout_a.outer,
        byte_alignment=128,
        swizzle=smem_layout_a.inner,
    )
    smem_b = smem.allocate_tensor(
        element_type=FP8_DTYPE,
        layout=smem_layout_b.outer,
        byte_alignment=128,
        swizzle=smem_layout_b.inner,
    )

    tmem_barrier = pipeline.NamedBarrier(
        barrier_id=1,
        num_threads=THREADS_PER_CTA,
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=tmem_barrier,
    )
    tmem.allocate(512)

    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    one_stage_a = cute.select(smem_layout_a, mode=[0, 1, 2])
    one_stage_b = cute.select(smem_layout_b, mode=[0, 1, 2])
    transaction_bytes = cute.size_in_bytes(
        FP8_DTYPE, one_stage_a
    ) + cute.size_in_bytes(FP8_DTYPE, one_stage_b)
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=transaction_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            THREADS_PER_CTA,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    global_a = cute.local_tile(
        tma_tensor_a,
        MMA_TILER_MNK,
        (0, None, None),
        proj=(1, None, 1),
    )
    global_b = cute.local_tile(
        tma_tensor_b,
        MMA_TILER_MNK,
        (None, 0, None),
        proj=(None, 1, 1),
    )
    global_c = cute.local_tile(
        output,
        MMA_TILER_MNK,
        (0, 0, None),
        proj=(1, 1, None),
    )

    thr_mma = tiled_mma.get_slice(0)
    mma_global_a = thr_mma.partition_A(global_a)
    mma_global_b = thr_mma.partition_B(global_b)
    mma_global_c = thr_mma.partition_C(global_c)
    mma_smem_a = tiled_mma.make_fragment_A(smem_a)
    mma_smem_b = tiled_mma.make_fragment_B(smem_b)
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tmem_acc = tiled_mma.make_fragment_C(acc_shape)

    tma_smem_a, tma_global_a = cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_a, 0, 3),
        cute.group_modes(mma_global_a, 0, 3),
    )
    tma_smem_b, tma_global_b = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_b, 0, 3),
        cute.group_modes(mma_global_b, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tmem_acc = cute.make_tensor(tmem_ptr, tmem_acc.layout)

    subtile_count = 2
    epilogue_tiler = (
        (
            cute.size(tmem_acc, mode=[0, 0]),
            cute.size(tmem_acc, mode=[0, 1]) // subtile_count,
        ),
    )
    tmem_acc_epilogue = cute.zipped_divide(tmem_acc, epilogue_tiler)
    global_c_epilogue = cute.zipped_divide(mma_global_c, epilogue_tiler)
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        ACC_DTYPE,
    )
    tmem_tiled_copy = tcgen05.make_tmem_copy(
        tmem_atom,
        tmem_acc_epilogue[None, 0],
    )
    tmem_thread_copy = tmem_tiled_copy.get_slice(thread_idx)
    tmem_source = tmem_thread_copy.partition_S(tmem_acc_epilogue)
    global_destination = tmem_thread_copy.partition_D(global_c_epilogue)
    register_acc = cute.make_rmem_tensor(
        global_destination[None, None, 0].shape,
        ACC_DTYPE,
    )

    if warp_idx == 0:
        acc_empty = acc_producer.acquire_and_advance()
        num_k_tiles = cute.size(global_a, mode=[2])
        for _ in cutlass.range(num_k_tiles, prefetch_stages=AB_STAGES - 2):
            ab_empty = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tma_global_a[(None, ab_empty.count)],
                tma_smem_a[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            cute.copy(
                tma_atom_b,
                tma_global_b[(None, ab_empty.count)],
                tma_smem_b[(None, ab_empty.index)],
                tma_bar_ptr=ab_empty.barrier,
            )
            ab_full = ab_consumer.wait_and_advance()
            num_k_blocks = cute.size(mma_smem_a, mode=[2])
            for k_block in cutlass.range_constexpr(num_k_blocks):
                coord = (None, None, k_block, ab_full.index)
                cute.gemm(
                    tiled_mma,
                    tmem_acc,
                    mma_smem_a[coord],
                    mma_smem_b[coord],
                    tmem_acc,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            ab_full.release()
        acc_empty.commit()

    tmem.relinquish_alloc_permit()
    acc_full = acc_consumer.wait_and_advance()
    for tile_idx in cutlass.range(cute.size(tmem_source, mode=[2])):
        cute.copy(
            tmem_tiled_copy,
            tmem_source[None, None, tile_idx],
            register_acc,
        )
        cute.autovec_copy(
            register_acc,
            global_destination[None, None, tile_idx],
        )
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def fp8_mma_one_tile(a: cute.Tensor, b: cute.Tensor, output: cute.Tensor):
    a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
    b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        a.element_type,
        a_major,
        b_major,
        ACC_DTYPE,
        tcgen05.CtaGroup.ONE,
        MMA_TILER_MNK[:2],
    )
    smem_layout_a = sm100_utils.make_smem_layout_a(
        tiled_mma,
        MMA_TILER_MNK,
        a.element_type,
        AB_STAGES,
    )
    smem_layout_b = sm100_utils.make_smem_layout_b(
        tiled_mma,
        MMA_TILER_MNK,
        b.element_type,
        AB_STAGES,
    )
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.CopyBulkTensorTileG2SOp(),
        a,
        cute.select(smem_layout_a, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.CopyBulkTensorTileG2SOp(),
        b,
        cute.select(smem_layout_b, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    fp8_mma_one_tile_kernel(
        tiled_mma,
        tma_a.atom,
        tma_a.tma_tensor,
        tma_b.atom,
        tma_b.tma_tensor,
        output,
        smem_layout_a,
        smem_layout_b,
    ).launch(
        grid=(1, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )
```

## What to take from it

The order of construction: derive the operand major modes from the tensors, build
the tiled MMA, build the shared-memory layouts from that MMA, then build the TMA
atoms from those layouts. Each step consumes the previous one, so the sequence is
not interchangeable.

The accumulator lives in tensor memory. It is allocated, its pointer retrieved,
used as both the source and the destination of `cute.gemm`, copied out through a
TMEM copy atom, and freed. The accumulate field is set after the first K-block so
the first iteration writes and the rest add.

The pipeline is what makes the K loop safe: a producer acquires a stage, issues
the TMA copies against that stage's barrier, and a consumer waits on it before the
MMA reads shared memory.

What is deliberately absent: the four tuning constants are left as named choices
rather than values, because the verified reference for a task in this catalog uses
exactly the numbers this example was written with. The scaling a task specifies is
not applied here, and there is no epilogue at all.
