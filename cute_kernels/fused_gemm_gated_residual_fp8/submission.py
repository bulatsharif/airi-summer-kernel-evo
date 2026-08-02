# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Full license text: ../NVIDIA_BSD_3_CLAUSE.txt
#
# The FP8 mainloop and TMEM epilogue are educational adaptations of NVIDIA's
# CuTe DSL Blackwell tutorial GEMM.

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


M = 2048
N = 4096
K = 4096
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
OUTPUT_SCALE = SCALE_A * SCALE_B
AB_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
C_DTYPE = cutlass.Float32
MMA_TILER_MNK = (128, 256, 128)
THREADS = 128
AB_STAGES = 4
ACC_STAGES = 1
SEED = 20260731


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def gemm_gated_residual_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    matrix_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    matrix_b_nk: cute.Tensor,
    gate: cute.Tensor,
    residual: cute.Tensor,
    output: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    block_m, block_n, _ = cute.arch.block_idx()
    tile_coord = (block_m, block_n, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    smem_a = smem.allocate_tensor(
        element_type=AB_DTYPE,
        layout=a_smem_layout.outer,
        byte_alignment=128,
        swizzle=a_smem_layout.inner,
    )
    smem_b = smem.allocate_tensor(
        element_type=AB_DTYPE,
        layout=b_smem_layout.outer,
        byte_alignment=128,
        swizzle=b_smem_layout.inner,
    )

    allocation_barrier = pipeline.NamedBarrier(
        barrier_id=1, num_threads=THREADS
    )
    tmem = utils.TmemAllocator(
        storage.tmem_holding_buf.ptr,
        barrier_for_retrieve=allocation_barrier,
    )
    tmem.allocate(512)
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_a)
        cpasync.prefetch_descriptor(tma_atom_b)

    tma_copy_bytes = cute.size_in_bytes(
        AB_DTYPE, cute.select(a_smem_layout, mode=[0, 1, 2])
    ) + cute.size_in_bytes(
        AB_DTYPE, cute.select(b_smem_layout, mode=[0, 1, 2])
    )
    ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
        num_stages=AB_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        tx_count=tma_copy_bytes,
        barrier_storage=storage.ab_mbar_ptr.data_ptr(),
    ).make_participants()
    acc_producer, acc_consumer = pipeline.PipelineUmmaAsync.create(
        num_stages=ACC_STAGES,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, THREADS
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    global_a = cute.local_tile(
        matrix_a, MMA_TILER_MNK, tile_coord, proj=(1, None, 1)
    )
    global_b = cute.local_tile(
        matrix_b_nk, MMA_TILER_MNK, tile_coord, proj=(None, 1, 1)
    )
    global_gate = cute.local_tile(
        gate, MMA_TILER_MNK, tile_coord, proj=(1, 1, None)
    )
    global_residual = cute.local_tile(
        residual, MMA_TILER_MNK, tile_coord, proj=(1, 1, None)
    )
    global_output = cute.local_tile(
        output, MMA_TILER_MNK, tile_coord, proj=(1, 1, None)
    )

    mma_slice = tiled_mma.get_slice(0)
    mma_global_a = mma_slice.partition_A(global_a)
    mma_global_b = mma_slice.partition_B(global_b)
    mma_global_gate = mma_slice.partition_C(global_gate)
    mma_global_residual = mma_slice.partition_C(global_residual)
    mma_global_output = mma_slice.partition_C(global_output)
    mma_smem_a = tiled_mma.make_fragment_A(smem_a)
    mma_smem_b = tiled_mma.make_fragment_B(smem_b)
    tmem_accumulator = tiled_mma.make_fragment_C(
        tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    )

    tma_smem_a, tma_global_a = cute.nvgpu.cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_a, 0, 3),
        cute.group_modes(mma_global_a, 0, 3),
    )
    tma_smem_b, tma_global_b = cute.nvgpu.cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_b, 0, 3),
        cute.group_modes(mma_global_b, 0, 3),
    )

    tmem.wait_for_alloc()
    tmem_ptr = tmem.retrieve_ptr(ACC_DTYPE)
    tmem_accumulator = cute.make_tensor(tmem_ptr, tmem_accumulator.layout)
    epilogue_tiler = (
        (
            cute.size(tmem_accumulator, mode=[0, 0]),
            cute.size(tmem_accumulator, mode=[0, 1]) // 4,
        ),
    )
    accumulator_epilogue = cute.zipped_divide(
        tmem_accumulator, epilogue_tiler
    )
    gate_epilogue = cute.zipped_divide(mma_global_gate, epilogue_tiler)
    residual_epilogue = cute.zipped_divide(
        mma_global_residual, epilogue_tiler
    )
    output_epilogue = cute.zipped_divide(
        mma_global_output, epilogue_tiler
    )
    tmem_copy = tcgen05.make_tmem_copy(
        cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), ACC_DTYPE
        ),
        accumulator_epilogue[None, 0],
    )
    thread_copy = tmem_copy.get_slice(thread_idx)
    thread_accumulator = thread_copy.partition_S(accumulator_epilogue)
    thread_gate = thread_copy.partition_D(gate_epilogue)
    thread_residual = thread_copy.partition_D(residual_epilogue)
    thread_output = thread_copy.partition_D(output_epilogue)
    register_accumulator = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape, ACC_DTYPE
    )
    register_gate = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape, C_DTYPE
    )
    register_residual = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape, C_DTYPE
    )
    register_output = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape, C_DTYPE
    )

    if warp_idx == 0:
        empty_accumulator = acc_producer.acquire_and_advance()
        for _ in cutlass.range(
            cute.size(global_a, mode=[2]),
            prefetch_stages=AB_STAGES - 2,
        ):
            empty_ab = ab_producer.acquire_and_advance()
            cute.copy(
                tma_atom_a,
                tma_global_a[(None, empty_ab.count)],
                tma_smem_a[(None, empty_ab.index)],
                tma_bar_ptr=empty_ab.barrier,
            )
            cute.copy(
                tma_atom_b,
                tma_global_b[(None, empty_ab.count)],
                tma_smem_b[(None, empty_ab.index)],
                tma_bar_ptr=empty_ab.barrier,
            )
            full_ab = ab_consumer.wait_and_advance()
            for k_block in cutlass.range_constexpr(
                cute.size(mma_smem_a, mode=[2])
            ):
                k_coord = (None, None, k_block, full_ab.index)
                cute.gemm(
                    tiled_mma,
                    tmem_accumulator,
                    mma_smem_a[k_coord],
                    mma_smem_b[k_coord],
                    tmem_accumulator,
                )
                tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
            full_ab.release()
        empty_accumulator.commit()

    tmem.relinquish_alloc_permit()
    full_accumulator = acc_consumer.wait_and_advance()
    for epilogue_tile in cutlass.range(
        cute.size(thread_accumulator, mode=[2])
    ):
        cute.copy(
            tmem_copy,
            thread_accumulator[None, None, epilogue_tile],
            register_accumulator,
        )
        cute.autovec_copy(
            thread_gate[None, None, epilogue_tile], register_gate
        )
        cute.autovec_copy(
            thread_residual[None, None, epilogue_tile], register_residual
        )
        register_output.store(
            register_accumulator.load()
            * OUTPUT_SCALE
            * register_gate.load()
            + register_residual.load()
        )
        cute.autovec_copy(
            register_output, thread_output[None, None, epilogue_tile]
        )
    full_accumulator.release()
    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def gemm_gated_residual(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    gate: cute.Tensor,
    residual: cute.Tensor,
    output: cute.Tensor,
):
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        matrix_a.element_type,
        utils.LayoutEnum.from_tensor(matrix_a).mma_major_mode(),
        utils.LayoutEnum.from_tensor(matrix_b_nk).mma_major_mode(),
        ACC_DTYPE,
        tcgen05.CtaGroup.ONE,
        MMA_TILER_MNK[:2],
    )
    a_smem_layout = sm100_utils.make_smem_layout_a(
        tiled_mma,
        MMA_TILER_MNK,
        matrix_a.element_type,
        AB_STAGES,
    )
    b_smem_layout = sm100_utils.make_smem_layout_b(
        tiled_mma,
        MMA_TILER_MNK,
        matrix_b_nk.element_type,
        AB_STAGES,
    )
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
        tcgen05.CtaGroup.ONE
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        matrix_a,
        cute.select(a_smem_layout, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        matrix_b_nk,
        cute.select(b_smem_layout, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    gemm_gated_residual_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        gate,
        residual,
        output,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=cute.ceil_div((*output.layout.shape, 1), MMA_TILER_MNK[:2]),
        block=(THREADS, 1, 1),
    )


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    torch.manual_seed(SEED)
    source_a = torch.empty(
        (M, K), device="cuda", dtype=torch.float32
    ).uniform_(-1.0, 1.0)
    source_b_nk = torch.empty(
        (N, K), device="cuda", dtype=torch.float32
    ).uniform_(-WEIGHT_BOUND, WEIGHT_BOUND)
    gate = torch.randn((M, N), device="cuda", dtype=torch.float32)
    residual = torch.randn((M, N), device="cuda", dtype=torch.float32)
    storage_a = torch.empty_like(source_a, dtype=torch.uint8)
    storage_b = torch.empty_like(source_b_nk, dtype=torch.uint8)
    output = torch.empty((M, N), device="cuda", dtype=torch.float32)

    matrix_a = create_cute_tensor_for_fp8(
        storage_a, AB_DTYPE, 1, source_a * FP8_MAX
    )
    matrix_b_nk = create_cute_tensor_for_fp8(
        storage_b,
        AB_DTYPE,
        1,
        source_b_nk * (FP8_MAX / WEIGHT_BOUND),
    )
    gate_tensor = from_dlpack(gate).mark_layout_dynamic(leading_dim=1)
    residual_tensor = from_dlpack(residual).mark_layout_dynamic(leading_dim=1)
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)
    compiled = cute.compile(
        gemm_gated_residual,
        matrix_a,
        matrix_b_nk,
        gate_tensor,
        residual_tensor,
        output_tensor,
    )

    compiled(
        matrix_a,
        matrix_b_nk,
        gate_tensor,
        residual_tensor,
        output_tensor,
    )
    reference = (
        torch._scaled_mm(
            storage_a.view(torch.float8_e4m3fn),
            storage_b.view(torch.float8_e4m3fn).t(),
            scale_a=torch.tensor(SCALE_A, device="cuda"),
            scale_b=torch.tensor(SCALE_B, device="cuda"),
            out_dtype=torch.float32,
        )
        * gate
        + residual
    )
    error = (output - reference).abs()
    max_abs = error.max().item()
    mean_abs = error.mean().item()
    if not torch.isfinite(output).all().item() or max_abs > 0.002:
        raise RuntimeError(
            f"validation failed: max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.9f}"
        )

    for _ in range(5):
        compiled(
            matrix_a,
            matrix_b_nk,
            gate_tensor,
            residual_tensor,
            output_tensor,
        )
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(20):
        compiled(
            matrix_a,
            matrix_b_nk,
            gate_tensor,
            residual_tensor,
            output_tensor,
        )
    end.record()
    end.synchronize()
    kernel_time_ms = start.elapsed_time(end) / 20
    print(
        "kernel=fused_gemm_gated_residual_fp8 "
        f"shape=({M},{N},{K}) max_abs={max_abs:.6f} "
        f"mean_abs={mean_abs:.9f} kernel_time_ms={kernel_time_ms:.6f} PASS"
    )


main()
