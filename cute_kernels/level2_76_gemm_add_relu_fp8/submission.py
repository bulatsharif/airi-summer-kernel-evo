# Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: BSD-3-Clause
# Full license text: ../NVIDIA_BSD_3_CLAUSE.txt
#
# The FP8 GEMM pipeline is an educational adaptation of NVIDIA's CuTe DSL
# Blackwell tutorial GEMM. BiasAdd + ReLU are a second CuTe kernel, matching the
# separate operations in KernelBench level2/76.

import torch

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.runtime import from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8


M = 1024
N = 8192
K = 8192
SEED = 20260726

FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
OUTPUT_SCALE = SCALE_A * SCALE_B

AB_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
C_DTYPE = cutlass.Float32

MMA_TILER_MNK = (128, 256, 128)
THREADS_PER_CTA = 128
AB_STAGES = 4
ACC_STAGES = 1
EPILOGUE_THREADS = 256
OUTPUT_ELEMENTS = M * N


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def fp8_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    matrix_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
    a_smem_layout: cute.ComposedLayout,
    b_smem_layout: cute.ComposedLayout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    block_m, block_n, _ = cute.arch.block_idx()
    mma_coord = (block_m, block_n, None)

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
        barrier_id=1,
        num_threads=THREADS_PER_CTA,
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
        AB_DTYPE,
        cute.select(a_smem_layout, mode=[0, 1, 2]),
    ) + cute.size_in_bytes(
        AB_DTYPE,
        cute.select(b_smem_layout, mode=[0, 1, 2]),
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
            pipeline.Agent.Thread,
            THREADS_PER_CTA,
        ),
        barrier_storage=storage.acc_mbar_ptr.data_ptr(),
    ).make_participants()

    global_a = cute.local_tile(
        matrix_a,
        MMA_TILER_MNK,
        mma_coord,
        proj=(1, None, 1),
    )
    global_b = cute.local_tile(
        matrix_b_nk,
        MMA_TILER_MNK,
        mma_coord,
        proj=(None, 1, 1),
    )
    global_c = cute.local_tile(
        output,
        MMA_TILER_MNK,
        mma_coord,
        proj=(1, 1, None),
    )

    mma_slice = tiled_mma.get_slice(0)
    mma_global_a = mma_slice.partition_A(global_a)
    mma_global_b = mma_slice.partition_B(global_b)
    mma_global_c = mma_slice.partition_C(global_c)
    mma_smem_a = tiled_mma.make_fragment_A(smem_a)
    mma_smem_b = tiled_mma.make_fragment_B(smem_b)
    accumulator_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tmem_accumulator = tiled_mma.make_fragment_C(accumulator_shape)

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

    subtile_count = 4
    epilogue_tiler = (
        (
            cute.size(tmem_accumulator, mode=[0, 0]),
            cute.size(tmem_accumulator, mode=[0, 1]) // subtile_count,
        ),
    )
    accumulator_epilogue = cute.zipped_divide(
        tmem_accumulator,
        epilogue_tiler,
    )
    output_epilogue = cute.zipped_divide(mma_global_c, epilogue_tiler)
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        ACC_DTYPE,
    )
    tmem_copy = tcgen05.make_tmem_copy(
        tmem_atom,
        accumulator_epilogue[None, 0],
    )
    tmem_thread_copy = tmem_copy.get_slice(thread_idx)
    thread_accumulator = tmem_thread_copy.partition_S(accumulator_epilogue)
    thread_output = tmem_thread_copy.partition_D(output_epilogue)
    register_accumulator = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape,
        ACC_DTYPE,
    )
    register_output = cute.make_rmem_tensor(
        thread_output[None, None, 0].shape,
        C_DTYPE,
    )

    num_k_tiles = cute.size(global_a, mode=[2])
    if warp_idx == 0:
        empty_accumulator = acc_producer.acquire_and_advance()
        for _ in cutlass.range(
            num_k_tiles,
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
            num_k_blocks = cute.size(mma_smem_a, mode=[2])
            for k_block in cutlass.range_constexpr(num_k_blocks):
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
        register_output.store(
            register_accumulator.load() * OUTPUT_SCALE
        )
        cute.autovec_copy(
            register_output,
            thread_output[None, None, epilogue_tile],
        )
    full_accumulator.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.kernel
def bias_relu_kernel(output: cute.Tensor, bias: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    block_idx, _, _ = cute.arch.block_idx()
    linear_idx = block_idx * EPILOGUE_THREADS + thread_idx
    row_idx = linear_idx // N
    column_idx = linear_idx - row_idx * N

    value = output[row_idx, column_idx] + bias[column_idx]
    zero = cutlass.Float32(0.0)
    if value > zero:
        output[row_idx, column_idx] = value
    else:
        output[row_idx, column_idx] = zero


@cute.jit
def gemm_add_relu(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    a_major = utils.LayoutEnum.from_tensor(matrix_a).mma_major_mode()
    b_major = utils.LayoutEnum.from_tensor(matrix_b_nk).mma_major_mode()
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        matrix_a.element_type,
        a_major,
        b_major,
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
    a_smem_one_stage = cute.select(a_smem_layout, mode=[0, 1, 2])
    b_smem_one_stage = cute.select(b_smem_layout, mode=[0, 1, 2])
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
        tcgen05.CtaGroup.ONE
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        matrix_a,
        a_smem_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        matrix_b_nk,
        b_smem_one_stage,
        MMA_TILER_MNK,
        tiled_mma,
    )

    fp8_gemm_kernel(
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        output,
        a_smem_layout,
        b_smem_layout,
    ).launch(
        grid=cute.ceil_div((*output.layout.shape, 1), MMA_TILER_MNK[:2]),
        block=(THREADS_PER_CTA, 1, 1),
    )
    bias_relu_kernel(output, bias).launch(
        grid=(OUTPUT_ELEMENTS // EPILOGUE_THREADS, 1, 1),
        block=(EPILOGUE_THREADS, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    torch.manual_seed(SEED)
    source_a = torch.rand((M, K), device="cuda", dtype=torch.float32)
    source_b_nk = torch.empty(
        (N, K),
        device="cuda",
        dtype=torch.float32,
    ).uniform_(-WEIGHT_BOUND, WEIGHT_BOUND)
    bias = torch.randn((N,), device="cuda", dtype=torch.float32)

    storage_a = torch.empty((M, K), device="cuda", dtype=torch.uint8)
    storage_b = torch.empty((N, K), device="cuda", dtype=torch.uint8)
    output = torch.empty((M, N), device="cuda", dtype=torch.float32)

    matrix_a = create_cute_tensor_for_fp8(
        storage_a,
        AB_DTYPE,
        1,
        source_a * FP8_MAX,
    )
    matrix_b_nk = create_cute_tensor_for_fp8(
        storage_b,
        AB_DTYPE,
        1,
        source_b_nk * (FP8_MAX / WEIGHT_BOUND),
    )
    bias_tensor = from_dlpack(bias)
    output_tensor = from_dlpack(output).mark_layout_dynamic(leading_dim=1)

    compiled = cute.compile(
        gemm_add_relu,
        matrix_a,
        matrix_b_nk,
        bias_tensor,
        output_tensor,
    )
    compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)

    matrix_a_fp8 = storage_a.view(torch.float8_e4m3fn)
    matrix_b_fp8 = storage_b.view(torch.float8_e4m3fn)
    scale_a = torch.tensor(SCALE_A, device="cuda", dtype=torch.float32)
    scale_b = torch.tensor(SCALE_B, device="cuda", dtype=torch.float32)
    reference = torch.relu(
        torch._scaled_mm(
            matrix_a_fp8,
            matrix_b_fp8.t(),
            scale_a=scale_a,
            scale_b=scale_b,
            out_dtype=torch.float32,
        )
        + bias
    )

    absolute_error = (output - reference).abs()
    full_max_abs = absolute_error.max().item()
    full_mean_abs = absolute_error.mean().item()
    if not torch.isfinite(output).all().item() or full_max_abs > 0.01:
        raise RuntimeError(
            "CuTe FP8 GEMM + BiasAdd + ReLU mismatch: "
            f"max abs error {full_max_abs:.6f}"
        )

    sample_rows = torch.tensor(
        [0, 1, 7, 31, 127, 255, 511, 1023],
        device="cuda",
        dtype=torch.long,
    )
    sample_columns = torch.tensor(
        [0, 3, 31, 255, 1023, 2047, 4095, 8191],
        device="cuda",
        dtype=torch.long,
    )
    original_reference = torch.relu(
        source_a.index_select(0, sample_rows)
        @ source_b_nk.index_select(0, sample_columns).t()
        + bias.index_select(0, sample_columns)
    )
    actual = (
        output.index_select(0, sample_rows)
        .index_select(1, sample_columns)
    )
    sample_max_abs = (actual - original_reference).abs().max().item()
    if sample_max_abs > 0.10:
        raise RuntimeError(
            "FP8 quantization error is unexpectedly high: "
            f"{sample_max_abs:.6f}"
        )

    checksum = output[::128, ::1024].sum().item()
    print(f"result={checksum:.6f}")
    print(
        "task=level2_76_gemm_add_relu "
        f"shape={tuple(output.shape)} "
        "kernels=CuTe_tcgen05+CuTe_bias_relu "
        "gemm_inputs=torch.float8_e4m3fn "
        "accumulation=torch.float32 epilogue=torch.float32 "
        f"full_max_abs_vs_torch_fp8={full_max_abs:.6f} "
        f"full_mean_abs_vs_torch_fp8={full_mean_abs:.9f} "
        f"sample_max_abs_vs_fp32={sample_max_abs:.6f} PASS"
    )
    torch.cuda.synchronize()


main()
