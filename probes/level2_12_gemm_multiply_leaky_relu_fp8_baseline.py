import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


M = 1024
N = 8192
K = 8192
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_A = 1.0 / FP8_MAX
SCALE_B = WEIGHT_BOUND / FP8_MAX
MULTIPLIER = 2.0
NEGATIVE_SLOPE = 0.1
SUBTRACT_VALUE = 2.0
POST_MULTIPLY_VALUE = 1.5
DIVISOR = 2.0
SCALING_FACTOR = 0.5
SUM_SCALING_FACTOR = 1.5
FP8_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32
MMA_TILER_MNK = (128, 128, 64)
THREADS_PER_CTA = 128
AB_STAGES = 2
ACC_STAGES = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def fp8_gemm_kernel(
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
    block_m, block_n, _ = cute.arch.block_idx()
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
        (block_m, None, None),
        proj=(1, None, 1),
    )
    global_b = cute.local_tile(
        tma_tensor_b,
        MMA_TILER_MNK,
        (None, block_n, None),
        proj=(None, 1, 1),
    )
    global_c = cute.local_tile(
        output,
        MMA_TILER_MNK,
        (block_m, block_n, None),
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


@cute.kernel
def multiply_leaky_relu_kernel(output: cute.Tensor, bias: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(N // THREADS_PER_CTA):
        column = iteration * THREADS_PER_CTA + thread_idx
        value = output[row_idx, column].to(cutlass.Float32)
        value = (value * (SCALE_A * SCALE_B) + bias[column]) * MULTIPLIER
        output[row_idx, column] = (
            value * (value >= 0.0)
            + value * NEGATIVE_SLOPE * (value < 0.0)
        )


@cute.jit
def launch_fp8_gemm(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
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
    smem_layout_a = sm100_utils.make_smem_layout_a(
        tiled_mma,
        MMA_TILER_MNK,
        matrix_a.element_type,
        AB_STAGES,
    )
    smem_layout_b = sm100_utils.make_smem_layout_b(
        tiled_mma,
        MMA_TILER_MNK,
        matrix_b_nk.element_type,
        AB_STAGES,
    )
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        sm100_utils.CopyBulkTensorTileG2SOp(),
        matrix_a,
        cute.select(smem_layout_a, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        sm100_utils.CopyBulkTensorTileG2SOp(),
        matrix_b_nk,
        cute.select(smem_layout_b, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    fp8_gemm_kernel(
        tiled_mma,
        tma_a.atom,
        tma_a.tma_tensor,
        tma_b.atom,
        tma_b.tma_tensor,
        output,
        smem_layout_a,
        smem_layout_b,
    ).launch(
        grid=(M // MMA_TILER_MNK[0], N // MMA_TILER_MNK[1], 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.kernel
def subtract_multiply_relu_kernel(output: cute.Tensor, bias: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(N // THREADS_PER_CTA):
        column = iteration * THREADS_PER_CTA + thread_idx
        value = output[row_idx, column].to(cutlass.Float32)
        value = value * (SCALE_A * SCALE_B) + bias[column]
        value = (value - SUBTRACT_VALUE) * POST_MULTIPLY_VALUE
        output[row_idx, column] = value * (value > 0.0)


@cute.kernel
def scale_residual_kernel(output: cute.Tensor, bias: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(N // THREADS_PER_CTA):
        column = iteration * THREADS_PER_CTA + thread_idx
        value = output[row_idx, column].to(cutlass.Float32)
        value = value * (SCALE_A * SCALE_B) + bias[column]
        output[row_idx, column] = value * SCALING_FACTOR + value


@cute.kernel
def relu_divide_kernel(output: cute.Tensor, bias: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    row_idx, _, _ = cute.arch.block_idx()
    for iteration in cutlass.range(N // THREADS_PER_CTA):
        column = iteration * THREADS_PER_CTA + thread_idx
        value = output[row_idx, column].to(cutlass.Float32)
        value = value * (SCALE_A * SCALE_B) + bias[column]
        output[row_idx, column] = value * (value > 0.0) / DIVISOR


@cute.kernel
def divide_sum_scale_kernel(scratch: cute.Tensor, output: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    partial = cutlass.Float32(0.0)
    for iteration in cutlass.range(N // 32):
        column = iteration * 32 + lane
        value = scratch[row, column].to(cutlass.Float32)
        partial += value * (SCALE_A * SCALE_B) / 2.0
    partial += cute.arch.shuffle_sync_bfly(partial, 16)
    partial += cute.arch.shuffle_sync_bfly(partial, 8)
    partial += cute.arch.shuffle_sync_bfly(partial, 4)
    partial += cute.arch.shuffle_sync_bfly(partial, 2)
    partial += cute.arch.shuffle_sync_bfly(partial, 1)
    if lane == 0:
        output[row, 0] = partial * SUM_SCALING_FACTOR


@cute.jit
def gemm_multiply_leaky_relu(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    launch_fp8_gemm(matrix_a, matrix_b_nk, output)
    multiply_leaky_relu_kernel(output, bias).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.jit
def matmul_subtract_multiply_relu(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    launch_fp8_gemm(matrix_a, matrix_b_nk, output)
    subtract_multiply_relu_kernel(output, bias).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.jit
def matmul_scaling_residual_add(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    launch_fp8_gemm(matrix_a, matrix_b_nk, output)
    scale_residual_kernel(output, bias).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.jit
def gemm_relu_divide(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    launch_fp8_gemm(matrix_a, matrix_b_nk, output)
    relu_divide_kernel(output, bias).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.jit
def gemm_divide_sum_scaling(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    scratch: cute.Tensor,
    output: cute.Tensor,
):
    launch_fp8_gemm(matrix_a, matrix_b_nk, scratch)
    divide_sum_scale_kernel(scratch, output).launch(
        grid=(M, 1, 1),
        block=(32, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _harness_torch

import cutlass as _harness_cutlass
import cutlass.cute as _harness_cute
from cutlass.cute.runtime import from_dlpack as _harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _harness_create_cute_tensor_for_fp8,
)


_HARNESS_M = 1024
_HARNESS_N = 8192
_HARNESS_K = 8192
_HARNESS_SEED = 20260727
_HARNESS_FP8_MAX = 448.0
_HARNESS_WEIGHT_BOUND = _HARNESS_K ** -0.5
_HARNESS_SCALE_A = 1.0 / _HARNESS_FP8_MAX
_HARNESS_SCALE_B = _HARNESS_WEIGHT_BOUND / _HARNESS_FP8_MAX
_HARNESS_MULTIPLIER = 2.0
_HARNESS_NEGATIVE_SLOPE = 0.1
_HARNESS_FP8_DTYPE = _harness_cutlass.Float8E4M3FN


def main():
    if not _harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _harness_torch.manual_seed(_HARNESS_SEED)
    source_a = _harness_torch.rand(
        (_HARNESS_M, _HARNESS_K),
        device="cuda",
        dtype=_harness_torch.float32,
    )
    source_b_nk = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_K),
        device="cuda",
        dtype=_harness_torch.float32,
    ).uniform_(-_HARNESS_WEIGHT_BOUND, _HARNESS_WEIGHT_BOUND)
    bias = _harness_torch.empty(
        (_HARNESS_N,), device="cuda", dtype=_harness_torch.float32
    ).uniform_(-_HARNESS_WEIGHT_BOUND, _HARNESS_WEIGHT_BOUND)
    storage_a = _harness_torch.empty(
        (_HARNESS_M, _HARNESS_K), device="cuda", dtype=_harness_torch.uint8
    )
    storage_b = _harness_torch.empty(
        (_HARNESS_N, _HARNESS_K), device="cuda", dtype=_harness_torch.uint8
    )
    output = _harness_torch.empty(
        (_HARNESS_M, _HARNESS_N), device="cuda", dtype=_harness_torch.float32
    )

    matrix_a = _harness_create_cute_tensor_for_fp8(
        storage_a,
        _HARNESS_FP8_DTYPE,
        1,
        source_a * _HARNESS_FP8_MAX,
    )
    matrix_b_nk = _harness_create_cute_tensor_for_fp8(
        storage_b,
        _HARNESS_FP8_DTYPE,
        1,
        source_b_nk * (_HARNESS_FP8_MAX / _HARNESS_WEIGHT_BOUND),
    )
    bias_tensor = _harness_from_dlpack(bias)
    output_tensor = _harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _harness_cute.compile(
        gemm_multiply_leaky_relu,
        matrix_a,
        matrix_b_nk,
        bias_tensor,
        output_tensor,
    )
    compiled(matrix_a, matrix_b_nk, bias_tensor, output_tensor)

    a_fp8 = storage_a.view(_harness_torch.float8_e4m3fn)
    b_fp8 = storage_b.view(_harness_torch.float8_e4m3fn)
    scale_a = _harness_torch.tensor(
        _HARNESS_SCALE_A, device="cuda", dtype=_harness_torch.float32
    )
    scale_b = _harness_torch.tensor(
        _HARNESS_SCALE_B, device="cuda", dtype=_harness_torch.float32
    )
    fp8_linear = _harness_torch._scaled_mm(
        a_fp8,
        b_fp8.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=_harness_torch.float32,
    ) + bias
    reference = _harness_torch.nn.functional.leaky_relu(
        fp8_linear * _HARNESS_MULTIPLIER,
        negative_slope=_HARNESS_NEGATIVE_SLOPE,
    )
    full_max_abs = (output - reference).abs().max().item()

    rows = _harness_torch.tensor(
        [0, 1, 7, 31, 127, 255, 511, 1023], device="cuda"
    )
    columns = _harness_torch.tensor(
        [0, 3, 31, 255, 1023, 2047, 4095, 8191], device="cuda"
    )
    fp32_linear = (
        source_a.index_select(0, rows)
        @ source_b_nk.index_select(0, columns).t()
        + bias.index_select(0, columns)
    )
    fp32_reference = _harness_torch.nn.functional.leaky_relu(
        fp32_linear * _HARNESS_MULTIPLIER,
        negative_slope=_HARNESS_NEGATIVE_SLOPE,
    )
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()

    if (
        not _harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.01
        or sample_max_abs > 0.2
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, "
            f"sample_abs={sample_max_abs:.6f}"
        )

    print(
        "task=level2_12_gemm_multiply_leaky_relu "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} PASS"
    )
    _harness_torch.cuda.synchronize()


main()
