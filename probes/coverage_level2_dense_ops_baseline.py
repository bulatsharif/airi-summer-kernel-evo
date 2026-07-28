"""Developer baseline for three Level 2 coverage tasks.

This is candidate-only framework context, not a complete benchmark answer.
Adapt M/N/K, scales, stage count, public function names, and the task-specific
epilogue. Keep the pipeline, TMA, TMEM, and copy APIs unchanged until this core
passes the remote evaluator.

Required call structure:

1. Keep `dense_fp8_gemm_kernel` as the only device kernel for the GEMM core.
2. Keep `dense_fp8_gemm` as `@cute.jit`; it constructs the MMA/TMA/layout
   objects and launches `dense_fp8_gemm_kernel`.
3. The task's public `@cute.jit` entrypoint calls `dense_fp8_gemm(...)`, then
   launches any separate elementwise/reduction kernel.

Never add an `@cute.kernel` wrapper around `dense_fp8_gemm`. Never call a
decorated kernel without `.launch(...)`.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


FP8_MAX = 448.0
COMMON_M = 1024
COMMON_N = 8192
COMMON_K = 8192
POOL_M = 128
POOL_N = 32768
POOL_K = 32768
COMMON_OUTPUT_SCALE = (1.0 / FP8_MAX) * (
    COMMON_K ** -0.5 / FP8_MAX
)
POOL_OUTPUT_SCALE = (1.0 / FP8_MAX) * (
    POOL_K ** -0.5 / FP8_MAX
)

MMA_TILER_MNK = (128, 256, 128)
THREADS_PER_CTA = 128
AB_STAGES = 3
ACC_STAGES = 1
AB_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def dense_fp8_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    output: cute.Tensor,
    smem_layout_a: cute.ComposedLayout,
    smem_layout_b: cute.ComposedLayout,
    output_scale: cutlass.Constexpr,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    block_m, block_n, _ = cute.arch.block_idx()
    mma_coord = (block_m, block_n, None)

    smem = utils.SmemAllocator()
    storage = smem.allocate(SharedStorage)
    smem_a = smem.allocate_tensor(
        element_type=AB_DTYPE,
        layout=smem_layout_a.outer,
        byte_alignment=128,
        swizzle=smem_layout_a.inner,
    )
    smem_b = smem.allocate_tensor(
        element_type=AB_DTYPE,
        layout=smem_layout_b.outer,
        byte_alignment=128,
        swizzle=smem_layout_b.inner,
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

    one_stage_a = cute.select(smem_layout_a, mode=[0, 1, 2])
    one_stage_b = cute.select(smem_layout_b, mode=[0, 1, 2])
    transaction_bytes = cute.size_in_bytes(
        AB_DTYPE,
        one_stage_a,
    ) + cute.size_in_bytes(
        AB_DTYPE,
        one_stage_b,
    )

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
        mma_coord,
        proj=(1, None, 1),
    )
    global_b = cute.local_tile(
        tma_tensor_b,
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
    acc_shape = tiled_mma.partition_shape_C(MMA_TILER_MNK[:2])
    tmem_accumulator = tiled_mma.make_fragment_C(acc_shape)

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
    tmem_accumulator = cute.make_tensor(
        tmem_ptr,
        tmem_accumulator.layout,
    )

    subtile_count = 2
    epilogue_tiler = (
        (
            cute.size(tmem_accumulator, mode=[0, 0]),
            cute.size(tmem_accumulator, mode=[0, 1]) // subtile_count,
        ),
    )
    tmem_acc_epilogue = cute.zipped_divide(
        tmem_accumulator,
        epilogue_tiler,
    )
    global_c_epilogue = cute.zipped_divide(
        mma_global_c,
        epilogue_tiler,
    )
    tmem_atom = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64),
        ACC_DTYPE,
    )
    tmem_copy = tcgen05.make_tmem_copy(
        tmem_atom,
        tmem_acc_epilogue[None, 0],
    )
    tmem_thread_copy = tmem_copy.get_slice(thread_idx)
    tmem_source = tmem_thread_copy.partition_S(tmem_acc_epilogue)
    global_destination = tmem_thread_copy.partition_D(global_c_epilogue)
    register_accumulator = cute.make_rmem_tensor(
        global_destination[None, None, 0].shape,
        ACC_DTYPE,
    )

    if warp_idx == 0:
        empty_accumulator = acc_producer.acquire_and_advance()
        num_k_tiles = cute.size(global_a, mode=[2])
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
    for tile_idx in cutlass.range(cute.size(tmem_source, mode=[2])):
        cute.copy(
            tmem_copy,
            tmem_source[None, None, tile_idx],
            register_accumulator,
        )
        register_accumulator.store(
            register_accumulator.load() * output_scale
        )
        cute.autovec_copy(
            register_accumulator,
            global_destination[None, None, tile_idx],
        )
    full_accumulator.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def dense_fp8_gemm(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    output: cute.Tensor,
    output_scale: cutlass.Constexpr,
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
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
        tcgen05.CtaGroup.ONE
    )
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        matrix_a,
        cute.select(smem_layout_a, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        matrix_b_nk,
        cute.select(smem_layout_b, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    dense_fp8_gemm_kernel(
        tiled_mma,
        tma_a.atom,
        tma_a.tma_tensor,
        tma_b.atom,
        tma_b.tma_tensor,
        output,
        smem_layout_a,
        smem_layout_b,
        output_scale,
    ).launch(
        grid=cute.ceil_div(
            (*output.layout.shape, 1),
            MMA_TILER_MNK[:2],
        ),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.jit
def warp_max(value):
    other = cute.arch.shuffle_sync_bfly(value, 16)
    value = value * (value >= other) + other * (other > value)
    other = cute.arch.shuffle_sync_bfly(value, 8)
    value = value * (value >= other) + other * (other > value)
    other = cute.arch.shuffle_sync_bfly(value, 4)
    value = value * (value >= other) + other * (other > value)
    other = cute.arch.shuffle_sync_bfly(value, 2)
    value = value * (value >= other) + other * (other > value)
    other = cute.arch.shuffle_sync_bfly(value, 1)
    value = value * (value >= other) + other * (other > value)
    return value


@cute.jit
def mish(value):
    value = cutlass.Float32(value)
    softplus = cute.log(cutlass.Float32(1.0 + cute.exp(value)))
    return cutlass.Float32(
        value * cute.tanh(cutlass.Float32(softplus))
    )


@cute.jit
def gelu(value):
    return 0.5 * value * (
        1.0 + cute.erf(cutlass.Float32(value * 0.7071067811865476))
    )


@cute.kernel
def mish_mish_kernel(output: cute.Tensor, bias: cute.Tensor):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    for column_block in cutlass.range(COMMON_N // THREADS_PER_CTA):
        column = column_block * THREADS_PER_CTA + lane
        value = output[row, column].to(ACC_DTYPE)
        value += bias[column].to(ACC_DTYPE)
        output[row, column] = mish(mish(value))


@cute.jit
def matmul_mish_mish(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    dense_fp8_gemm(
        matrix_a,
        matrix_b_nk,
        output,
        COMMON_OUTPUT_SCALE,
    )
    mish_mish_kernel(output, bias).launch(
        grid=(COMMON_M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.kernel
def maxpool_sum_scale_kernel(
    matrix: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    partial = cutlass.Float32(0.0)
    for iteration in cutlass.range((POOL_N // 2) // 32):
        first_column = (iteration * 32 + lane) * 2
        first = matrix[row, first_column].to(ACC_DTYPE)
        first += bias[first_column].to(ACC_DTYPE)
        second = matrix[row, first_column + 1].to(ACC_DTYPE)
        second += bias[first_column + 1].to(ACC_DTYPE)
        pooled = first * (first >= second) + second * (second > first)
        partial += pooled
    total = warp_sum(partial)
    if lane == 0:
        output[row] = total * 0.5


@cute.jit
def matmul_maxpool_sum_scale(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    scratch: cute.Tensor,
    output: cute.Tensor,
):
    dense_fp8_gemm(
        matrix_a,
        matrix_b_nk,
        scratch,
        POOL_OUTPUT_SCALE,
    )
    maxpool_sum_scale_kernel(scratch, bias, output).launch(
        grid=(POOL_M, 1, 1),
        block=(32, 1, 1),
    )


@cute.kernel
def gelu_softmax_kernel(
    matrix: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    partial_max = cutlass.Float32(-3.402823466e38)
    for iteration in cutlass.range(COMMON_N // 32):
        column = iteration * 32 + lane
        value = matrix[row, column].to(ACC_DTYPE)
        value += bias[column].to(ACC_DTYPE)
        transformed = gelu(value)
        partial_max = (
            partial_max * (partial_max >= transformed)
            + transformed * (transformed > partial_max)
        )
    maximum = warp_max(partial_max)

    partial_sum = cutlass.Float32(0.0)
    for iteration in cutlass.range(COMMON_N // 32):
        column = iteration * 32 + lane
        value = matrix[row, column].to(ACC_DTYPE)
        value += bias[column].to(ACC_DTYPE)
        partial_sum += cute.exp(cutlass.Float32(gelu(value) - maximum))
    denominator = warp_sum(partial_sum)

    for iteration in cutlass.range(COMMON_N // 32):
        column = iteration * 32 + lane
        value = matrix[row, column].to(ACC_DTYPE)
        value += bias[column].to(ACC_DTYPE)
        numerator = cute.exp(cutlass.Float32(gelu(value) - maximum))
        output[row, column] = numerator / denominator


@cute.jit
def matmul_gelu_softmax(
    matrix_a: cute.Tensor,
    matrix_b_nk: cute.Tensor,
    bias: cute.Tensor,
    scratch: cute.Tensor,
    output: cute.Tensor,
):
    dense_fp8_gemm(
        matrix_a,
        matrix_b_nk,
        scratch,
        COMMON_OUTPUT_SCALE,
    )
    gelu_softmax_kernel(scratch, bias, output).launch(
        grid=(COMMON_M, 1, 1),
        block=(32, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
