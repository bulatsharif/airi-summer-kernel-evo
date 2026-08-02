import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


M = 8192
N = 2304
K = 768
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_X = 1.0 / FP8_MAX
SCALE_W = WEIGHT_BOUND / FP8_MAX
OUTPUT_SCALE = SCALE_X * SCALE_W
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
            register_accumulator.load() * OUTPUT_SCALE
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
    hidden_states: cute.Tensor,
    packed_qkv_weight: cute.Tensor,
    output: cute.Tensor,
):
    a_major = utils.LayoutEnum.from_tensor(hidden_states).mma_major_mode()
    b_major = utils.LayoutEnum.from_tensor(packed_qkv_weight).mma_major_mode()
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        hidden_states.element_type,
        a_major,
        b_major,
        ACC_DTYPE,
        tcgen05.CtaGroup.ONE,
        MMA_TILER_MNK[:2],
    )
    smem_layout_a = sm100_utils.make_smem_layout_a(
        tiled_mma,
        MMA_TILER_MNK,
        hidden_states.element_type,
        AB_STAGES,
    )
    smem_layout_b = sm100_utils.make_smem_layout_b(
        tiled_mma,
        MMA_TILER_MNK,
        packed_qkv_weight.element_type,
        AB_STAGES,
    )
    tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(
        tcgen05.CtaGroup.ONE
    )
    tma_a = cute.nvgpu.make_tiled_tma_atom_A(
        tma_op,
        hidden_states,
        cute.select(smem_layout_a, mode=[0, 1, 2]),
        MMA_TILER_MNK,
        tiled_mma,
    )
    tma_b = cute.nvgpu.make_tiled_tma_atom_B(
        tma_op,
        packed_qkv_weight,
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
    ).launch(
        grid=cute.ceil_div(
            (*output.layout.shape, 1),
            MMA_TILER_MNK[:2],
        ),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.kernel
def add_qkv_bias_kernel(output: cute.Tensor, bias_qkv: cute.Tensor):
    thread_idx, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    columns_per_thread = N // THREADS_PER_CTA

    for column_block in cutlass.range(columns_per_thread):
        column = column_block * THREADS_PER_CTA + thread_idx
        output[row, column] = (
            output[row, column].to(ACC_DTYPE)
            + bias_qkv[column].to(ACC_DTYPE)
        )


@cute.jit
def gpt2_qkv_projection(
    hidden_states: cute.Tensor,
    packed_qkv_weight: cute.Tensor,
    bias_qkv: cute.Tensor,
    output: cute.Tensor,
):
    dense_fp8_gemm(hidden_states, packed_qkv_weight, output)
    add_qkv_bias_kernel(output, bias_qkv).launch(
        grid=(M, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
