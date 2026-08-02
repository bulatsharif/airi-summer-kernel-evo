import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05


TOKENS = 128
HIDDEN_SIZE = 2560
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEADS_PER_KV = NUM_HEADS // NUM_KV_HEADS
HEAD_DIM = 256
ROTARY_DIM = 64
ROTARY_HALF = ROTARY_DIM // 2
INTERMEDIATE_SIZE = 9216
Q_GATE_SIZE = 2 * NUM_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
QUERY_SIZE = NUM_HEADS * HEAD_DIM
EPSILON = 1.0e-6

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_PROJECTION_SCALE = 1.0 / 64.0
MLP_ACTIVATION_SCALE = 1.0 / 32.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_CONTEXT_SCALE = QUERY_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = INTERMEDIATE_SIZE ** -0.5 / 448.0

S_QKV = NORM_SCALE * WEIGHT_H_SCALE / QKV_SCALE
S_MLP = NORM_SCALE * WEIGHT_H_SCALE / MLP_PROJECTION_SCALE
S_OUT = CONTEXT_SCALE * WEIGHT_CONTEXT_SCALE
S_DOWN = MLP_ACTIVATION_SCALE * WEIGHT_MLP_SCALE

FP8_DTYPE = cutlass.Float8E4M3FN
ACC_DTYPE = cutlass.Float32

# Blackwell tcgen05 FP8 GEMM configuration.
MMA_M = TOKENS
MMA_N = 128
MMA_K = 32
MMA_TILER_MNK = (MMA_M, MMA_N, MMA_K)
THREADS_PER_CTA = 128
AB_STAGES = 3
ACC_STAGES = 1


@cute.struct
class SharedStorage:
    ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, AB_STAGES * 2]
    acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, ACC_STAGES * 2]
    tmem_holding_buf: cutlass.Int32


@cute.kernel
def rms_norm_kernel(hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
    t, _, _ = cute.arch.block_idx()
    th, _, _ = cute.arch.thread_idx()
    s = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(HIDDEN_SIZE // 32):
        x = hidden[t, i * 32 + th].to(cutlass.Float32) * INPUT_SCALE
        s = s + x * x
    s = s + cute.arch.shuffle_sync_bfly(s, 16)
    s = s + cute.arch.shuffle_sync_bfly(s, 8)
    s = s + cute.arch.shuffle_sync_bfly(s, 4)
    s = s + cute.arch.shuffle_sync_bfly(s, 2)
    s = s + cute.arch.shuffle_sync_bfly(s, 1)
    inv = cute.rsqrt(s / HIDDEN_SIZE + EPSILON)
    for i in cutlass.range_constexpr(HIDDEN_SIZE // 32):
        idx = i * 32 + th
        x = hidden[t, idx].to(cutlass.Float32) * INPUT_SCALE
        n = x * inv * (1.0 + weight[idx].to(cutlass.Float32))
        output[t, idx] = cutlass.Float8E4M3FN(n / NORM_SCALE)


@cute.kernel
def fp8_gemm_kernel(
    tiled_mma: cute.TiledMma,
    tma_atom_a: cute.CopyAtom,
    tma_tensor_a: cute.Tensor,
    tma_atom_b: cute.CopyAtom,
    tma_tensor_b: cute.Tensor,
    c_tensor: cute.Tensor,
    extra: cute.Tensor,
    out_scale: cutlass.Constexpr,
    variant: cutlass.Constexpr,
    smem_layout_a: cute.ComposedLayout,
    smem_layout_b: cute.ComposedLayout,
):
    thread_idx, _, _ = cute.arch.thread_idx()
    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    nb, _, _ = cute.arch.block_idx()

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
        (None, nb, None),
        proj=(None, 1, 1),
    )
    global_c = cute.local_tile(
        c_tensor,
        MMA_TILER_MNK,
        (0, nb, None),
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

    global_extra = cute.local_tile(
        extra,
        MMA_TILER_MNK,
        (0, nb, None),
        proj=(1, 1, None),
    )
    extra_epilogue = cute.zipped_divide(
        thr_mma.partition_C(global_extra), epilogue_tiler
    )
    extra_dest = tmem_thread_copy.partition_D(extra_epilogue)

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
        for i in cutlass.range_constexpr(cute.size(register_acc, mode=0)):
            for j in cutlass.range_constexpr(cute.size(register_acc, mode=1)):
                val = register_acc[i, j]
                if cutlass.const_expr(variant == 0):
                    global_destination[None, None, tile_idx][i, j] = (
                        cutlass.Float8E4M3FN(val * out_scale)
                    )
                elif cutlass.const_expr(variant == 1):
                    global_destination[None, None, tile_idx][i, j] = (
                        cutlass.Float32(
                            extra_dest[None, None, tile_idx][i, j].to(
                                cutlass.Float32
                            )
                            * INPUT_SCALE
                            + val * out_scale
                        )
                    )
                else:
                    global_destination[None, None, tile_idx][i, j] = (
                        cutlass.Float32(
                            extra_dest[None, None, tile_idx][i, j].to(
                                cutlass.Float32
                            )
                            + val * out_scale
                        )
                    )
    acc_full.release()

    pipeline.sync(barrier_id=1)
    tmem.free(tmem_ptr)


@cute.jit
def fp8_gemm(
    a: cute.Tensor,
    b: cute.Tensor,
    c: cute.Tensor,
    extra: cute.Tensor,
    out_scale: cutlass.Constexpr,
    variant: cutlass.Constexpr,
):
    a_major = utils.LayoutEnum.from_tensor(a).mma_major_mode()
    b_major = utils.LayoutEnum.from_tensor(b).mma_major_mode()
    tiled_mma = sm100_utils.make_trivial_tiled_mma(
        a.element_type,
        b.element_type,
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
    n_tiles = cute.size(c, mode=1) // MMA_N
    fp8_gemm_kernel(
        tiled_mma,
        tma_a.atom,
        tma_a.tma_tensor,
        tma_b.atom,
        tma_b.tma_tensor,
        c,
        extra,
        out_scale,
        variant,
        smem_layout_a,
        smem_layout_b,
    ).launch(
        grid=(n_tiles, 1, 1),
        block=(THREADS_PER_CTA, 1, 1),
    )


@cute.kernel
def q_headnorm_kernel(
    q_gate: cute.Tensor,
    q_norm_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    query_out: cute.Tensor,
):
    # grid = (NUM_HEADS, TOKENS); warp per (head, token)
    h, t, _ = cute.arch.block_idx()
    th, _, _ = cute.arch.thread_idx()
    s = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        d = i * 32 + th
        x = q_gate[t, h * 512 + d].to(cutlass.Float32) * QKV_SCALE
        s = s + x * x
    s = s + cute.arch.shuffle_sync_bfly(s, 16)
    s = s + cute.arch.shuffle_sync_bfly(s, 8)
    s = s + cute.arch.shuffle_sync_bfly(s, 4)
    s = s + cute.arch.shuffle_sync_bfly(s, 2)
    s = s + cute.arch.shuffle_sync_bfly(s, 1)
    inv = cute.rsqrt(s / HEAD_DIM + EPSILON)
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        d = i * 32 + th
        x = q_gate[t, h * 512 + d].to(cutlass.Float32) * QKV_SCALE
        n = x * inv * (1.0 + q_norm_weight[d].to(cutlass.Float32))
        out_v = n
        if d < ROTARY_HALF:
            x2 = q_gate[t, h * 512 + (d + ROTARY_HALF)].to(cutlass.Float32) * QKV_SCALE
            n2 = x2 * inv * (1.0 + q_norm_weight[d + ROTARY_HALF].to(cutlass.Float32))
            c = cos[t, d].to(cutlass.Float32)
            si = sin[t, d].to(cutlass.Float32)
            out_v = n * c - n2 * si
        elif d < ROTARY_DIM:
            c = cos[t, d - ROTARY_HALF].to(cutlass.Float32)
            si = sin[t, d - ROTARY_HALF].to(cutlass.Float32)
            xp = q_gate[t, h * 512 + (d - ROTARY_HALF)].to(cutlass.Float32) * QKV_SCALE
            np = xp * inv * (1.0 + q_norm_weight[d - ROTARY_HALF].to(cutlass.Float32))
            out_v = n * c + np * si
        query_out[t, h * HEAD_DIM + d] = cutlass.Float8E4M3FN(out_v / QKV_SCALE)


@cute.kernel
def k_headnorm_kernel(
    k_in: cute.Tensor,
    k_norm_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    key_out: cute.Tensor,
):
    # grid = (NUM_KV_HEADS, TOKENS); warp per (head, token)
    h, t, _ = cute.arch.block_idx()
    th, _, _ = cute.arch.thread_idx()
    s = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        d = i * 32 + th
        x = k_in[t, h * HEAD_DIM + d].to(cutlass.Float32) * QKV_SCALE
        s = s + x * x
    s = s + cute.arch.shuffle_sync_bfly(s, 16)
    s = s + cute.arch.shuffle_sync_bfly(s, 8)
    s = s + cute.arch.shuffle_sync_bfly(s, 4)
    s = s + cute.arch.shuffle_sync_bfly(s, 2)
    s = s + cute.arch.shuffle_sync_bfly(s, 1)
    inv = cute.rsqrt(s / HEAD_DIM + EPSILON)
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        d = i * 32 + th
        x = k_in[t, h * HEAD_DIM + d].to(cutlass.Float32) * QKV_SCALE
        n = x * inv * (1.0 + k_norm_weight[d].to(cutlass.Float32))
        out_v = n
        if d < ROTARY_HALF:
            x2 = k_in[t, h * HEAD_DIM + (d + ROTARY_HALF)].to(cutlass.Float32) * QKV_SCALE
            n2 = x2 * inv * (1.0 + k_norm_weight[d + ROTARY_HALF].to(cutlass.Float32))
            c = cos[t, d].to(cutlass.Float32)
            si = sin[t, d].to(cutlass.Float32)
            out_v = n * c - n2 * si
        elif d < ROTARY_DIM:
            c = cos[t, d - ROTARY_HALF].to(cutlass.Float32)
            si = sin[t, d - ROTARY_HALF].to(cutlass.Float32)
            xp = k_in[t, h * HEAD_DIM + (d - ROTARY_HALF)].to(cutlass.Float32) * QKV_SCALE
            np = xp * inv * (1.0 + k_norm_weight[d - ROTARY_HALF].to(cutlass.Float32))
            out_v = n * c + np * si
        key_out[t, h * HEAD_DIM + d] = cutlass.Float8E4M3FN(out_v / QKV_SCALE)


@cute.kernel
def attention_kernel(
    query: cute.Tensor,
    key: cute.Tensor,
    v: cute.Tensor,
    q_gate: cute.Tensor,
    score_ws: cute.Tensor,
    prob_ws: cute.Tensor,
    context_out: cute.Tensor,
):
    # grid = (NUM_HEADS, TOKENS); warp per (query head, token)
    qh, t, _ = cute.arch.block_idx()
    th, _, _ = cute.arch.thread_idx()
    kv = qh // HEADS_PER_KV
    q_base = qh * HEAD_DIM
    kv_base = kv * HEAD_DIM
    qg_base = qh * 2 * HEAD_DIM

    # Pass 1: raw scores to workspace, running max/sum for causal softmax.
    max_s = cutlass.Float32(-1.0e30)
    sum_e = cutlass.Float32(0.0)
    for s in cutlass.range(TOKENS):
        if s <= t:
            part = cutlass.Float32(0.0)
            for i in cutlass.range_constexpr(HEAD_DIM // 32):
                d = i * 32 + th
                qr = query[t, q_base + d].to(cutlass.Float32) * QKV_SCALE
                kr = key[s, kv_base + d].to(cutlass.Float32) * QKV_SCALE
                part = part + qr * kr
            part = part + cute.arch.shuffle_sync_bfly(part, 16)
            part = part + cute.arch.shuffle_sync_bfly(part, 8)
            part = part + cute.arch.shuffle_sync_bfly(part, 4)
            part = part + cute.arch.shuffle_sync_bfly(part, 2)
            part = part + cute.arch.shuffle_sync_bfly(part, 1)
            score = part / 16.0
            score_ws[t, qh * TOKENS + s] = score
            new_max = max_s
            if score > max_s:
                new_max = score
            sum_e = sum_e * cute.exp(max_s - new_max) + cute.exp(score - new_max)
            max_s = new_max

    # Pass 2: probabilities + context over the dims this lane owns, gate.
    ctx = cute.make_rmem_tensor((HEAD_DIM // 32,), cutlass.Float32)
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        ctx[i] = cutlass.Float32(0.0)
    for s in cutlass.range(TOKENS):
        if s <= t:
            sc = score_ws[t, qh * TOKENS + s].to(cutlass.Float32)
            p = cute.exp(sc - max_s) / sum_e
            for i in cutlass.range_constexpr(HEAD_DIM // 32):
                d = i * 32 + th
                ctx[i] = ctx[i] + p * v[s, kv_base + d].to(cutlass.Float32) * QKV_SCALE
    for i in cutlass.range_constexpr(HEAD_DIM // 32):
        d = i * 32 + th
        gr = q_gate[t, qg_base + HEAD_DIM + d].to(cutlass.Float32) * QKV_SCALE
        sig = 1.0 / (1.0 + cute.exp(-gr))
        context_out[t, q_base + d] = cutlass.Float8E4M3FN(
            ctx[i] * sig / CONTEXT_SCALE
        )


@cute.kernel
def post_norm_kernel(residual: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
    t, _, _ = cute.arch.block_idx()
    th, _, _ = cute.arch.thread_idx()
    s = cutlass.Float32(0.0)
    for i in cutlass.range_constexpr(HIDDEN_SIZE // 32):
        x = residual[t, i * 32 + th].to(cutlass.Float32)
        s = s + x * x
    s = s + cute.arch.shuffle_sync_bfly(s, 16)
    s = s + cute.arch.shuffle_sync_bfly(s, 8)
    s = s + cute.arch.shuffle_sync_bfly(s, 4)
    s = s + cute.arch.shuffle_sync_bfly(s, 2)
    s = s + cute.arch.shuffle_sync_bfly(s, 1)
    inv = cute.rsqrt(s / HIDDEN_SIZE + EPSILON)
    for i in cutlass.range_constexpr(HIDDEN_SIZE // 32):
        idx = i * 32 + th
        x = residual[t, idx].to(cutlass.Float32)
        n = x * inv * (1.0 + weight[idx].to(cutlass.Float32))
        output[t, idx] = cutlass.Float8E4M3FN(n / NORM_SCALE)


SWIGLU_THREADS = 256


@cute.kernel
def swiglu_kernel(gate: cute.Tensor, up: cute.Tensor, mlp_out: cute.Tensor):
    # grid = (TOKENS,); block streams over columns.
    t, _, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    for i in cutlass.range(INTERMEDIATE_SIZE // SWIGLU_THREADS):
        j = i * SWIGLU_THREADS + tid
        g = gate[t, j].to(cutlass.Float32) * MLP_PROJECTION_SCALE
        u = up[t, j].to(cutlass.Float32) * MLP_PROJECTION_SCALE
        silu = g / (1.0 + cute.exp(-g))
        mlp_out[t, j] = cutlass.Float8E4M3FN(silu * u / MLP_ACTIVATION_SCALE)


@cute.jit
def qwen35_full_attention_block(
    hidden: cute.Tensor,
    input_norm_weight: cute.Tensor,
    q_norm_weight: cute.Tensor,
    k_norm_weight: cute.Tensor,
    q_gate_weight: cute.Tensor,
    k_weight: cute.Tensor,
    v_weight: cute.Tensor,
    out_weight: cute.Tensor,
    post_norm_weight: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    down_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    norm1_workspace: cute.Tensor,
    q_gate_workspace: cute.Tensor,
    k_workspace: cute.Tensor,
    v_workspace: cute.Tensor,
    query_workspace: cute.Tensor,
    key_workspace: cute.Tensor,
    score_workspace: cute.Tensor,
    probability_workspace: cute.Tensor,
    context_workspace: cute.Tensor,
    residual_workspace: cute.Tensor,
    norm2_workspace: cute.Tensor,
    gate_workspace: cute.Tensor,
    up_workspace: cute.Tensor,
    mlp_workspace: cute.Tensor,
    output: cute.Tensor,
):
    rms_norm_kernel(hidden, input_norm_weight, norm1_workspace).launch(
        grid=(TOKENS, 1, 1), block=(32, 1, 1)
    )
    fp8_gemm(
        norm1_workspace, q_gate_weight, q_gate_workspace,
        q_gate_workspace, S_QKV, 0,
    )
    fp8_gemm(
        norm1_workspace, k_weight, k_workspace,
        k_workspace, S_QKV, 0,
    )
    fp8_gemm(
        norm1_workspace, v_weight, v_workspace,
        v_workspace, S_QKV, 0,
    )
    q_headnorm_kernel(
        q_gate_workspace, q_norm_weight, cos, sin, query_workspace
    ).launch(grid=(NUM_HEADS, TOKENS, 1), block=(32, 1, 1))
    k_headnorm_kernel(
        k_workspace, k_norm_weight, cos, sin, key_workspace
    ).launch(grid=(NUM_KV_HEADS, TOKENS, 1), block=(32, 1, 1))
    attention_kernel(
        query_workspace, key_workspace, v_workspace, q_gate_workspace,
        score_workspace, probability_workspace, context_workspace,
    ).launch(grid=(NUM_HEADS, TOKENS, 1), block=(32, 1, 1))
    fp8_gemm(
        context_workspace, out_weight, residual_workspace,
        hidden, S_OUT, 1,
    )
    post_norm_kernel(residual_workspace, post_norm_weight, norm2_workspace).launch(
        grid=(TOKENS, 1, 1), block=(32, 1, 1)
    )
    fp8_gemm(
        norm2_workspace, gate_weight, gate_workspace,
        gate_workspace, S_MLP, 0,
    )
    fp8_gemm(
        norm2_workspace, up_weight, up_workspace,
        up_workspace, S_MLP, 0,
    )
    swiglu_kernel(gate_workspace, up_workspace, mlp_workspace).launch(
        grid=(TOKENS, 1, 1), block=(SWIGLU_THREADS, 1, 1)
    )
    fp8_gemm(
        mlp_workspace, down_weight, output,
        residual_workspace, S_DOWN, 2,
    )


class ModelNew:
    forward = staticmethod(qwen35_full_attention_block)
