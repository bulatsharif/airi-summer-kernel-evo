import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 2560
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
INTERMEDIATE_SIZE = 9216
Q_GATE_SIZE = 2 * NUM_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
QUERY_SIZE = NUM_HEADS * HEAD_DIM
QG_STRIDE = 2 * HEAD_DIM
THREADS = 128
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
FP8_DTYPE = cutlass.Float8E4M3FN
F32 = cutlass.Float32

# Tiled-GEMM tuning. Each block handles BM output rows; the weight tile for
# [BM rows x BK K-columns] is staged into shared memory with coalesced global
# loads (consecutive threads read consecutive columns), then replayed from smem
# for the dot products. This removes the stride-K uncoalesced weight reads that
# dominated memory traffic.
BM = THREADS
BK = 128
ROW_STRIDE = BK + 4

# Number of threads per warp / warps per block (THREADS = 128 => 4 warps).
WARP_SIZE = 32
NUM_WARPS = THREADS // WARP_SIZE


@cute.kernel
def norm1_kernel(hidden, input_norm_weight, norm1_workspace):
    # One block per token; 128 threads cooperate on the token's RMS statistics
    # and normalization output. Reduces norm1 latency by parallelizing the
    # serial per-token reduction that dominated the previous scalar kernel.
    tid, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    smem = cute.make_tensor(
        cute.arch.alloc_smem(F32, NUM_WARPS, 4), cute.make_layout(NUM_WARPS)
    )
    acc = F32(0.0)
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        x = hidden[t, it * THREADS + tid].to(F32) * INPUT_SCALE
        acc = acc + x * x
    # Warp reduction: butterfly shuffle over descending offsets.
    v = acc
    for off in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, off)
    if lane == 0:
        smem[warp] = v
    cute.arch.sync_threads()
    # Reduce the NUM_WARPS per-warp partials in warp 0.
    if warp == 0:
        wv = smem[lane] if lane < NUM_WARPS else F32(0.0)
        for off in (2, 1):
            wv = wv + cute.arch.shuffle_sync_bfly(wv, off)
        if lane == 0:
            smem[0] = wv
    cute.arch.sync_threads()
    mean = smem[0] * (1.0 / float(HIDDEN_SIZE))
    inv = cute.rsqrt(mean + F32(EPSILON))
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        d = it * THREADS + tid
        x = hidden[t, d].to(F32) * INPUT_SCALE
        xn = x * inv * (F32(1.0) + input_norm_weight[d].to(F32))
        norm1_workspace[t, d] = FP8_DTYPE(xn * (1.0 / NORM_SCALE))


QG_TILES = Q_GATE_SIZE // BM  # 64
KV_TILES = KV_SIZE // BM       # 8
QKV_TOTAL_TILES = QG_TILES + 2 * KV_TILES  # 80
QKV_CHUNKS = HIDDEN_SIZE // BK  # 20


@cute.kernel
def qkv_kernel(
    norm1_workspace,
    q_gate_weight,
    k_weight,
    v_weight,
    q_gate_workspace,
    k_workspace,
    v_workspace,
):
    t, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem_n = cute.make_tensor(
        cute.arch.alloc_smem(F32, HIDDEN_SIZE, 4), cute.make_layout(HIDDEN_SIZE)
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, BM * ROW_STRIDE, 4),
        cute.make_layout(BM * ROW_STRIDE),
    )
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        smem_n[it * THREADS + tid] = norm1_workspace[t, it * THREADS + tid].to(F32)
    cute.arch.sync_threads()
    w = q_gate_weight
    out = q_gate_workspace
    o_base = 0
    if tile < QG_TILES:
        w = q_gate_weight
        out = q_gate_workspace
        o_base = tile * BM
    elif tile < QG_TILES + KV_TILES:
        w = k_weight
        out = k_workspace
        o_base = (tile - QG_TILES) * BM
    else:
        w = v_weight
        out = v_workspace
        o_base = (tile - QG_TILES - KV_TILES) * BM
    acc = F32(0.0)
    for kb in cutlass.range(QKV_CHUNKS):
        k0 = kb * BK
        for r in cutlass.range(BM):
            smem_w[r * ROW_STRIDE + tid] = w[o_base + r, k0 + tid]
        cute.arch.sync_threads()
        for d in cutlass.range_constexpr(BK):
            acc = acc + smem_n[k0 + d] * smem_w[tid * ROW_STRIDE + d].to(F32)
        cute.arch.sync_threads()
    out[t, o_base + tid] = FP8_DTYPE(acc * WEIGHT_H_SCALE)


@cute.kernel
def rotary_q_kernel(q_gate_workspace, q_norm_weight, cos, sin, query_workspace):
    idx, _, _ = cute.arch.block_idx()
    t = idx // NUM_HEADS
    h = idx % NUM_HEADS
    acc = F32(0.0)
    for it in cutlass.range(HEAD_DIM):
        x = q_gate_workspace[t, h * QG_STRIDE + it].to(F32) * QKV_SCALE
        acc = acc + x * x
    mean = acc * (1.0 / float(HEAD_DIM))
    inv = cute.rsqrt(mean + F32(EPSILON))
    half = ROTARY_DIM // 2
    for it in cutlass.range(half):
        d = it
        xa = q_gate_workspace[t, h * QG_STRIDE + d].to(F32) * QKV_SCALE
        xb = q_gate_workspace[t, h * QG_STRIDE + d + half].to(F32) * QKV_SCALE
        a = xa * inv * (F32(1.0) + q_norm_weight[d].to(F32))
        b = xb * inv * (F32(1.0) + q_norm_weight[d + half].to(F32))
        c = cos[t, d].to(F32)
        s = sin[t, d].to(F32)
        c2 = cos[t, d + half].to(F32)
        s2 = sin[t, d + half].to(F32)
        query_workspace[t, h * HEAD_DIM + d] = FP8_DTYPE((a * c - b * s) * (1.0 / QKV_SCALE))
        query_workspace[t, h * HEAD_DIM + d + half] = FP8_DTYPE((a * s2 + b * c2) * (1.0 / QKV_SCALE))
    for it in cutlass.range(HEAD_DIM - ROTARY_DIM):
        d = ROTARY_DIM + it
        x = q_gate_workspace[t, h * QG_STRIDE + d].to(F32) * QKV_SCALE
        xn = x * inv * (F32(1.0) + q_norm_weight[d].to(F32))
        query_workspace[t, h * HEAD_DIM + d] = FP8_DTYPE(xn * (1.0 / QKV_SCALE))


@cute.kernel
def rotary_k_kernel(k_workspace, k_norm_weight, cos, sin, key_workspace):
    idx, _, _ = cute.arch.block_idx()
    t = idx // NUM_KV_HEADS
    kv = idx % NUM_KV_HEADS
    acc = F32(0.0)
    for it in cutlass.range(HEAD_DIM):
        x = k_workspace[t, kv * HEAD_DIM + it].to(F32) * QKV_SCALE
        acc = acc + x * x
    mean = acc * (1.0 / float(HEAD_DIM))
    inv = cute.rsqrt(mean + F32(EPSILON))
    half = ROTARY_DIM // 2
    for it in cutlass.range(half):
        d = it
        xa = k_workspace[t, kv * HEAD_DIM + d].to(F32) * QKV_SCALE
        xb = k_workspace[t, kv * HEAD_DIM + d + half].to(F32) * QKV_SCALE
        a = xa * inv * (F32(1.0) + k_norm_weight[d].to(F32))
        b = xb * inv * (F32(1.0) + k_norm_weight[d + half].to(F32))
        c = cos[t, d].to(F32)
        s = sin[t, d].to(F32)
        c2 = cos[t, d + half].to(F32)
        s2 = sin[t, d + half].to(F32)
        key_workspace[t, kv * HEAD_DIM + d] = FP8_DTYPE((a * c - b * s) * (1.0 / QKV_SCALE))
        key_workspace[t, kv * HEAD_DIM + d + half] = FP8_DTYPE((a * s2 + b * c2) * (1.0 / QKV_SCALE))
    for it in cutlass.range(HEAD_DIM - ROTARY_DIM):
        d = ROTARY_DIM + it
        x = k_workspace[t, kv * HEAD_DIM + d].to(F32) * QKV_SCALE
        xn = x * inv * (F32(1.0) + k_norm_weight[d].to(F32))
        key_workspace[t, kv * HEAD_DIM + d] = FP8_DTYPE(xn * (1.0 / QKV_SCALE))


@cute.kernel
def attention_kernel(
    query_workspace,
    key_workspace,
    v_workspace,
    q_gate_workspace,
    context_workspace,
):
    idx, _, _ = cute.arch.block_idx()
    t = idx // NUM_HEADS
    h = idx % NUM_HEADS
    kv = h // (NUM_HEADS // NUM_KV_HEADS)
    tid, _, _ = cute.arch.thread_idx()
    smem_q = cute.make_tensor(
        cute.arch.alloc_smem(F32, HEAD_DIM, 4), cute.make_layout(HEAD_DIM)
    )
    smem_s = cute.make_tensor(
        cute.arch.alloc_smem(F32, TOKENS, 4), cute.make_layout(TOKENS)
    )
    smem_p = cute.make_tensor(
        cute.arch.alloc_smem(F32, TOKENS, 4), cute.make_layout(TOKENS)
    )
    smem_q[tid] = query_workspace[t, h * HEAD_DIM + tid].to(F32) * QKV_SCALE
    smem_q[tid + THREADS] = (
        query_workspace[t, h * HEAD_DIM + tid + THREADS].to(F32) * QKV_SCALE
    )
    cute.arch.sync_threads()
    if tid <= t:
        s = F32(0.0)
        for dm in cutlass.range(HEAD_DIM):
            s = s + smem_q[dm] * key_workspace[tid, kv * HEAD_DIM + dm].to(F32) * QKV_SCALE
        smem_s[tid] = s / F32(16.0)
    else:
        smem_s[tid] = F32(-1000000.0)
    cute.arch.sync_threads()
    if tid == 0:
        maxs = smem_s[0]
        for t2 in cutlass.range(t + 1):
            if smem_s[t2] > maxs:
                maxs = smem_s[t2]
        sume = F32(0.0)
        for t2 in cutlass.range(t + 1):
            sume = sume + cute.exp(smem_s[t2] - maxs)
        for t2 in cutlass.range(t + 1):
            smem_p[t2] = cute.exp(smem_s[t2] - maxs) / sume
    cute.arch.sync_threads()
    for dd in cutlass.range(HEAD_DIM // THREADS):
        d = dd * THREADS + tid
        ctx = F32(0.0)
        for t2 in cutlass.range(t + 1):
            ctx = ctx + smem_p[t2] * v_workspace[t2, kv * HEAD_DIM + d].to(F32) * QKV_SCALE
        gv = q_gate_workspace[t, h * QG_STRIDE + HEAD_DIM + d].to(F32) * QKV_SCALE
        sg = F32(1.0) / (F32(1.0) + cute.exp(F32(0.0) - gv))
        gated = ctx * sg
        context_workspace[t, h * HEAD_DIM + d] = FP8_DTYPE(gated * (1.0 / CONTEXT_SCALE))


OUT_TILES = HIDDEN_SIZE // BM  # 20
OUT_CHUNKS = QUERY_SIZE // BK   # 32


@cute.kernel
def out_proj_kernel(
    context_workspace,
    out_weight,
    hidden,
    residual_workspace,
):
    t, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    o_base = tile * BM
    o = o_base + tid
    smem_c = cute.make_tensor(
        cute.arch.alloc_smem(F32, QUERY_SIZE, 4), cute.make_layout(QUERY_SIZE)
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, BM * ROW_STRIDE, 4),
        cute.make_layout(BM * ROW_STRIDE),
    )
    for it in cutlass.range(QUERY_SIZE // THREADS):
        smem_c[it * THREADS + tid] = context_workspace[t, it * THREADS + tid].to(F32)
    cute.arch.sync_threads()
    acc = F32(0.0)
    for kb in cutlass.range(OUT_CHUNKS):
        k0 = kb * BK
        for r in cutlass.range(BM):
            smem_w[r * ROW_STRIDE + tid] = out_weight[o_base + r, k0 + tid]
        cute.arch.sync_threads()
        for d in cutlass.range_constexpr(BK):
            acc = acc + smem_c[k0 + d] * smem_w[tid * ROW_STRIDE + d].to(F32)
        cute.arch.sync_threads()
    attn = acc * CONTEXT_SCALE * WEIGHT_CONTEXT_SCALE
    hr = hidden[t, o].to(F32) * INPUT_SCALE
    residual_workspace[t, o] = hr + attn


@cute.kernel
def norm2_kernel(residual_workspace, post_norm_weight, norm2_workspace):
    # Same parallel-block reduction strategy as norm1_kernel.
    tid, _, _ = cute.arch.thread_idx()
    t, _, _ = cute.arch.block_idx()
    lane = cute.arch.lane_idx()
    warp = cute.arch.warp_idx()
    smem = cute.make_tensor(
        cute.arch.alloc_smem(F32, NUM_WARPS, 4), cute.make_layout(NUM_WARPS)
    )
    acc = F32(0.0)
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        x = residual_workspace[t, it * THREADS + tid].to(F32)
        acc = acc + x * x
    v = acc
    for off in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, off)
    if lane == 0:
        smem[warp] = v
    cute.arch.sync_threads()
    if warp == 0:
        wv = smem[lane] if lane < NUM_WARPS else F32(0.0)
        for off in (2, 1):
            wv = wv + cute.arch.shuffle_sync_bfly(wv, off)
        if lane == 0:
            smem[0] = wv
    cute.arch.sync_threads()
    mean = smem[0] * (1.0 / float(HIDDEN_SIZE))
    inv = cute.rsqrt(mean + F32(EPSILON))
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        d = it * THREADS + tid
        x = residual_workspace[t, d].to(F32)
        xn = x * inv * (F32(1.0) + post_norm_weight[d].to(F32))
        norm2_workspace[t, d] = FP8_DTYPE(xn * (1.0 / NORM_SCALE))


MLP_TILES = INTERMEDIATE_SIZE // BM  # 72
MLP_CHUNKS = HIDDEN_SIZE // BK


@cute.kernel
def mlp_proj_kernel(norm2_workspace, gate_weight, up_weight, gate_workspace, up_workspace):
    t, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    smem_n = cute.make_tensor(
        cute.arch.alloc_smem(F32, HIDDEN_SIZE, 4), cute.make_layout(HIDDEN_SIZE)
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, BM * ROW_STRIDE, 4),
        cute.make_layout(BM * ROW_STRIDE),
    )
    for it in cutlass.range(HIDDEN_SIZE // THREADS):
        smem_n[it * THREADS + tid] = norm2_workspace[t, it * THREADS + tid].to(F32)
    cute.arch.sync_threads()
    w = gate_weight
    out = gate_workspace
    o_base = 0
    if tile < MLP_TILES:
        w = gate_weight
        out = gate_workspace
        o_base = tile * BM
    else:
        w = up_weight
        out = up_workspace
        o_base = (tile - MLP_TILES) * BM
    acc = F32(0.0)
    for kb in cutlass.range(MLP_CHUNKS):
        k0 = kb * BK
        for r in cutlass.range(BM):
            smem_w[r * ROW_STRIDE + tid] = w[o_base + r, k0 + tid]
        cute.arch.sync_threads()
        for d in cutlass.range_constexpr(BK):
            acc = acc + smem_n[k0 + d] * smem_w[tid * ROW_STRIDE + d].to(F32)
        cute.arch.sync_threads()
    out[t, o_base + tid] = FP8_DTYPE(acc * WEIGHT_H_SCALE)


@cute.kernel
def swiglu_kernel(gate_workspace, up_workspace, mlp_workspace):
    t, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    o = tile * THREADS + tid
    g = gate_workspace[t, o].to(F32) * MLP_PROJECTION_SCALE
    u = up_workspace[t, o].to(F32) * MLP_PROJECTION_SCALE
    sg = F32(1.0) / (F32(1.0) + cute.exp(F32(0.0) - g))
    act = g * sg * u
    mlp_workspace[t, o] = FP8_DTYPE(act * (1.0 / MLP_ACTIVATION_SCALE))


DOWN_CHUNKS = INTERMEDIATE_SIZE // BK  # 72


@cute.kernel
def down_proj_kernel(mlp_workspace, down_weight, residual_workspace, output):
    t, tile, _ = cute.arch.block_idx()
    tid, _, _ = cute.arch.thread_idx()
    o_base = tile * BM
    o = o_base + tid
    smem_m = cute.make_tensor(
        cute.arch.alloc_smem(F32, INTERMEDIATE_SIZE, 4), cute.make_layout(INTERMEDIATE_SIZE)
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, BM * ROW_STRIDE, 4),
        cute.make_layout(BM * ROW_STRIDE),
    )
    for it in cutlass.range(INTERMEDIATE_SIZE // THREADS):
        smem_m[it * THREADS + tid] = mlp_workspace[t, it * THREADS + tid].to(F32)
    cute.arch.sync_threads()
    acc = F32(0.0)
    for kb in cutlass.range(DOWN_CHUNKS):
        k0 = kb * BK
        for r in cutlass.range(BM):
            smem_w[r * ROW_STRIDE + tid] = down_weight[o_base + r, k0 + tid]
        cute.arch.sync_threads()
        for d in cutlass.range_constexpr(BK):
            acc = acc + smem_m[k0 + d] * smem_w[tid * ROW_STRIDE + d].to(F32)
        cute.arch.sync_threads()
    mlp_real = acc * MLP_ACTIVATION_SCALE * WEIGHT_MLP_SCALE
    res = residual_workspace[t, o].to(F32)
    output[t, o] = res + mlp_real


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
    norm1_kernel(hidden, input_norm_weight, norm1_workspace).launch(
        grid=(TOKENS, 1, 1), block=(THREADS, 1, 1)
    )
    qkv_kernel(
        norm1_workspace,
        q_gate_weight,
        k_weight,
        v_weight,
        q_gate_workspace,
        k_workspace,
        v_workspace,
    ).launch(grid=(TOKENS, QKV_TOTAL_TILES, 1), block=(THREADS, 1, 1))
    rotary_q_kernel(q_gate_workspace, q_norm_weight, cos, sin, query_workspace).launch(
        grid=(TOKENS * NUM_HEADS, 1, 1), block=(1, 1, 1)
    )
    rotary_k_kernel(k_workspace, k_norm_weight, cos, sin, key_workspace).launch(
        grid=(TOKENS * NUM_KV_HEADS, 1, 1), block=(1, 1, 1)
    )
    attention_kernel(
        query_workspace,
        key_workspace,
        v_workspace,
        q_gate_workspace,
        context_workspace,
    ).launch(grid=(TOKENS * NUM_HEADS, 1, 1), block=(THREADS, 1, 1))
    out_proj_kernel(
        context_workspace, out_weight, hidden, residual_workspace
    ).launch(grid=(TOKENS, OUT_TILES, 1), block=(THREADS, 1, 1))
    norm2_kernel(residual_workspace, post_norm_weight, norm2_workspace).launch(
        grid=(TOKENS, 1, 1), block=(THREADS, 1, 1)
    )
    mlp_proj_kernel(
        norm2_workspace, gate_weight, up_weight, gate_workspace, up_workspace
    ).launch(grid=(TOKENS, 2 * MLP_TILES, 1), block=(THREADS, 1, 1))
    swiglu_kernel(gate_workspace, up_workspace, mlp_workspace).launch(
        grid=(TOKENS, MLP_TILES, 1), block=(THREADS, 1, 1)
    )
    down_proj_kernel(mlp_workspace, down_weight, residual_workspace, output).launch(
        grid=(TOKENS, OUT_TILES, 1), block=(THREADS, 1, 1)
    )


class ModelNew:
    forward = staticmethod(qwen35_full_attention_block)
