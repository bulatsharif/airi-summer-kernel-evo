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
THREADS = 128
EPSILON = 1.0e-6
ATTN_SCALE = 1.0 / 16.0

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
NEG_INF = -1.0e30
MLP_J_TILES = 8
MLP_M_TILES = 8
MLP_J_SLICE = INTERMEDIATE_SIZE // MLP_J_TILES
MLP_M_SLICE = HIDDEN_SIZE // MLP_M_TILES
QKV_Q_TILES = 8
QKV_KV_TILES = 4
QKV_Q_SLICE = Q_GATE_SIZE // QKV_Q_TILES
QKV_KV_SLICE = KV_SIZE // QKV_KV_TILES
OUT_M_TILES = 8
OUT_M_SLICE = HIDDEN_SIZE // OUT_M_TILES


@cute.kernel
def rms_norm_kernel(hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
    bx = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    acc = F32(0.0)
    for n in cutlass.range(0, HIDDEN_SIZE):
        h = hidden[bx, n].to(F32) * INPUT_SCALE
        acc += h * h
    rstd = cute.rsqrt(acc / HIDDEN_SIZE + EPSILON)
    for n in cutlass.range(tx, HIDDEN_SIZE, THREADS):
        h = hidden[bx, n].to(F32) * INPUT_SCALE
        x = h * rstd * (F32(1.0) + weight[n])
        output[bx, n] = FP8_DTYPE(x / NORM_SCALE)


@cute.kernel
def qkv_q_kernel(
    norm1: cute.Tensor,
    q_gate_weight: cute.Tensor,
    q_gate_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    jt = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    j_start = jt * QKV_Q_SLICE
    for j in cutlass.range(j_start + tx, j_start + QKV_Q_SLICE, THREADS):
        acc = F32(0.0)
        for n in cutlass.range(0, HIDDEN_SIZE):
            acc += norm1[bx, n].to(F32) * NORM_SCALE * q_gate_weight[j, n].to(F32) * WEIGHT_H_SCALE
        q_gate_out[bx, j] = FP8_DTYPE(acc / QKV_SCALE)


@cute.kernel
def qkv_kv_kernel(
    norm1: cute.Tensor,
    kv_weight: cute.Tensor,
    kv_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    jt = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    j_start = jt * QKV_KV_SLICE
    for j in cutlass.range(j_start + tx, j_start + QKV_KV_SLICE, THREADS):
        acc = F32(0.0)
        for n in cutlass.range(0, HIDDEN_SIZE):
            acc += norm1[bx, n].to(F32) * NORM_SCALE * kv_weight[j, n].to(F32) * WEIGHT_H_SCALE
        kv_out[bx, j] = FP8_DTYPE(acc / QKV_SCALE)


@cute.kernel
def q_rope_norm_kernel(
    q_gate_ws: cute.Tensor,
    q_norm_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    query_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    for h in cutlass.range(0, NUM_HEADS):
        base = 2 * h * HEAD_DIM
        acc = F32(0.0)
        for d in cutlass.range(0, HEAD_DIM):
            v = q_gate_ws[bx, base + d].to(F32) * QKV_SCALE
            acc += v * v
        rstd = cute.rsqrt(acc / HEAD_DIM + EPSILON)
        for d in cutlass.range(tx, ROTARY_DIM // 2, THREADS):
            p = d + ROTARY_DIM // 2
            vd = q_gate_ws[bx, base + d].to(F32) * QKV_SCALE
            vp = q_gate_ws[bx, base + p].to(F32) * QKV_SCALE
            xd = vd * rstd * (F32(1.0) + q_norm_weight[d])
            xp = vp * rstd * (F32(1.0) + q_norm_weight[p])
            c = cos[bx, d]
            s = sin[bx, d]
            query_out[bx, h * HEAD_DIM + d] = FP8_DTYPE((xd * c - xp * s) / QKV_SCALE)
            query_out[bx, h * HEAD_DIM + p] = FP8_DTYPE((xd * s + xp * c) / QKV_SCALE)
        for d in cutlass.range(ROTARY_DIM + tx, HEAD_DIM, THREADS):
            vd = q_gate_ws[bx, base + d].to(F32) * QKV_SCALE
            xd = vd * rstd * (F32(1.0) + q_norm_weight[d])
            query_out[bx, h * HEAD_DIM + d] = FP8_DTYPE(xd / QKV_SCALE)


@cute.kernel
def k_rope_norm_kernel(
    k_ws: cute.Tensor,
    k_norm_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    key_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    for kh in cutlass.range(0, NUM_KV_HEADS):
        base = kh * HEAD_DIM
        acc = F32(0.0)
        for d in cutlass.range(0, HEAD_DIM):
            v = k_ws[bx, base + d].to(F32) * QKV_SCALE
            acc += v * v
        rstd = cute.rsqrt(acc / HEAD_DIM + EPSILON)
        for d in cutlass.range(tx, ROTARY_DIM // 2, THREADS):
            p = d + ROTARY_DIM // 2
            vd = k_ws[bx, base + d].to(F32) * QKV_SCALE
            vp = k_ws[bx, base + p].to(F32) * QKV_SCALE
            xd = vd * rstd * (F32(1.0) + k_norm_weight[d])
            xp = vp * rstd * (F32(1.0) + k_norm_weight[p])
            c = cos[bx, d]
            s = sin[bx, d]
            key_out[bx, base + d] = FP8_DTYPE((xd * c - xp * s) / QKV_SCALE)
            key_out[bx, base + p] = FP8_DTYPE((xd * s + xp * c) / QKV_SCALE)
        for d in cutlass.range(ROTARY_DIM + tx, HEAD_DIM, THREADS):
            vd = k_ws[bx, base + d].to(F32) * QKV_SCALE
            xd = vd * rstd * (F32(1.0) + k_norm_weight[d])
            key_out[bx, base + d] = FP8_DTYPE(xd / QKV_SCALE)


@cute.kernel
def attention_kernel(
    query_ws: cute.Tensor,
    key_ws: cute.Tensor,
    v_ws: cute.Tensor,
    q_gate_ws: cute.Tensor,
    score_ws: cute.Tensor,
    prob_ws: cute.Tensor,
    context_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    score_ptr = cute.arch.alloc_smem(F32, NUM_HEADS * TOKENS, 16)
    score_sh = cute.make_tensor(score_ptr, cute.make_layout(NUM_HEADS * TOKENS))
    if tx < NUM_HEADS:
        h = tx
        kh = h // (NUM_HEADS // NUM_KV_HEADS)
        mx = F32(NEG_INF)
        lse = F32(0.0)
        for kk in cutlass.range(0, bx + 1):
            s = F32(0.0)
            for dd in cutlass.range(0, HEAD_DIM):
                qv = query_ws[bx, h * HEAD_DIM + dd].to(F32) * QKV_SCALE
                kv = key_ws[kk, kh * HEAD_DIM + dd].to(F32) * QKV_SCALE
                s += qv * kv
            s = s * ATTN_SCALE
            score_sh[h * TOKENS + kk] = s
            if s > mx:
                lse = lse * cute.exp(mx - s) + F32(1.0)
                mx = s
            else:
                lse = lse + cute.exp(s - mx)
        for kk in cutlass.range(0, bx + 1):
            score_sh[h * TOKENS + kk] = cute.exp(score_sh[h * TOKENS + kk] - mx) / lse
    cute.arch.sync_threads()
    if tx < NUM_HEADS:
        h = tx
        kh = h // (NUM_HEADS // NUM_KV_HEADS)
        for d in cutlass.range(0, HEAD_DIM):
            acc = F32(0.0)
            for kk in cutlass.range(0, bx + 1):
                acc += score_sh[h * TOKENS + kk] * v_ws[kk, kh * HEAD_DIM + d].to(F32) * QKV_SCALE
            g = q_gate_ws[bx, (2 * h + 1) * HEAD_DIM + d].to(F32) * QKV_SCALE
            gs = F32(1.0) / (F32(1.0) + cute.exp(-g))
            context_out[bx, h * HEAD_DIM + d] = FP8_DTYPE(acc * gs / CONTEXT_SCALE)


@cute.kernel
def out_proj_kernel(
    context_ws: cute.Tensor,
    out_weight: cute.Tensor,
    hidden: cute.Tensor,
    residual_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    mt = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    m_start = mt * OUT_M_SLICE
    for m in cutlass.range(m_start + tx, m_start + OUT_M_SLICE, THREADS):
        acc = F32(0.0)
        for j in cutlass.range(0, QUERY_SIZE):
            acc += context_ws[bx, j].to(F32) * CONTEXT_SCALE * out_weight[m, j].to(F32) * WEIGHT_CONTEXT_SCALE
        res = hidden[bx, m].to(F32) * INPUT_SCALE + acc
        residual_out[bx, m] = res


@cute.kernel
def mlp_norm_kernel(
    residual_ws: cute.Tensor,
    post_norm_weight: cute.Tensor,
    norm2_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    tx = cute.arch.thread_idx()[0]
    acc = F32(0.0)
    for m in cutlass.range(0, HIDDEN_SIZE):
        v = residual_ws[bx, m]
        acc += v * v
    rstd = cute.rsqrt(acc / HIDDEN_SIZE + EPSILON)
    for m in cutlass.range(tx, HIDDEN_SIZE, THREADS):
        x = residual_ws[bx, m] * rstd * (F32(1.0) + post_norm_weight[m])
        norm2_out[bx, m] = FP8_DTYPE(x / NORM_SCALE)


@cute.kernel
def mlp_gateup_kernel(
    norm2_out: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    mlp_out: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    jt = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    j_start = jt * MLP_J_SLICE
    for j in cutlass.range(j_start + tx, j_start + MLP_J_SLICE, THREADS):
        ga = F32(0.0)
        ua = F32(0.0)
        for m in cutlass.range(0, HIDDEN_SIZE):
            x = norm2_out[bx, m].to(F32) * NORM_SCALE
            ga += x * gate_weight[j, m].to(F32) * WEIGHT_H_SCALE
            ua += x * up_weight[j, m].to(F32) * WEIGHT_H_SCALE
        gq = FP8_DTYPE(ga / MLP_PROJECTION_SCALE).to(F32) * MLP_PROJECTION_SCALE
        uq = FP8_DTYPE(ua / MLP_PROJECTION_SCALE).to(F32) * MLP_PROJECTION_SCALE
        gs = F32(1.0) / (F32(1.0) + cute.exp(-gq))
        silu = gq * gs
        mlp_out[bx, j] = FP8_DTYPE(silu * uq / MLP_ACTIVATION_SCALE)


@cute.kernel
def mlp_down_kernel(
    mlp_out: cute.Tensor,
    down_weight: cute.Tensor,
    residual_ws: cute.Tensor,
    output: cute.Tensor,
):
    bx = cute.arch.block_idx()[0]
    mt = cute.arch.block_idx()[1]
    tx = cute.arch.thread_idx()[0]
    m_start = mt * MLP_M_SLICE
    for m in cutlass.range(m_start + tx, m_start + MLP_M_SLICE, THREADS):
        acc = F32(0.0)
        for j in cutlass.range(0, INTERMEDIATE_SIZE):
            acc += mlp_out[bx, j].to(F32) * MLP_ACTIVATION_SCALE * down_weight[m, j].to(F32) * WEIGHT_MLP_SCALE
        output[bx, m] = residual_ws[bx, m] + acc


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
    grid = (TOKENS, 1, 1)
    block = (THREADS, 1, 1)

    rms_norm_kernel(hidden, input_norm_weight, norm1_workspace).launch(grid=grid, block=block)
    qkv_q_kernel(
        norm1_workspace, q_gate_weight, q_gate_workspace
    ).launch(grid=(TOKENS, QKV_Q_TILES, 1), block=block)
    qkv_kv_kernel(
        norm1_workspace, k_weight, k_workspace
    ).launch(grid=(TOKENS, QKV_KV_TILES, 1), block=block)
    qkv_kv_kernel(
        norm1_workspace, v_weight, v_workspace
    ).launch(grid=(TOKENS, QKV_KV_TILES, 1), block=block)
    q_rope_norm_kernel(
        q_gate_workspace, q_norm_weight, cos, sin, query_workspace
    ).launch(grid=grid, block=block)
    k_rope_norm_kernel(
        k_workspace, k_norm_weight, cos, sin, key_workspace
    ).launch(grid=grid, block=block)
    attention_kernel(
        query_workspace,
        key_workspace,
        v_workspace,
        q_gate_workspace,
        score_workspace,
        probability_workspace,
        context_workspace,
    ).launch(grid=grid, block=block)
    out_proj_kernel(
        context_workspace, out_weight, hidden, residual_workspace
    ).launch(grid=(TOKENS, OUT_M_TILES, 1), block=block)
    mlp_norm_kernel(
        residual_workspace, post_norm_weight, norm2_workspace
    ).launch(grid=grid, block=block)
    mlp_gateup_kernel(
        norm2_workspace, gate_weight, up_weight, mlp_workspace
    ).launch(grid=(TOKENS, MLP_J_TILES, 1), block=block)
    mlp_down_kernel(
        mlp_workspace, down_weight, residual_workspace, output
    ).launch(grid=(TOKENS, MLP_M_TILES, 1), block=block)


class ModelNew:
    forward = staticmethod(qwen35_full_attention_block)
