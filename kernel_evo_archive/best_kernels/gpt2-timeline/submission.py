import cutlass
import cutlass.cute as cute
import cutlass.utils as utils

TOKENS = 128
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
MLP_SIZE = 3072
QKV_SIZE = 3 * HIDDEN_SIZE
THREADS = 128
EPSILON = 1.0e-5

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = MLP_SIZE ** -0.5 / 448.0

S_QKV = NORM_SCALE * WEIGHT_H_SCALE
S_OUT = CONTEXT_SCALE * WEIGHT_H_SCALE
S_FC = NORM_SCALE * WEIGHT_H_SCALE
S_PROJ = MLP_SCALE * WEIGHT_MLP_SCALE
SCORE_SCALE = QKV_SCALE * QKV_SCALE / 8.0
CONTEXT_RATIO = QKV_SCALE / CONTEXT_SCALE

FP8_DTYPE = cutlass.Float8E4M3FN

# GEMM tiling. Smaller N_TILE gives more CTAs so the whole 148-SM B300 is
# covered; each CTA handles an M_TILE x N_TILE output tile.
M_TILE = 16
N_TILE = 32
KT = 256
TM = 2
TN = 2
NROWG = M_TILE // TM   # 8
NCOLG = N_TILE // TN   # 16


@cute.jit
def gelu_new(x):
    c = cutlass.Float32(0.7978845608028654)  # sqrt(2/pi)
    t = x + cutlass.Float32(0.044715) * x * x * x
    return cutlass.Float32(0.5) * x * (cutlass.Float32(1.0) + cute.tanh(c * t))


@cute.jit
def layernorm_stats(hidden, token, tid, smem, in_scale):
    s = cutlass.Float32(0.0)
    ss = cutlass.Float32(0.0)
    for t in cutlass.range_constexpr(HIDDEN_SIZE // THREADS):
        f = t * THREADS + tid
        x = hidden[token, f].to(cutlass.Float32) * cutlass.Float32(in_scale)
        s = s + x
        ss = ss + x * x
    smem[tid] = s
    smem[THREADS + tid] = ss
    cute.arch.sync_threads()
    total = cutlass.Float32(0.0)
    total_ss = cutlass.Float32(0.0)
    for j in cutlass.range_constexpr(THREADS):
        total = total + smem[j]
    for j in cutlass.range_constexpr(THREADS):
        total_ss = total_ss + smem[THREADS + j]
    mean = total / cutlass.Float32(HIDDEN_SIZE)
    var = total_ss / cutlass.Float32(HIDDEN_SIZE) - mean * mean
    rstd = cute.rsqrt(var + cutlass.Float32(EPSILON))
    return mean, rstd


@cute.jit
def gemm_stage_and_compute(a, w, sA, sB, tid, m_base, n_base, K):
    """Stage A[M_TILE,KT] and B[N_TILE,KT] tiles into shared memory, register
    blocked accumulation over K. Returns per-thread accumulator (TM x TN)."""
    row_g = tid // NCOLG
    col_g = tid % NCOLG
    acc = [
        [cutlass.Float32(0.0) for _ in range(TN)] for _ in range(TM)
    ]
    for kt in cutlass.range(K // KT):
        k0 = kt * KT
        # load A tile, coalesced along K (stride-1 within a row)
        for i in cutlass.range_constexpr(M_TILE * KT // THREADS):
            idx = i * THREADS + tid
            m = idx // KT
            kk = idx % KT
            sA[m, kk] = a[m_base + m, k0 + kk]
        # load B tile, coalesced along K
        for i in cutlass.range_constexpr(N_TILE * KT // THREADS):
            idx = i * THREADS + tid
            n = idx // KT
            kk = idx % KT
            sB[n, kk] = w[n_base + n, k0 + kk]
        cute.arch.sync_threads()
        for kk in cutlass.range(KT):
            b0 = sB[col_g * TN + 0, kk].to(cutlass.Float32)
            b1 = sB[col_g * TN + 1, kk].to(cutlass.Float32)
            for i in cutlass.range_constexpr(TM):
                av = sA[row_g * TM + i, kk].to(cutlass.Float32)
                acc[i][0] = acc[i][0] + av * b0
                acc[i][1] = acc[i][1] + av * b1
        cute.arch.sync_threads()
    return acc, row_g, col_g


@cute.kernel
def ln1_kernel(
    hidden: cute.Tensor,
    ln1_weight: cute.Tensor,
    ln1_bias: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    red = alloc.allocate_tensor(
        cutlass.Float32, cute.make_layout(2 * THREADS), 4
    )
    mean, rstd = layernorm_stats(hidden, token, tid, red, INPUT_SCALE)
    for t in cutlass.range_constexpr(HIDDEN_SIZE // THREADS):
        f = t * THREADS + tid
        x = hidden[token, f].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
        norm = (x - mean) * rstd * ln1_weight[f] + ln1_bias[f]
        out[token, f] = FP8_DTYPE(norm / cutlass.Float32(NORM_SCALE))


@cute.kernel
def qkv_gemm_kernel(
    a: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    nbx, nby, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    sA = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((M_TILE, KT)), 128
    )
    sB = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((N_TILE, KT)), 128
    )
    m_base = nby * M_TILE
    n_base = nbx * N_TILE
    acc, row_g, col_g = gemm_stage_and_compute(
        a, w, sA, sB, tid, m_base, n_base, HIDDEN_SIZE
    )
    for i in cutlass.range_constexpr(TM):
        m = m_base + row_g * TM + i
        for j in cutlass.range_constexpr(TN):
            n = n_base + col_g * TN + j
            val = acc[i][j] * cutlass.Float32(S_QKV) + bias[n].to(
                cutlass.Float32
            )
            out[m, n] = FP8_DTYPE(val / cutlass.Float32(QKV_SCALE))


@cute.kernel
def attention_kernel(qkv: cute.Tensor, context: cute.Tensor):
    h, i, _ = cute.arch.block_idx()
    j = cute.arch.thread_idx()[0]
    alloc = utils.SmemAllocator()
    score = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(TOKENS), 4)
    prob = alloc.allocate_tensor(cutlass.Float32, cute.make_layout(TOKENS), 4)
    base = h * HEAD_DIM
    acc = cutlass.Float32(0.0)
    for d in cutlass.range_constexpr(HEAD_DIM):
        acc = acc + qkv[i, base + d].to(cutlass.Float32) * qkv[
            j, HIDDEN_SIZE + base + d
        ].to(cutlass.Float32)
    sval = acc * cutlass.Float32(SCORE_SCALE)
    causal = sval if (j <= i) else cutlass.Float32(-3.0e38)
    score[j] = causal
    cute.arch.sync_threads()
    mx = cutlass.Float32(-3.0e38)
    for jj in cutlass.range_constexpr(TOKENS):
        vv = score[jj]
        mx = vv if (vv > mx) else mx
    e = cute.exp(causal - mx)
    prob[j] = e
    cute.arch.sync_threads()
    tot = cutlass.Float32(0.0)
    for jj in cutlass.range_constexpr(TOKENS):
        tot = tot + prob[jj]
    if j < HEAD_DIM:
        cacc = cutlass.Float32(0.0)
        for kk in cutlass.range_constexpr(TOKENS):
            cacc = cacc + prob[kk] * qkv[
                kk, 2 * HIDDEN_SIZE + base + j
            ].to(cutlass.Float32)
        cacc = cacc / tot
        context[i, base + j] = FP8_DTYPE(cacc * cutlass.Float32(CONTEXT_RATIO))


@cute.kernel
def out_gemm_kernel(
    a: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    hidden: cute.Tensor,
    residual: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    nbx, nby, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    sA = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((M_TILE, KT)), 128
    )
    sB = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((N_TILE, KT)), 128
    )
    m_base = nby * M_TILE
    n_base = nbx * N_TILE
    acc, row_g, col_g = gemm_stage_and_compute(
        a, w, sA, sB, tid, m_base, n_base, HIDDEN_SIZE
    )
    for i in cutlass.range_constexpr(TM):
        m = m_base + row_g * TM + i
        for j in cutlass.range_constexpr(TN):
            n = n_base + col_g * TN + j
            residual[m, n] = (
                hidden[m, n].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
                + acc[i][j] * cutlass.Float32(S_OUT)
                + bias[n].to(cutlass.Float32)
            )


@cute.kernel
def ln2_kernel(
    residual: cute.Tensor,
    ln2_weight: cute.Tensor,
    ln2_bias: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    token, _, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    red = alloc.allocate_tensor(
        cutlass.Float32, cute.make_layout(2 * THREADS), 4
    )
    mean, rstd = layernorm_stats(residual, token, tid, red, 1.0)
    for t in cutlass.range_constexpr(HIDDEN_SIZE // THREADS):
        f = t * THREADS + tid
        x = residual[token, f].to(cutlass.Float32)
        norm = (x - mean) * rstd * ln2_weight[f] + ln2_bias[f]
        out[token, f] = FP8_DTYPE(norm / cutlass.Float32(NORM_SCALE))


@cute.kernel
def fc_gelu_kernel(
    a: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    nbx, nby, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    sA = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((M_TILE, KT)), 128
    )
    sB = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((N_TILE, KT)), 128
    )
    m_base = nby * M_TILE
    n_base = nbx * N_TILE
    acc, row_g, col_g = gemm_stage_and_compute(
        a, w, sA, sB, tid, m_base, n_base, HIDDEN_SIZE
    )
    for i in cutlass.range_constexpr(TM):
        m = m_base + row_g * TM + i
        for j in cutlass.range_constexpr(TN):
            n = n_base + col_g * TN + j
            g = gelu_new(
                acc[i][j] * cutlass.Float32(S_FC)
                + bias[n].to(cutlass.Float32)
            )
            out[m, n] = FP8_DTYPE(g / cutlass.Float32(MLP_SCALE))


@cute.kernel
def proj_gemm_kernel(
    a: cute.Tensor,
    w: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
    out: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    nbx, nby, _ = cute.arch.block_idx()
    alloc = utils.SmemAllocator()
    sA = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((M_TILE, KT)), 128
    )
    sB = alloc.allocate_tensor(
        FP8_DTYPE, cute.make_layout((N_TILE, KT)), 128
    )
    m_base = nby * M_TILE
    n_base = nbx * N_TILE
    acc, row_g, col_g = gemm_stage_and_compute(
        a, w, sA, sB, tid, m_base, n_base, MLP_SIZE
    )
    for i in cutlass.range_constexpr(TM):
        m = m_base + row_g * TM + i
        for j in cutlass.range_constexpr(TN):
            n = n_base + col_g * TN + j
            out[m, n] = (
                residual[m, n].to(cutlass.Float32)
                + acc[i][j] * cutlass.Float32(S_PROJ)
                + bias[n].to(cutlass.Float32)
            )


@cute.jit
def gpt2_transformer_block(
    hidden: cute.Tensor,
    ln1_weight: cute.Tensor,
    ln1_bias: cute.Tensor,
    qkv_weight: cute.Tensor,
    qkv_bias: cute.Tensor,
    out_weight: cute.Tensor,
    out_bias: cute.Tensor,
    ln2_weight: cute.Tensor,
    ln2_bias: cute.Tensor,
    fc_weight: cute.Tensor,
    fc_bias: cute.Tensor,
    proj_weight: cute.Tensor,
    proj_bias: cute.Tensor,
    norm1_workspace: cute.Tensor,
    qkv_workspace: cute.Tensor,
    score_workspace: cute.Tensor,
    probability_workspace: cute.Tensor,
    context_workspace: cute.Tensor,
    residual_workspace: cute.Tensor,
    norm2_workspace: cute.Tensor,
    mlp_workspace: cute.Tensor,
    output: cute.Tensor,
):
    n_m_tiles = TOKENS // M_TILE
    ln1_kernel(hidden, ln1_weight, ln1_bias, norm1_workspace).launch(
        grid=(TOKENS, 1, 1), block=(THREADS, 1, 1)
    )
    qkv_gemm_kernel(norm1_workspace, qkv_weight, qkv_bias, qkv_workspace).launch(
        grid=(QKV_SIZE // N_TILE, n_m_tiles, 1), block=(THREADS, 1, 1)
    )
    attention_kernel(qkv_workspace, context_workspace).launch(
        grid=(NUM_HEADS, TOKENS, 1), block=(THREADS, 1, 1)
    )
    out_gemm_kernel(
        context_workspace, out_weight, out_bias, hidden, residual_workspace
    ).launch(grid=(HIDDEN_SIZE // N_TILE, n_m_tiles, 1), block=(THREADS, 1, 1))
    ln2_kernel(residual_workspace, ln2_weight, ln2_bias, norm2_workspace).launch(
        grid=(TOKENS, 1, 1), block=(THREADS, 1, 1)
    )
    fc_gelu_kernel(norm2_workspace, fc_weight, fc_bias, mlp_workspace).launch(
        grid=(MLP_SIZE // N_TILE, n_m_tiles, 1), block=(THREADS, 1, 1)
    )
    proj_gemm_kernel(
        mlp_workspace, proj_weight, proj_bias, residual_workspace, output
    ).launch(grid=(HIDDEN_SIZE // N_TILE, n_m_tiles, 1), block=(THREADS, 1, 1))


class ModelNew:
    forward = staticmethod(gpt2_transformer_block)
