import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
MLP_SIZE = 3072
QKV_SIZE = 3 * HIDDEN_SIZE
THREADS = 128
EPSILON = 1.0e-5
UNROLL = 8

# Output columns computed per thread in each dense projection kernel.
QKV_CPT = 2
OUT_CPT = 2
FC_CPT = 4
PROJ_CPT = 3

# Shared-memory weight staging: one K-chunk per tile, padded row stride to
# reduce shared-memory bank conflicts on the compute reads.
KCHUNK = 64
STRIDE = KCHUNK + 1

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = MLP_SIZE ** -0.5 / 448.0
FP8_DTYPE = cutlass.Float8E4M3FN

SQRT_HEAD_DIM = 8.0  # sqrt(64)
NEG_INF = -1.0e30
GELU_CONST = 0.7978845608028654  # sqrt(2/pi)


@cute.jit
def gelu_new(x):
    # GPT-2 "new" GELU (tanh approximation).
    return 0.5 * x * (1.0 + cute.tanh(GELU_CONST * (x + 0.044715 * x * x * x)))


# ---------------------------------------------------------------------------
# Layer norm over an FP8 input dequantized by INPUT_SCALE. Output requantized
# to FP8 with NORM_SCALE.
# ---------------------------------------------------------------------------
@cute.kernel
def layer_norm_fp8_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem_sum = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_sumsq = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_mean = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )
    smem_rstd = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )

    local_sum = cutlass.Float32(0.0)
    local_sumsq = cutlass.Float32(0.0)
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        col = i * THREADS + tid
        v = hidden[row, col].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
        local_sum = local_sum + v
        local_sumsq = local_sumsq + v * v
    smem_sum[tid] = local_sum
    smem_sumsq[tid] = local_sumsq
    cute.arch.sync_threads()

    if tid == 0:
        total_sum = cutlass.Float32(0.0)
        total_sumsq = cutlass.Float32(0.0)
        for i in cutlass.range(THREADS):
            total_sum = total_sum + smem_sum[i]
            total_sumsq = total_sumsq + smem_sumsq[i]
        mean = total_sum / cutlass.Float32(HIDDEN_SIZE)
        var = total_sumsq / cutlass.Float32(HIDDEN_SIZE) - mean * mean
        smem_mean[0] = mean
        smem_rstd[0] = cute.rsqrt(var + cutlass.Float32(EPSILON))
    cute.arch.sync_threads()

    mean = smem_mean[0]
    rstd = smem_rstd[0]
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        col = i * THREADS + tid
        v = hidden[row, col].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
        norm = (v - mean) * rstd
        out = norm * weight[col].to(cutlass.Float32) + bias[col].to(cutlass.Float32)
        output[row, col] = cutlass.Float8E4M3FN(out / cutlass.Float32(NORM_SCALE))


# ---------------------------------------------------------------------------
# Layer norm over an already-real FP32 input (the residual). Output requantized
# to FP8 with NORM_SCALE.
# ---------------------------------------------------------------------------
@cute.kernel
def layer_norm_fp32_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()

    smem_sum = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_sumsq = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_mean = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )
    smem_rstd = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )

    local_sum = cutlass.Float32(0.0)
    local_sumsq = cutlass.Float32(0.0)
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        col = i * THREADS + tid
        v = hidden[row, col].to(cutlass.Float32)
        local_sum = local_sum + v
        local_sumsq = local_sumsq + v * v
    smem_sum[tid] = local_sum
    smem_sumsq[tid] = local_sumsq
    cute.arch.sync_threads()

    if tid == 0:
        total_sum = cutlass.Float32(0.0)
        total_sumsq = cutlass.Float32(0.0)
        for i in cutlass.range(THREADS):
            total_sum = total_sum + smem_sum[i]
            total_sumsq = total_sumsq + smem_sumsq[i]
        mean = total_sum / cutlass.Float32(HIDDEN_SIZE)
        var = total_sumsq / cutlass.Float32(HIDDEN_SIZE) - mean * mean
        smem_mean[0] = mean
        smem_rstd[0] = cute.rsqrt(var + cutlass.Float32(EPSILON))
    cute.arch.sync_threads()

    mean = smem_mean[0]
    rstd = smem_rstd[0]
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        col = i * THREADS + tid
        v = hidden[row, col].to(cutlass.Float32)
        norm = (v - mean) * rstd
        out = norm * weight[col].to(cutlass.Float32) + bias[col].to(cutlass.Float32)
        output[row, col] = cutlass.Float8E4M3FN(out / cutlass.Float32(NORM_SCALE))


# ---------------------------------------------------------------------------
# Dense projections. One CTA per (output column tile, token). The weight tile is
# staged into shared memory in coalesced K-chunks so that the (previously badly
# scattered) weight reads become unit-stride global loads. The A-operand row is
# staged once per CTA. Accumulation stays a single sequential pass over K, which
# reproduces the reference accumulation order exactly.
# ---------------------------------------------------------------------------
@cute.kernel
def qkv_projection_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    n_tile, row, _ = cute.arch.block_idx()
    NB = THREADS * QKV_CPT
    n0 = n_tile * NB
    NUM_KCHUNKS = HIDDEN_SIZE // KCHUNK

    smem_h = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, HIDDEN_SIZE, 4),
        cute.make_layout(HIDDEN_SIZE),
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, NB * STRIDE, 4),
        cute.make_layout(NB * STRIDE),
    )
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        smem_h[i * THREADS + tid] = hidden[row, i * THREADS + tid]
    cute.arch.sync_threads()

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    for kc in cutlass.range(NUM_KCHUNKS):
        for i in cutlass.range(NB * KCHUNK // THREADS):
            idx = i * THREADS + tid
            nn = idx // KCHUNK
            kk = idx % KCHUNK
            smem_w[nn * STRIDE + kk] = weight[n0 + nn, kc * KCHUNK + kk]
        cute.arch.sync_threads()
        for k in cutlass.range(KCHUNK):
            av = smem_h[kc * KCHUNK + k].to(cutlass.Float32)
            acc0 = acc0 + av * smem_w[(tid * QKV_CPT + 0) * STRIDE + k].to(cutlass.Float32)
            acc1 = acc1 + av * smem_w[(tid * QKV_CPT + 1) * STRIDE + k].to(cutlass.Float32)
        cute.arch.sync_threads()

    real0 = acc0 * cutlass.Float32(NORM_SCALE * WEIGHT_H_SCALE) + bias[n0 + tid * QKV_CPT + 0].to(cutlass.Float32)
    real1 = acc1 * cutlass.Float32(NORM_SCALE * WEIGHT_H_SCALE) + bias[n0 + tid * QKV_CPT + 1].to(cutlass.Float32)
    output[row, n0 + tid * QKV_CPT + 0] = cutlass.Float8E4M3FN(real0 / cutlass.Float32(QKV_SCALE))
    output[row, n0 + tid * QKV_CPT + 1] = cutlass.Float8E4M3FN(real1 / cutlass.Float32(QKV_SCALE))


@cute.kernel
def attention_fused_kernel(qkv: cute.Tensor, context: cute.Tensor):
    # One CTA per (query token, head). Thread t owns key index t for that head.
    # Causal scores, per-head softmax and the 64-dim context output are computed
    # in shared memory. THREADS == TOKENS.
    tid, _, _ = cute.arch.thread_idx()
    row, head, _ = cute.arch.block_idx()

    smem_score = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_prob = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_max = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_sum = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, THREADS, 4),
        cute.make_layout(THREADS),
    )
    smem_mx = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )
    smem_total = cute.make_tensor(
        cute.arch.alloc_smem(cutlass.Float32, 1, 4),
        cute.make_layout(1),
    )

    # Stage 1: score[tid] = q[row,head,:] . k[tid,head,:] / sqrt(64), causal masked.
    hbase = head * HEAD_DIM
    q0 = cutlass.Float32(0.0)
    q1 = cutlass.Float32(0.0)
    q2 = cutlass.Float32(0.0)
    q3 = cutlass.Float32(0.0)
    q4 = cutlass.Float32(0.0)
    q5 = cutlass.Float32(0.0)
    q6 = cutlass.Float32(0.0)
    q7 = cutlass.Float32(0.0)
    for b in cutlass.range(HEAD_DIM // UNROLL):
        base = b * UNROLL
        q0 = q0 + (qkv[row, hbase + base + 0].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 0].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q1 = q1 + (qkv[row, hbase + base + 1].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 1].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q2 = q2 + (qkv[row, hbase + base + 2].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 2].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q3 = q3 + (qkv[row, hbase + base + 3].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 3].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q4 = q4 + (qkv[row, hbase + base + 4].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 4].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q5 = q5 + (qkv[row, hbase + base + 5].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 5].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q6 = q6 + (qkv[row, hbase + base + 6].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 6].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        q7 = q7 + (qkv[row, hbase + base + 7].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE)) * (qkv[tid, HIDDEN_SIZE + hbase + base + 7].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
    acc = ((q0 + q1) + (q2 + q3)) + ((q4 + q5) + (q6 + q7))
    s = acc / cutlass.Float32(SQRT_HEAD_DIM)
    if tid > row:
        s = cutlass.Float32(NEG_INF)
    smem_score[tid] = s
    cute.arch.sync_threads()

    # Stage 2: softmax over the key axis.
    s = smem_score[tid]
    smem_max[tid] = s
    cute.arch.sync_threads()
    if tid == 0:
        mx = smem_max[0]
        for i in cutlass.range(1, THREADS):
            if smem_max[i] > mx:
                mx = smem_max[i]
        smem_mx[0] = mx
    cute.arch.sync_threads()
    mx = smem_mx[0]
    e = cute.exp(s - mx)
    smem_sum[tid] = e
    cute.arch.sync_threads()
    if tid == 0:
        tot = smem_sum[0]
        for i in cutlass.range(1, THREADS):
            tot = tot + smem_sum[i]
        smem_total[0] = tot
    cute.arch.sync_threads()
    tot = smem_total[0]
    smem_prob[tid] = e / tot
    cute.arch.sync_threads()

    # Stage 3: context[row, head*64+d] = sum_n prob[n] * v[n,head,d], requantized.
    if tid < HEAD_DIM:
        d = tid
        c0 = cutlass.Float32(0.0)
        c1 = cutlass.Float32(0.0)
        c2 = cutlass.Float32(0.0)
        c3 = cutlass.Float32(0.0)
        c4 = cutlass.Float32(0.0)
        c5 = cutlass.Float32(0.0)
        c6 = cutlass.Float32(0.0)
        c7 = cutlass.Float32(0.0)
        for b in cutlass.range(TOKENS // UNROLL):
            base = b * UNROLL
            c0 = c0 + smem_prob[base + 0] * (qkv[base + 0, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c1 = c1 + smem_prob[base + 1] * (qkv[base + 1, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c2 = c2 + smem_prob[base + 2] * (qkv[base + 2, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c3 = c3 + smem_prob[base + 3] * (qkv[base + 3, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c4 = c4 + smem_prob[base + 4] * (qkv[base + 4, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c5 = c5 + smem_prob[base + 5] * (qkv[base + 5, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c6 = c6 + smem_prob[base + 6] * (qkv[base + 6, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
            c7 = c7 + smem_prob[base + 7] * (qkv[base + 7, 2 * HIDDEN_SIZE + hbase + d].to(cutlass.Float32) * cutlass.Float32(QKV_SCALE))
        cacc = ((c0 + c1) + (c2 + c3)) + ((c4 + c5) + (c6 + c7))
        context[row, hbase + d] = cutlass.Float8E4M3FN(
            cacc / cutlass.Float32(CONTEXT_SCALE)
        )


@cute.kernel
def attention_projection_kernel(
    hidden: cute.Tensor,
    context: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    n_tile, row, _ = cute.arch.block_idx()
    NB = THREADS * OUT_CPT
    n0 = n_tile * NB
    NUM_KCHUNKS = HIDDEN_SIZE // KCHUNK

    smem_h = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, HIDDEN_SIZE, 4),
        cute.make_layout(HIDDEN_SIZE),
    )
    smem_c = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, HIDDEN_SIZE, 4),
        cute.make_layout(HIDDEN_SIZE),
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, NB * STRIDE, 4),
        cute.make_layout(NB * STRIDE),
    )
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        smem_h[i * THREADS + tid] = hidden[row, i * THREADS + tid]
        smem_c[i * THREADS + tid] = context[row, i * THREADS + tid]
    cute.arch.sync_threads()

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    for kc in cutlass.range(NUM_KCHUNKS):
        for i in cutlass.range(NB * KCHUNK // THREADS):
            idx = i * THREADS + tid
            nn = idx // KCHUNK
            kk = idx % KCHUNK
            smem_w[nn * STRIDE + kk] = weight[n0 + nn, kc * KCHUNK + kk]
        cute.arch.sync_threads()
        for k in cutlass.range(KCHUNK):
            av = smem_c[kc * KCHUNK + k].to(cutlass.Float32)
            acc0 = acc0 + av * smem_w[(tid * OUT_CPT + 0) * STRIDE + k].to(cutlass.Float32)
            acc1 = acc1 + av * smem_w[(tid * OUT_CPT + 1) * STRIDE + k].to(cutlass.Float32)
        cute.arch.sync_threads()

    real0 = acc0 * cutlass.Float32(CONTEXT_SCALE * WEIGHT_H_SCALE) + bias[n0 + tid * OUT_CPT + 0].to(cutlass.Float32)
    real1 = acc1 * cutlass.Float32(CONTEXT_SCALE * WEIGHT_H_SCALE) + bias[n0 + tid * OUT_CPT + 1].to(cutlass.Float32)
    residual[row, n0 + tid * OUT_CPT + 0] = smem_h[n0 + tid * OUT_CPT + 0].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE) + real0
    residual[row, n0 + tid * OUT_CPT + 1] = smem_h[n0 + tid * OUT_CPT + 1].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE) + real1


@cute.kernel
def mlp_fc_gelu_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    n_tile, row, _ = cute.arch.block_idx()
    NB = THREADS * FC_CPT
    n0 = n_tile * NB
    NUM_KCHUNKS = HIDDEN_SIZE // KCHUNK

    smem_h = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, HIDDEN_SIZE, 4),
        cute.make_layout(HIDDEN_SIZE),
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, NB * STRIDE, 4),
        cute.make_layout(NB * STRIDE),
    )
    for i in cutlass.range(HIDDEN_SIZE // THREADS):
        smem_h[i * THREADS + tid] = hidden[row, i * THREADS + tid]
    cute.arch.sync_threads()

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    acc3 = cutlass.Float32(0.0)
    for kc in cutlass.range(NUM_KCHUNKS):
        for i in cutlass.range(NB * KCHUNK // THREADS):
            idx = i * THREADS + tid
            nn = idx // KCHUNK
            kk = idx % KCHUNK
            smem_w[nn * STRIDE + kk] = weight[n0 + nn, kc * KCHUNK + kk]
        cute.arch.sync_threads()
        for k in cutlass.range(KCHUNK):
            av = smem_h[kc * KCHUNK + k].to(cutlass.Float32)
            acc0 = acc0 + av * smem_w[(tid * FC_CPT + 0) * STRIDE + k].to(cutlass.Float32)
            acc1 = acc1 + av * smem_w[(tid * FC_CPT + 1) * STRIDE + k].to(cutlass.Float32)
            acc2 = acc2 + av * smem_w[(tid * FC_CPT + 2) * STRIDE + k].to(cutlass.Float32)
            acc3 = acc3 + av * smem_w[(tid * FC_CPT + 3) * STRIDE + k].to(cutlass.Float32)
        cute.arch.sync_threads()

    for j in cutlass.range_constexpr(FC_CPT):
        fc_real = [acc0, acc1, acc2, acc3][j] * cutlass.Float32(NORM_SCALE * WEIGHT_H_SCALE) + bias[n0 + tid * FC_CPT + j].to(cutlass.Float32)
        g = gelu_new(fc_real)
        output[row, n0 + tid * FC_CPT + j] = cutlass.Float8E4M3FN(g / cutlass.Float32(MLP_SCALE))


@cute.kernel
def mlp_projection_kernel(
    residual: cute.Tensor,
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    tid, _, _ = cute.arch.thread_idx()
    n_tile, row, _ = cute.arch.block_idx()
    NB = THREADS * PROJ_CPT
    n0 = n_tile * NB
    NUM_KCHUNKS = MLP_SIZE // KCHUNK

    smem_h = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, MLP_SIZE, 4),
        cute.make_layout(MLP_SIZE),
    )
    smem_w = cute.make_tensor(
        cute.arch.alloc_smem(FP8_DTYPE, NB * STRIDE, 4),
        cute.make_layout(NB * STRIDE),
    )
    for i in cutlass.range(MLP_SIZE // THREADS):
        smem_h[i * THREADS + tid] = hidden[row, i * THREADS + tid]
    cute.arch.sync_threads()

    acc0 = cutlass.Float32(0.0)
    acc1 = cutlass.Float32(0.0)
    acc2 = cutlass.Float32(0.0)
    for kc in cutlass.range(NUM_KCHUNKS):
        for i in cutlass.range(NB * KCHUNK // THREADS):
            idx = i * THREADS + tid
            nn = idx // KCHUNK
            kk = idx % KCHUNK
            smem_w[nn * STRIDE + kk] = weight[n0 + nn, kc * KCHUNK + kk]
        cute.arch.sync_threads()
        for k in cutlass.range(KCHUNK):
            av = smem_h[kc * KCHUNK + k].to(cutlass.Float32)
            acc0 = acc0 + av * smem_w[(tid * PROJ_CPT + 0) * STRIDE + k].to(cutlass.Float32)
            acc1 = acc1 + av * smem_w[(tid * PROJ_CPT + 1) * STRIDE + k].to(cutlass.Float32)
            acc2 = acc2 + av * smem_w[(tid * PROJ_CPT + 2) * STRIDE + k].to(cutlass.Float32)
        cute.arch.sync_threads()

    real0 = acc0 * cutlass.Float32(MLP_SCALE * WEIGHT_MLP_SCALE) + bias[n0 + tid * PROJ_CPT + 0].to(cutlass.Float32)
    real1 = acc1 * cutlass.Float32(MLP_SCALE * WEIGHT_MLP_SCALE) + bias[n0 + tid * PROJ_CPT + 1].to(cutlass.Float32)
    real2 = acc2 * cutlass.Float32(MLP_SCALE * WEIGHT_MLP_SCALE) + bias[n0 + tid * PROJ_CPT + 2].to(cutlass.Float32)
    output[row, n0 + tid * PROJ_CPT + 0] = residual[row, n0 + tid * PROJ_CPT + 0].to(cutlass.Float32) + real0
    output[row, n0 + tid * PROJ_CPT + 1] = residual[row, n0 + tid * PROJ_CPT + 1].to(cutlass.Float32) + real1
    output[row, n0 + tid * PROJ_CPT + 2] = residual[row, n0 + tid * PROJ_CPT + 2].to(cutlass.Float32) + real2


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
    grid = (TOKENS, 1, 1)
    block = (THREADS, 1, 1)

    layer_norm_fp8_kernel(hidden, ln1_weight, ln1_bias, norm1_workspace).launch(grid=grid, block=block)
    qkv_projection_kernel(norm1_workspace, qkv_weight, qkv_bias, qkv_workspace).launch(
        grid=(QKV_SIZE // (THREADS * QKV_CPT), TOKENS, 1), block=block
    )
    attention_fused_kernel(qkv_workspace, context_workspace).launch(
        grid=(TOKENS, NUM_HEADS, 1), block=block
    )
    attention_projection_kernel(hidden, context_workspace, out_weight, out_bias, residual_workspace).launch(
        grid=(HIDDEN_SIZE // (THREADS * OUT_CPT), TOKENS, 1), block=block
    )
    layer_norm_fp32_kernel(residual_workspace, ln2_weight, ln2_bias, norm2_workspace).launch(grid=grid, block=block)
    mlp_fc_gelu_kernel(norm2_workspace, fc_weight, fc_bias, mlp_workspace).launch(
        grid=(MLP_SIZE // (THREADS * FC_CPT), TOKENS, 1), block=block
    )
    mlp_projection_kernel(residual_workspace, mlp_workspace, proj_weight, proj_bias, output).launch(
        grid=(HIDDEN_SIZE // (THREADS * PROJ_CPT), TOKENS, 1), block=block
    )


class ModelNew:
    forward = staticmethod(gpt2_transformer_block)
