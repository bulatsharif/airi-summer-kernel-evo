"""GPT-2 Small transformer block in CuTe DSL (FP8).

Fixed GPT-2 Small shape: tokens=128, hidden=768, heads=12, head_dim=64,
mlp=3072, qkv=2304.

Precision contract: hidden state and weights are physically FP8 E4M3FN.
LayerNorm output, packed QKV, attention context and GELU output are requantized
to FP8 at their declared boundaries; dense accumulation, LayerNorm statistics,
attention scores/softmax, biases, residuals and the final output are FP32.
"""

import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 768
NUM_HEADS = 12
HEAD_DIM = 64
MLP_SIZE = 3072
QKV_SIZE = 3 * HIDDEN_SIZE
EPSILON = 1.0e-5

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = MLP_SIZE ** -0.5 / 448.0
FP8_DTYPE = cutlass.Float8E4M3FN

NEG_INF = -1.0e30
SQRT_2_OVER_PI = 0.7978845608028654
GELU_C = 0.044715

WARP = 32
# Elements each lane accumulates serially in the reduction kernels.
K_STEPS_H = HIDDEN_SIZE // WARP      # 24
K_STEPS_M = MLP_SIZE // WARP         # 96
HD_STEPS = HEAD_DIM // WARP          # 2

# Launch geometry: one warp per output row / per (head, token).
QKV_WARPS = TOKENS * QKV_SIZE        # 294912 warps -> 294912/32 lanes
H_WARPS = TOKENS * HIDDEN_SIZE       # 98304
M_WARPS = TOKENS * MLP_SIZE          # 393216
ATTN_WARPS = NUM_HEADS * TOKENS      # 1536
BLOCK = 256


@cute.jit
def warp_reduce_sum(v):
    v = v + cute.arch.shuffle_sync_bfly(v, 16)
    v = v + cute.arch.shuffle_sync_bfly(v, 8)
    v = v + cute.arch.shuffle_sync_bfly(v, 4)
    v = v + cute.arch.shuffle_sync_bfly(v, 2)
    v = v + cute.arch.shuffle_sync_bfly(v, 1)
    return v


@cute.jit
def gelu_new(x):
    x3 = x * x * x
    inner = cutlass.Float32(SQRT_2_OVER_PI) * (x + cutlass.Float32(GELU_C) * x3)
    t = cute.tanh(inner)
    return cutlass.Float32(0.5) * x * (cutlass.Float32(1.0) + t)


@cute.kernel
def layer_norm_fp8_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    token = tid // WARP
    lane = tid % WARP

    mean = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        mean = mean + hidden[token, h].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
    mean = warp_reduce_sum(mean) / cutlass.Float32(HIDDEN_SIZE)

    var = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        d = hidden[token, h].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE) - mean
        var = var + d * d
    var = warp_reduce_sum(var) / cutlass.Float32(HIDDEN_SIZE)
    rstd = cute.rsqrt(var + cutlass.Float32(EPSILON))

    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        val = (hidden[token, h].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE) - mean) * rstd
        n1 = val * weight[h].to(cutlass.Float32) + bias[h].to(cutlass.Float32)
        output[token, h] = FP8_DTYPE(n1 * cutlass.Float32(1.0 / NORM_SCALE))


@cute.kernel
def qkv_projection_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    warp = tid // WARP
    lane = tid % WARP
    token = warp // QKV_SIZE
    out_idx = warp % QKV_SIZE

    acc = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        k = lane + i * WARP
        acc = acc + hidden[token, k].to(cutlass.Float32) * weight[out_idx, k].to(cutlass.Float32)
    acc = warp_reduce_sum(acc)

    if lane == 0:
        real = acc * cutlass.Float32(NORM_SCALE * WEIGHT_H_SCALE) + bias[out_idx].to(cutlass.Float32)
        output[token, out_idx] = FP8_DTYPE(real * cutlass.Float32(1.0 / QKV_SCALE))


@cute.kernel
def attention_kernel(qkv: cute.Tensor, context: cute.Tensor):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    warp = tid // WARP
    lane = tid % WARP

    if warp < ATTN_WARPS:
        head = warp // TOKENS
        t1 = warp % TOKENS
        q_base = head * HEAD_DIM
        k_base = HIDDEN_SIZE + head * HEAD_DIM
        v_base = 2 * HIDDEN_SIZE + head * HEAD_DIM
        hbase = head * HEAD_DIM
        qq = cutlass.Float32(QKV_SCALE * QKV_SCALE)
        inv_sqrt = cutlass.Float32(1.0 / 8.0)

        acc = cute.make_rmem_tensor(cute.make_layout(HD_STEPS), cutlass.Float32)
        for j in cutlass.range_constexpr(HD_STEPS):
            acc[j] = cutlass.Float32(0.0)

        m = cutlass.Float32(NEG_INF)
        l = cutlass.Float32(0.0)
        for t2 in cutlass.range(TOKENS):
            s = cutlass.Float32(0.0)
            for j in cutlass.range(HD_STEPS):
                d = lane + j * WARP
                s = s + qkv[t1, q_base + d].to(cutlass.Float32) * qkv[t2, k_base + d].to(cutlass.Float32)
            s = warp_reduce_sum(s) * qq * inv_sqrt
            s = s if (t2 <= t1) else cutlass.Float32(NEG_INF)
            m_new = s if (s > m) else m
            corr = cute.exp(m - m_new)
            p = cute.exp(s - m_new)
            l = l * corr + p
            for j in cutlass.range_constexpr(HD_STEPS):
                d = lane + j * WARP
                acc[j] = acc[j] * corr + p * qkv[t2, v_base + d].to(cutlass.Float32)
            m = m_new

        inv = cutlass.Float32(1.0) / l
        for j in cutlass.range_constexpr(HD_STEPS):
            d = lane + j * WARP
            ctx = acc[j] * inv * cutlass.Float32(QKV_SCALE / CONTEXT_SCALE)
            context[t1, hbase + d] = FP8_DTYPE(ctx)


@cute.kernel
def attention_projection_kernel(
    hidden: cute.Tensor,
    context: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    warp = tid // WARP
    lane = tid % WARP
    token = warp // HIDDEN_SIZE
    out_idx = warp % HIDDEN_SIZE

    acc = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        k = lane + i * WARP
        acc = acc + context[token, k].to(cutlass.Float32) * weight[out_idx, k].to(cutlass.Float32)
    acc = warp_reduce_sum(acc)

    if lane == 0:
        linear = acc * cutlass.Float32(CONTEXT_SCALE * WEIGHT_H_SCALE) + bias[out_idx].to(cutlass.Float32)
        hidden_real = hidden[token, out_idx].to(cutlass.Float32) * cutlass.Float32(INPUT_SCALE)
        residual[token, out_idx] = hidden_real + linear


@cute.kernel
def layer_norm_fp32_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    token = tid // WARP
    lane = tid % WARP

    mean = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        mean = mean + hidden[token, h].to(cutlass.Float32)
    mean = warp_reduce_sum(mean) / cutlass.Float32(HIDDEN_SIZE)

    var = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        d = hidden[token, h].to(cutlass.Float32) - mean
        var = var + d * d
    var = warp_reduce_sum(var) / cutlass.Float32(HIDDEN_SIZE)
    rstd = cute.rsqrt(var + cutlass.Float32(EPSILON))

    for i in cutlass.range(K_STEPS_H):
        h = lane + i * WARP
        val = (hidden[token, h].to(cutlass.Float32) - mean) * rstd
        n2 = val * weight[h].to(cutlass.Float32) + bias[h].to(cutlass.Float32)
        output[token, h] = FP8_DTYPE(n2 * cutlass.Float32(1.0 / NORM_SCALE))


@cute.kernel
def mlp_fc_gelu_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    warp = tid // WARP
    lane = tid % WARP
    token = warp // MLP_SIZE
    out_idx = warp % MLP_SIZE

    acc = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_H):
        k = lane + i * WARP
        acc = acc + hidden[token, k].to(cutlass.Float32) * weight[out_idx, k].to(cutlass.Float32)
    acc = warp_reduce_sum(acc)

    if lane == 0:
        fc = acc * cutlass.Float32(NORM_SCALE * WEIGHT_H_SCALE) + bias[out_idx].to(cutlass.Float32)
        g = gelu_new(fc)
        output[token, out_idx] = FP8_DTYPE(g * cutlass.Float32(1.0 / MLP_SCALE))


@cute.kernel
def mlp_projection_kernel(
    residual: cute.Tensor,
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    flat, _, _ = cute.arch.thread_idx()
    blk, _, _ = cute.arch.block_idx()
    bdim, _, _ = cute.arch.block_dim()
    tid = blk * bdim + flat
    warp = tid // WARP
    lane = tid % WARP
    token = warp // HIDDEN_SIZE
    out_idx = warp % HIDDEN_SIZE

    acc = cutlass.Float32(0.0)
    for i in cutlass.range(K_STEPS_M):
        k = lane + i * WARP
        acc = acc + hidden[token, k].to(cutlass.Float32) * weight[out_idx, k].to(cutlass.Float32)
    acc = warp_reduce_sum(acc)

    if lane == 0:
        mlp = acc * cutlass.Float32(MLP_SCALE * WEIGHT_MLP_SCALE) + bias[out_idx].to(cutlass.Float32)
        output[token, out_idx] = residual[token, out_idx].to(cutlass.Float32) + mlp


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
    # 1. n1 = layer_norm(hidden, ln1_weight, ln1_bias) -> FP8 norm1_workspace
    layer_norm_fp8_kernel(hidden, ln1_weight, ln1_bias, norm1_workspace).launch(
        grid=(TOKENS * WARP // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
    )
    # 2. qkv = linear_fp8(n1, qkv_weight, qkv_bias) -> FP8 qkv_workspace
    qkv_projection_kernel(norm1_workspace, qkv_weight, qkv_bias, qkv_workspace).launch(
        grid=(QKV_WARPS // (BLOCK // WARP), 1, 1),
        block=(BLOCK, 1, 1),
    )
    # 3. context = causal_softmax(q @ k.T / sqrt(64)) @ v -> FP8 context_workspace
    attention_kernel(qkv_workspace, context_workspace).launch(
        grid=(ATTN_WARPS // (BLOCK // WARP), 1, 1),
        block=(BLOCK, 1, 1),
    )
    # 4. residual = hidden + linear_fp8(context, out_weight, out_bias) -> FP32
    attention_projection_kernel(hidden, context_workspace, out_weight, out_bias, residual_workspace).launch(
        grid=(H_WARPS // (BLOCK // WARP), 1, 1),
        block=(BLOCK, 1, 1),
    )
    # 5. n2 = layer_norm(residual, ln2_weight, ln2_bias) -> FP8 norm2_workspace
    layer_norm_fp32_kernel(residual_workspace, ln2_weight, ln2_bias, norm2_workspace).launch(
        grid=(TOKENS * WARP // BLOCK, 1, 1),
        block=(BLOCK, 1, 1),
    )
    # 6. mlp = linear_fp8(gelu_new(linear_fp8(n2, fc_weight, fc_bias)), proj_weight, proj_bias)
    mlp_fc_gelu_kernel(norm2_workspace, fc_weight, fc_bias, mlp_workspace).launch(
        grid=(M_WARPS // (BLOCK // WARP), 1, 1),
        block=(BLOCK, 1, 1),
    )
    mlp_projection_kernel(residual_workspace, mlp_workspace, proj_weight, proj_bias, output).launch(
        grid=(H_WARPS // (BLOCK // WARP), 1, 1),
        block=(BLOCK, 1, 1),
    )


class ModelNew:
    forward = staticmethod(gpt2_transformer_block)
