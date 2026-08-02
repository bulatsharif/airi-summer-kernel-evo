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

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = MLP_SIZE ** -0.5 / 448.0


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def layer_norm_fp8_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    partial_sum = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        partial_sum += hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
    mean = warp_sum(partial_sum) / HIDDEN_SIZE

    partial_variance = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        centered = (
            hidden[row, column].to(cutlass.Float32) * INPUT_SCALE - mean
        )
        partial_variance += centered * centered
    inverse_std = cute.rsqrt(
        warp_sum(partial_variance) / HIDDEN_SIZE + EPSILON
    )

    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
        normalized = (value - mean) * inverse_std
        output[row, column] = (
            (
                normalized * weight[column].to(cutlass.Float32)
                + bias[column].to(cutlass.Float32)
            )
            / NORM_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def qkv_projection_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * QKV_SIZE:
        row = linear // QKV_SIZE
        column = linear % QKV_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(HIDDEN_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        output[row, column] = (
            (
                accumulator * (NORM_SCALE * WEIGHT_H_SCALE)
                + bias[column].to(cutlass.Float32)
            )
            / QKV_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def attention_score_kernel(qkv: cute.Tensor, scores: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < NUM_HEADS * TOKENS * TOKENS:
        key = linear % TOKENS
        row = linear // TOKENS
        query = row % TOKENS
        head = row // TOKENS
        if key <= query:
            accumulator = cutlass.Float32(0.0)
            head_start = head * HEAD_DIM
            for column in cutlass.range(HEAD_DIM):
                accumulator += (
                    qkv[query, head_start + column].to(cutlass.Float32)
                    * qkv[
                        key, HIDDEN_SIZE + head_start + column
                    ].to(cutlass.Float32)
                )
            scores[row, key] = (
                accumulator * QKV_SCALE * QKV_SCALE / 8.0
            )
        else:
            scores[row, key] = -cutlass.Float32(1.0e30)


@cute.kernel
def causal_softmax_kernel(scores: cute.Tensor, probabilities: cute.Tensor):
    row, _, _ = cute.arch.block_idx()
    maximum = scores[row, 0].to(cutlass.Float32)
    for key in cutlass.range(TOKENS):
        value = scores[row, key].to(cutlass.Float32)
        if value > maximum:
            maximum = value

    denominator = cutlass.Float32(0.0)
    for key in cutlass.range(TOKENS):
        denominator += cute.exp(scores[row, key].to(cutlass.Float32) - maximum)
    for key in cutlass.range(TOKENS):
        probabilities[row, key] = cute.exp(
            scores[row, key].to(cutlass.Float32) - maximum
        ) / denominator


@cute.kernel
def attention_context_kernel(
    qkv: cute.Tensor,
    probabilities: cute.Tensor,
    context: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * HIDDEN_SIZE:
        query = linear // HIDDEN_SIZE
        column = linear % HIDDEN_SIZE
        head = column // HEAD_DIM
        probability_row = head * TOKENS + query
        accumulator = cutlass.Float32(0.0)
        for key in cutlass.range(TOKENS):
            accumulator += (
                probabilities[probability_row, key].to(cutlass.Float32)
                * qkv[key, 2 * HIDDEN_SIZE + column].to(cutlass.Float32)
            )
        context[query, column] = (
            accumulator * QKV_SCALE / CONTEXT_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def attention_projection_kernel(
    hidden: cute.Tensor,
    context: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * HIDDEN_SIZE:
        row = linear // HIDDEN_SIZE
        column = linear % HIDDEN_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(HIDDEN_SIZE):
            accumulator += (
                context[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        residual[row, column] = (
            hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
            + accumulator * (CONTEXT_SCALE * WEIGHT_H_SCALE)
            + bias[column].to(cutlass.Float32)
        )


@cute.kernel
def layer_norm_fp32_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    partial_sum = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        partial_sum += hidden[row, column].to(cutlass.Float32)
    mean = warp_sum(partial_sum) / HIDDEN_SIZE

    partial_variance = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        centered = hidden[row, column].to(cutlass.Float32) - mean
        partial_variance += centered * centered
    inverse_std = cute.rsqrt(
        warp_sum(partial_variance) / HIDDEN_SIZE + EPSILON
    )

    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        normalized = (
            hidden[row, column].to(cutlass.Float32) - mean
        ) * inverse_std
        output[row, column] = (
            (
                normalized * weight[column].to(cutlass.Float32)
                + bias[column].to(cutlass.Float32)
            )
            / NORM_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def mlp_fc_gelu_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * MLP_SIZE:
        row = linear // MLP_SIZE
        column = linear % MLP_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(HIDDEN_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        value = (
            accumulator * (NORM_SCALE * WEIGHT_H_SCALE)
            + bias[column].to(cutlass.Float32)
        )
        cubic = value * value * value
        gelu = 0.5 * value * (
            1.0 + cute.tanh(0.7978845608028654 * (value + 0.044715 * cubic))
        )
        output[row, column] = (gelu / MLP_SCALE).to(cutlass.Float8E4M3FN)


@cute.kernel
def mlp_projection_kernel(
    residual: cute.Tensor,
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * HIDDEN_SIZE:
        row = linear // HIDDEN_SIZE
        column = linear % HIDDEN_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(MLP_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        output[row, column] = (
            residual[row, column].to(cutlass.Float32)
            + accumulator * (MLP_SCALE * WEIGHT_MLP_SCALE)
            + bias[column].to(cutlass.Float32)
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
    layer_norm_fp8_kernel(
        hidden, ln1_weight, ln1_bias, norm1_workspace
    ).launch(grid=(TOKENS, 1, 1), block=(32, 1, 1))
    qkv_projection_kernel(
        norm1_workspace, qkv_weight, qkv_bias, qkv_workspace
    ).launch(
        grid=((TOKENS * QKV_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    attention_score_kernel(qkv_workspace, score_workspace).launch(
        grid=(
            (NUM_HEADS * TOKENS * TOKENS + THREADS - 1) // THREADS,
            1,
            1,
        ),
        block=(THREADS, 1, 1),
    )
    causal_softmax_kernel(
        score_workspace, probability_workspace
    ).launch(
        grid=(NUM_HEADS * TOKENS, 1, 1),
        block=(1, 1, 1),
    )
    attention_context_kernel(
        qkv_workspace, probability_workspace, context_workspace
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    attention_projection_kernel(
        hidden,
        context_workspace,
        out_weight,
        out_bias,
        residual_workspace,
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    layer_norm_fp32_kernel(
        residual_workspace, ln2_weight, ln2_bias, norm2_workspace
    ).launch(grid=(TOKENS, 1, 1), block=(32, 1, 1))
    mlp_fc_gelu_kernel(
        norm2_workspace, fc_weight, fc_bias, mlp_workspace
    ).launch(
        grid=((TOKENS * MLP_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    mlp_projection_kernel(
        residual_workspace, mlp_workspace, proj_weight, proj_bias, output
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
