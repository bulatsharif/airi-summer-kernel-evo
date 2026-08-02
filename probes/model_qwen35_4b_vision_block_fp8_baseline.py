import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE_SIZE = 4096
QKV_SIZE = 3072
THREADS = 128
EPSILON = 1.0e-6

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = INTERMEDIATE_SIZE ** -0.5 / 448.0


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def input_layer_norm_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    summed = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        summed += hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
    mean = warp_sum(summed) / HIDDEN_SIZE
    variance = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        centered = hidden[row, column].to(cutlass.Float32) * INPUT_SCALE - mean
        variance += centered * centered
    inverse_std = cute.rsqrt(warp_sum(variance) / HIDDEN_SIZE + EPSILON)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
        output[row, column] = (
            (value - mean) * inverse_std * weight[column].to(cutlass.Float32)
            + bias[column].to(cutlass.Float32)
        ) / NORM_SCALE


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
            accumulator * NORM_SCALE * WEIGHT_H_SCALE
            + bias[column].to(cutlass.Float32)
        ) / QKV_SCALE


@cute.kernel
def rotary_qkv_kernel(
    qkv: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * QKV_SIZE:
        row = linear // QKV_SIZE
        column = linear % QKV_SIZE
        component = column // HIDDEN_SIZE
        head_column = column % HEAD_DIM
        value = qkv[row, column].to(cutlass.Float32) * QKV_SCALE
        if component < 2:
            half = HEAD_DIM // 2
            paired_column = column - half
            if head_column < half:
                paired_column = column + half
            rotated = qkv[row, paired_column].to(cutlass.Float32) * QKV_SCALE
            if head_column < half:
                rotated = -rotated
            value = (
                value * cos[row, head_column].to(cutlass.Float32)
                + rotated * sin[row, head_column].to(cutlass.Float32)
            )
        output[row, column] = value / QKV_SCALE


@cute.kernel
def attention_score_kernel(qkv: cute.Tensor, scores: cute.Tensor):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < NUM_HEADS * TOKENS * TOKENS:
        key_token = linear % TOKENS
        row = linear // TOKENS
        query_token = row % TOKENS
        head = row // TOKENS
        accumulator = cutlass.Float32(0.0)
        head_start = head * HEAD_DIM
        for column in cutlass.range(HEAD_DIM):
            accumulator += (
                qkv[query_token, head_start + column].to(cutlass.Float32)
                * qkv[key_token, HIDDEN_SIZE + head_start + column].to(
                    cutlass.Float32
                )
            )
        scores[row, key_token] = accumulator * QKV_SCALE * QKV_SCALE / 8.0


@cute.kernel
def softmax_kernel(scores: cute.Tensor, probabilities: cute.Tensor):
    row, _, _ = cute.arch.block_idx()
    maximum = scores[row, 0].to(cutlass.Float32)
    for column in cutlass.range(TOKENS):
        value = scores[row, column].to(cutlass.Float32)
        if value > maximum:
            maximum = value
    denominator = cutlass.Float32(0.0)
    for column in cutlass.range(TOKENS):
        denominator += cute.exp(scores[row, column].to(cutlass.Float32) - maximum)
    for column in cutlass.range(TOKENS):
        probabilities[row, column] = cute.exp(
            scores[row, column].to(cutlass.Float32) - maximum
        ) / denominator


@cute.kernel
def context_kernel(
    qkv: cute.Tensor,
    probabilities: cute.Tensor,
    context: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * HIDDEN_SIZE:
        query_token = linear // HIDDEN_SIZE
        column = linear % HIDDEN_SIZE
        head = column // HEAD_DIM
        accumulator = cutlass.Float32(0.0)
        for key_token in cutlass.range(TOKENS):
            accumulator += (
                probabilities[head * TOKENS + query_token, key_token].to(
                    cutlass.Float32
                )
                * qkv[key_token, 2 * HIDDEN_SIZE + column].to(cutlass.Float32)
            )
        context[query_token, column] = accumulator * QKV_SCALE / CONTEXT_SCALE


@cute.kernel
def attention_output_kernel(
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
            + accumulator * CONTEXT_SCALE * WEIGHT_H_SCALE
            + bias[column].to(cutlass.Float32)
        )


@cute.kernel
def residual_layer_norm_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    summed = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        summed += hidden[row, column].to(cutlass.Float32)
    mean = warp_sum(summed) / HIDDEN_SIZE
    variance = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        centered = hidden[row, column].to(cutlass.Float32) - mean
        variance += centered * centered
    inverse_std = cute.rsqrt(warp_sum(variance) / HIDDEN_SIZE + EPSILON)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        normalized = (
            hidden[row, column].to(cutlass.Float32) - mean
        ) * inverse_std
        output[row, column] = (
            normalized * weight[column].to(cutlass.Float32)
            + bias[column].to(cutlass.Float32)
        ) / NORM_SCALE


@cute.kernel
def fc1_gelu_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * INTERMEDIATE_SIZE:
        row = linear // INTERMEDIATE_SIZE
        column = linear % INTERMEDIATE_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(HIDDEN_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        value = (
            accumulator * NORM_SCALE * WEIGHT_H_SCALE
            + bias[column].to(cutlass.Float32)
        )
        cubic = value * value * value
        output[row, column] = (
            0.5
            * value
            * (1.0 + cute.tanh(0.7978845608028654 * (value + 0.044715 * cubic)))
            / MLP_SCALE
        )


@cute.kernel
def fc2_kernel(
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
        for k in cutlass.range(INTERMEDIATE_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        output[row, column] = (
            residual[row, column].to(cutlass.Float32)
            + accumulator * MLP_SCALE * WEIGHT_MLP_SCALE
            + bias[column].to(cutlass.Float32)
        )


@cute.jit
def qwen35_vision_block(
    hidden: cute.Tensor,
    norm1_weight: cute.Tensor,
    norm1_bias: cute.Tensor,
    qkv_weight: cute.Tensor,
    qkv_bias: cute.Tensor,
    out_weight: cute.Tensor,
    out_bias: cute.Tensor,
    norm2_weight: cute.Tensor,
    norm2_bias: cute.Tensor,
    fc1_weight: cute.Tensor,
    fc1_bias: cute.Tensor,
    fc2_weight: cute.Tensor,
    fc2_bias: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    norm1_workspace: cute.Tensor,
    qkv_workspace: cute.Tensor,
    rotary_qkv_workspace: cute.Tensor,
    score_workspace: cute.Tensor,
    probability_workspace: cute.Tensor,
    context_workspace: cute.Tensor,
    residual_workspace: cute.Tensor,
    norm2_workspace: cute.Tensor,
    mlp_workspace: cute.Tensor,
    output: cute.Tensor,
):
    input_layer_norm_kernel(
        hidden, norm1_weight, norm1_bias, norm1_workspace
    ).launch(grid=(TOKENS, 1, 1), block=(32, 1, 1))
    qkv_projection_kernel(
        norm1_workspace, qkv_weight, qkv_bias, qkv_workspace
    ).launch(
        grid=((TOKENS * QKV_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    rotary_qkv_kernel(qkv_workspace, cos, sin, rotary_qkv_workspace).launch(
        grid=((TOKENS * QKV_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    attention_score_kernel(rotary_qkv_workspace, score_workspace).launch(
        grid=((NUM_HEADS * TOKENS * TOKENS + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    softmax_kernel(score_workspace, probability_workspace).launch(
        grid=(NUM_HEADS * TOKENS, 1, 1), block=(1, 1, 1)
    )
    context_kernel(
        rotary_qkv_workspace, probability_workspace, context_workspace
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    attention_output_kernel(
        hidden, context_workspace, out_weight, out_bias, residual_workspace
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    residual_layer_norm_kernel(
        residual_workspace, norm2_weight, norm2_bias, norm2_workspace
    ).launch(grid=(TOKENS, 1, 1), block=(32, 1, 1))
    fc1_gelu_kernel(
        norm2_workspace, fc1_weight, fc1_bias, mlp_workspace
    ).launch(
        grid=((TOKENS * INTERMEDIATE_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    fc2_kernel(
        residual_workspace, mlp_workspace, fc2_weight, fc2_bias, output
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
