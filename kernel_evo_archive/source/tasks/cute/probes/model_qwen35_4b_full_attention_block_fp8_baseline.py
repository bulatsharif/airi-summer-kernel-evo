import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 2560
NUM_HEADS = 16
NUM_KV_HEADS = 4
HEAD_DIM = 256
ROTARY_DIM = 64
INTERMEDIATE_SIZE = 9216
Q_GATE_SIZE = 8192
KV_SIZE = 1024
QUERY_SIZE = 4096
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


@cute.jit
def warp_sum(value):
    value += cute.arch.shuffle_sync_bfly(value, 16)
    value += cute.arch.shuffle_sync_bfly(value, 8)
    value += cute.arch.shuffle_sync_bfly(value, 4)
    value += cute.arch.shuffle_sync_bfly(value, 2)
    value += cute.arch.shuffle_sync_bfly(value, 1)
    return value


@cute.kernel
def input_rms_norm_kernel(
    hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    squared = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
        squared += value * value
    inverse_rms = cute.rsqrt(warp_sum(squared) / HIDDEN_SIZE + EPSILON)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
        output[row, column] = (
            (
            value * inverse_rms * (1.0 + weight[column].to(cutlass.Float32))
        ) / NORM_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def attention_input_projection_kernel(
    hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    output_size = cute.size(output, mode=[1])
    linear = block * THREADS + thread
    if linear < TOKENS * output_size:
        row = linear // output_size
        column = linear % output_size
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(HIDDEN_SIZE):
            accumulator += (
                hidden[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        output[row, column] = (
            (
            accumulator * NORM_SCALE * WEIGHT_H_SCALE / QKV_SCALE
        )
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def query_norm_rope_kernel(
    q_gate: cute.Tensor,
    weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    row = block // NUM_HEADS
    head = block % NUM_HEADS
    start = head * 2 * HEAD_DIM
    squared = cutlass.Float32(0.0)
    for iteration in cutlass.range(HEAD_DIM // 32):
        column = iteration * 32 + lane
        value = q_gate[row, start + column].to(cutlass.Float32) * QKV_SCALE
        squared += value * value
    inverse_rms = cute.rsqrt(warp_sum(squared) / HEAD_DIM + EPSILON)
    for iteration in cutlass.range(HEAD_DIM // 32):
        column = iteration * 32 + lane
        value = q_gate[row, start + column].to(cutlass.Float32) * QKV_SCALE
        value *= inverse_rms * (1.0 + weight[column].to(cutlass.Float32))
        if column < ROTARY_DIM:
            half = ROTARY_DIM // 2
            paired = column - half
            if column < half:
                paired = column + half
            rotated = q_gate[row, start + paired].to(cutlass.Float32) * QKV_SCALE
            rotated *= inverse_rms * (1.0 + weight[paired].to(cutlass.Float32))
            if column < half:
                rotated = -rotated
            value = (
                value * cos[row, column].to(cutlass.Float32)
                + rotated * sin[row, column].to(cutlass.Float32)
            )
        output[row, head * HEAD_DIM + column] = (
            value / QKV_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def key_norm_rope_kernel(
    key: cute.Tensor,
    weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
):
    lane, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    row = block // NUM_KV_HEADS
    head = block % NUM_KV_HEADS
    start = head * HEAD_DIM
    squared = cutlass.Float32(0.0)
    for iteration in cutlass.range(HEAD_DIM // 32):
        column = iteration * 32 + lane
        value = key[row, start + column].to(cutlass.Float32) * QKV_SCALE
        squared += value * value
    inverse_rms = cute.rsqrt(warp_sum(squared) / HEAD_DIM + EPSILON)
    for iteration in cutlass.range(HEAD_DIM // 32):
        column = iteration * 32 + lane
        value = key[row, start + column].to(cutlass.Float32) * QKV_SCALE
        value *= inverse_rms * (1.0 + weight[column].to(cutlass.Float32))
        if column < ROTARY_DIM:
            half = ROTARY_DIM // 2
            paired = column - half
            if column < half:
                paired = column + half
            rotated = key[row, start + paired].to(cutlass.Float32) * QKV_SCALE
            rotated *= inverse_rms * (1.0 + weight[paired].to(cutlass.Float32))
            if column < half:
                rotated = -rotated
            value = (
                value * cos[row, column].to(cutlass.Float32)
                + rotated * sin[row, column].to(cutlass.Float32)
            )
        output[row, start + column] = (
            value / QKV_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def attention_score_kernel(
    query: cute.Tensor, key: cute.Tensor, scores: cute.Tensor
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < NUM_HEADS * TOKENS * TOKENS:
        key_token = linear % TOKENS
        row = linear // TOKENS
        query_token = row % TOKENS
        head = row // TOKENS
        if key_token <= query_token:
            accumulator = cutlass.Float32(0.0)
            query_start = head * HEAD_DIM
            key_start = (head // 4) * HEAD_DIM
            for column in cutlass.range(HEAD_DIM):
                accumulator += (
                    query[query_token, query_start + column].to(cutlass.Float32)
                    * key[key_token, key_start + column].to(cutlass.Float32)
                )
            scores[row, key_token] = accumulator * QKV_SCALE * QKV_SCALE / 16.0
        else:
            scores[row, key_token] = -cutlass.Float32(1.0e30)


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
def gated_context_kernel(
    q_gate: cute.Tensor,
    value: cute.Tensor,
    probabilities: cute.Tensor,
    context: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * QUERY_SIZE:
        query_token = linear // QUERY_SIZE
        column = linear % QUERY_SIZE
        head = column // HEAD_DIM
        head_column = column % HEAD_DIM
        accumulator = cutlass.Float32(0.0)
        for key_token in cutlass.range(TOKENS):
            accumulator += (
                probabilities[head * TOKENS + query_token, key_token].to(
                    cutlass.Float32
                )
                * value[
                    key_token, (head // 4) * HEAD_DIM + head_column
                ].to(cutlass.Float32)
            )
        gate_column = head * 2 * HEAD_DIM + HEAD_DIM + head_column
        gate = q_gate[query_token, gate_column].to(cutlass.Float32) * QKV_SCALE
        sigmoid = 1.0 / (1.0 + cute.exp(-gate))
        context[query_token, column] = (
            (
            accumulator * QKV_SCALE * sigmoid / CONTEXT_SCALE
        )
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def attention_output_kernel(
    hidden: cute.Tensor,
    context: cute.Tensor,
    weight: cute.Tensor,
    residual: cute.Tensor,
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * HIDDEN_SIZE:
        row = linear // HIDDEN_SIZE
        column = linear % HIDDEN_SIZE
        accumulator = cutlass.Float32(0.0)
        for k in cutlass.range(QUERY_SIZE):
            accumulator += (
                context[row, k].to(cutlass.Float32)
                * weight[column, k].to(cutlass.Float32)
            )
        residual[row, column] = (
            hidden[row, column].to(cutlass.Float32) * INPUT_SCALE
            + accumulator * CONTEXT_SCALE * WEIGHT_CONTEXT_SCALE
        )


@cute.kernel
def residual_rms_norm_kernel(
    hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor
):
    lane, _, _ = cute.arch.thread_idx()
    row, _, _ = cute.arch.block_idx()
    squared = cutlass.Float32(0.0)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32)
        squared += value * value
    inverse_rms = cute.rsqrt(warp_sum(squared) / HIDDEN_SIZE + EPSILON)
    for iteration in cutlass.range(HIDDEN_SIZE // 32):
        column = iteration * 32 + lane
        value = hidden[row, column].to(cutlass.Float32)
        output[row, column] = (
            (
            value * inverse_rms * (1.0 + weight[column].to(cutlass.Float32))
        ) / NORM_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def mlp_input_projection_kernel(
    hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor
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
        output[row, column] = (
            (
            accumulator
            * NORM_SCALE
            * WEIGHT_H_SCALE
            / MLP_PROJECTION_SCALE
        )
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def swiglu_kernel(
    gate: cute.Tensor, up: cute.Tensor, output: cute.Tensor
):
    thread, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    linear = block * THREADS + thread
    if linear < TOKENS * INTERMEDIATE_SIZE:
        row = linear // INTERMEDIATE_SIZE
        column = linear % INTERMEDIATE_SIZE
        gate_value = gate[row, column].to(cutlass.Float32) * MLP_PROJECTION_SCALE
        up_value = up[row, column].to(cutlass.Float32) * MLP_PROJECTION_SCALE
        silu = gate_value / (1.0 + cute.exp(-gate_value))
        output[row, column] = (
            silu * up_value / MLP_ACTIVATION_SCALE
        ).to(cutlass.Float8E4M3FN)


@cute.kernel
def down_projection_kernel(
    residual: cute.Tensor,
    hidden: cute.Tensor,
    weight: cute.Tensor,
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
            + accumulator * MLP_ACTIVATION_SCALE * WEIGHT_MLP_SCALE
        )


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
    input_rms_norm_kernel(hidden, input_norm_weight, norm1_workspace).launch(
        grid=(TOKENS, 1, 1), block=(32, 1, 1)
    )
    for weight, workspace, size in (
        (q_gate_weight, q_gate_workspace, Q_GATE_SIZE),
        (k_weight, k_workspace, KV_SIZE),
        (v_weight, v_workspace, KV_SIZE),
    ):
        attention_input_projection_kernel(
            norm1_workspace, weight, workspace
        ).launch(
            grid=((TOKENS * size + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
        )
    query_norm_rope_kernel(
        q_gate_workspace, q_norm_weight, cos, sin, query_workspace
    ).launch(grid=(TOKENS * NUM_HEADS, 1, 1), block=(32, 1, 1))
    key_norm_rope_kernel(
        k_workspace, k_norm_weight, cos, sin, key_workspace
    ).launch(grid=(TOKENS * NUM_KV_HEADS, 1, 1), block=(32, 1, 1))
    attention_score_kernel(
        query_workspace, key_workspace, score_workspace
    ).launch(
        grid=((NUM_HEADS * TOKENS * TOKENS + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    softmax_kernel(score_workspace, probability_workspace).launch(
        grid=(NUM_HEADS * TOKENS, 1, 1), block=(1, 1, 1)
    )
    gated_context_kernel(
        q_gate_workspace, v_workspace, probability_workspace, context_workspace
    ).launch(
        grid=((TOKENS * QUERY_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    attention_output_kernel(
        hidden, context_workspace, out_weight, residual_workspace
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    residual_rms_norm_kernel(
        residual_workspace, post_norm_weight, norm2_workspace
    ).launch(grid=(TOKENS, 1, 1), block=(32, 1, 1))
    for weight, workspace in (
        (gate_weight, gate_workspace),
        (up_weight, up_workspace),
    ):
        mlp_input_projection_kernel(norm2_workspace, weight, workspace).launch(
            grid=((TOKENS * INTERMEDIATE_SIZE + THREADS - 1) // THREADS, 1, 1),
            block=(THREADS, 1, 1),
        )
    swiglu_kernel(gate_workspace, up_workspace, mlp_workspace).launch(
        grid=((TOKENS * INTERMEDIATE_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )
    down_projection_kernel(
        residual_workspace, mlp_workspace, down_weight, output
    ).launch(
        grid=((TOKENS * HIDDEN_SIZE + THREADS - 1) // THREADS, 1, 1),
        block=(THREADS, 1, 1),
    )


# === CUTE_HARNESS_EVALUATOR_V1 ===
