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


@cute.jit
def warp_sum(value):
    # TODO: warp butterfly reduction.
    return value


@cute.kernel
def rms_norm_kernel(hidden: cute.Tensor, weight: cute.Tensor, output: cute.Tensor):
    pass


@cute.kernel
def attention_kernel(
    hidden: cute.Tensor,
    q_gate_weight: cute.Tensor,
    k_weight: cute.Tensor,
    v_weight: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def swiglu_kernel(
    hidden: cute.Tensor,
    gate_weight: cute.Tensor,
    up_weight: cute.Tensor,
    down_weight: cute.Tensor,
    output: cute.Tensor,
):
    pass


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
    # TODO: launch the complete block in the order specified by TASK.md.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _cute_harness_torch
import torch.nn.functional as _cute_harness_functional

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import from_dlpack as _cute_harness_from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8 as _cute_harness_create_fp8


_CUTE_HARNESS_TOKENS = 128
_CUTE_HARNESS_HIDDEN = 2560
_CUTE_HARNESS_HEADS = 16
_CUTE_HARNESS_KV_HEADS = 4
_CUTE_HARNESS_HEAD_DIM = 256
_CUTE_HARNESS_ROTARY = 64
_CUTE_HARNESS_INTERMEDIATE = 9216
_CUTE_HARNESS_Q_GATE = 8192
_CUTE_HARNESS_KV = 1024
_CUTE_HARNESS_QUERY = 4096
_CUTE_HARNESS_EPSILON = 1.0e-6
_CUTE_HARNESS_INPUT_SCALE = 1.0 / 448.0
_CUTE_HARNESS_NORM_SCALE = 1.0 / 64.0
_CUTE_HARNESS_QKV_SCALE = 1.0 / 64.0
_CUTE_HARNESS_CONTEXT_SCALE = 1.0 / 64.0
_CUTE_HARNESS_MLP_PROJECTION_SCALE = 1.0 / 64.0
_CUTE_HARNESS_MLP_ACTIVATION_SCALE = 1.0 / 32.0
_CUTE_HARNESS_WEIGHT_H_BOUND = _CUTE_HARNESS_HIDDEN ** -0.5
_CUTE_HARNESS_WEIGHT_CONTEXT_BOUND = _CUTE_HARNESS_QUERY ** -0.5
_CUTE_HARNESS_WEIGHT_MLP_BOUND = _CUTE_HARNESS_INTERMEDIATE ** -0.5
_CUTE_HARNESS_WEIGHT_H_SCALE = _CUTE_HARNESS_WEIGHT_H_BOUND / 448.0
_CUTE_HARNESS_WEIGHT_CONTEXT_SCALE = (
    _CUTE_HARNESS_WEIGHT_CONTEXT_BOUND / 448.0
)
_CUTE_HARNESS_WEIGHT_MLP_SCALE = _CUTE_HARNESS_WEIGHT_MLP_BOUND / 448.0
_CUTE_HARNESS_FP8_DTYPE = _cute_harness_cutlass.Float8E4M3FN


def _cute_harness_fp8_tensor(source, scale):
    storage = _cute_harness_torch.empty(
        source.shape, device="cuda", dtype=_cute_harness_torch.uint8
    )
    tensor = _cute_harness_create_fp8(
        storage, _CUTE_HARNESS_FP8_DTYPE, 1, source / scale
    )
    return storage, tensor


def _cute_harness_empty_fp8(shape):
    return _cute_harness_fp8_tensor(
        _cute_harness_torch.zeros(shape, device="cuda"), 1.0
    )


def _cute_harness_scaled_linear(activation, weight, activation_scale, weight_scale):
    scale_a = _cute_harness_torch.tensor(activation_scale, device="cuda")
    scale_b = _cute_harness_torch.tensor(weight_scale, device="cuda")
    return _cute_harness_torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=_cute_harness_torch.float32,
    )


def _cute_harness_rms_norm(hidden, weight):
    inverse_rms = _cute_harness_torch.rsqrt(
        hidden.square().mean(dim=-1, keepdim=True) + _CUTE_HARNESS_EPSILON
    )
    return hidden * inverse_rms * (1.0 + weight)


def _cute_harness_apply_rope(hidden, cos, sin):
    rotary = hidden[..., :_CUTE_HARNESS_ROTARY]
    half = _CUTE_HARNESS_ROTARY // 2
    rotated = _cute_harness_torch.cat(
        (-rotary[..., half:], rotary[..., :half]), dim=-1
    )
    embedded = rotary * cos[:, None, :] + rotated * sin[:, None, :]
    return _cute_harness_torch.cat(
        (embedded, hidden[..., _CUTE_HARNESS_ROTARY:]), dim=-1
    )


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    hidden_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN), device="cuda"
    ).uniform_(-1.0, 1.0)
    input_norm_weight = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    post_norm_weight = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    q_norm_weight = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HEAD_DIM,), device="cuda"
    )
    k_norm_weight = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HEAD_DIM,), device="cuda"
    )

    positions = _cute_harness_torch.arange(
        _CUTE_HARNESS_TOKENS, device="cuda", dtype=_cute_harness_torch.float32
    )
    rotary_indices = _cute_harness_torch.arange(
        0,
        _CUTE_HARNESS_ROTARY,
        2,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    inverse_frequency = 1.0 / (
        10_000_000.0 ** (rotary_indices / _CUTE_HARNESS_ROTARY)
    )
    frequencies = positions[:, None] * inverse_frequency[None, :]
    embeddings = _cute_harness_torch.cat((frequencies, frequencies), dim=-1)
    cos_storage = embeddings.cos()
    sin_storage = embeddings.sin()

    weight_specs = (
        ("q_gate", _CUTE_HARNESS_Q_GATE, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("k", _CUTE_HARNESS_KV, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("v", _CUTE_HARNESS_KV, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("out", _CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_QUERY,
         _CUTE_HARNESS_WEIGHT_CONTEXT_BOUND, _CUTE_HARNESS_WEIGHT_CONTEXT_SCALE),
        ("gate", _CUTE_HARNESS_INTERMEDIATE, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("up", _CUTE_HARNESS_INTERMEDIATE, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("down", _CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_INTERMEDIATE,
         _CUTE_HARNESS_WEIGHT_MLP_BOUND, _CUTE_HARNESS_WEIGHT_MLP_SCALE),
    )
    weight_storages = {}
    weight_tensors = {}
    for name, rows, columns, bound, scale in weight_specs:
        source = _cute_harness_torch.empty(
            (rows, columns), device="cuda"
        ).uniform_(-bound, bound)
        storage, tensor = _cute_harness_fp8_tensor(source, scale)
        weight_storages[name] = storage
        weight_tensors[name] = tensor

    hidden_storage, hidden = _cute_harness_fp8_tensor(
        hidden_source, _CUTE_HARNESS_INPUT_SCALE
    )
    fp8_workspace_shapes = (
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_Q_GATE),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_KV),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_KV),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QUERY),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_KV),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QUERY),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_INTERMEDIATE),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_INTERMEDIATE),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_INTERMEDIATE),
    )
    fp8_workspaces = [_cute_harness_empty_fp8(shape) for shape in fp8_workspace_shapes]
    (
        norm1_pair,
        q_gate_pair,
        k_pair,
        v_pair,
        query_pair,
        key_pair,
        context_pair,
        norm2_pair,
        gate_pair,
        up_pair,
        mlp_pair,
    ) = fp8_workspaces

    score_storage = _cute_harness_torch.empty(
        (_CUTE_HARNESS_HEADS * _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_TOKENS),
        device="cuda",
    )
    probability_storage = _cute_harness_torch.empty_like(score_storage)
    residual_storage = _cute_harness_torch.empty(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN), device="cuda"
    )
    output_storage = _cute_harness_torch.empty_like(residual_storage)
    dynamic = lambda tensor: _cute_harness_from_dlpack(tensor).mark_layout_dynamic(
        leading_dim=1
    )
    arguments = (
        hidden,
        _cute_harness_from_dlpack(input_norm_weight),
        _cute_harness_from_dlpack(q_norm_weight),
        _cute_harness_from_dlpack(k_norm_weight),
        weight_tensors["q_gate"],
        weight_tensors["k"],
        weight_tensors["v"],
        weight_tensors["out"],
        _cute_harness_from_dlpack(post_norm_weight),
        weight_tensors["gate"],
        weight_tensors["up"],
        weight_tensors["down"],
        dynamic(cos_storage),
        dynamic(sin_storage),
        *(pair[1] for pair in fp8_workspaces[:6]),
        dynamic(score_storage),
        dynamic(probability_storage),
        context_pair[1],
        dynamic(residual_storage),
        *(pair[1] for pair in fp8_workspaces[7:]),
        dynamic(output_storage),
    )
    compiled = _cute_harness_cute.compile(qwen35_full_attention_block, *arguments)
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(*arguments)
    _cute_harness_torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(*arguments)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    fp8 = _cute_harness_torch.float8_e4m3fn
    hidden_fp8 = hidden_storage.view(fp8)
    weights_fp8 = {name: storage.view(fp8) for name, storage in weight_storages.items()}
    hidden_dequantized = hidden_fp8.float() * _CUTE_HARNESS_INPUT_SCALE
    norm1 = _cute_harness_rms_norm(hidden_dequantized, input_norm_weight)
    norm1_fp8 = (norm1 / _CUTE_HARNESS_NORM_SCALE).to(fp8)
    q_gate = _cute_harness_scaled_linear(
        norm1_fp8, weights_fp8["q_gate"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
    )
    k = _cute_harness_scaled_linear(
        norm1_fp8, weights_fp8["k"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
    )
    v = _cute_harness_scaled_linear(
        norm1_fp8, weights_fp8["v"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
    )
    q_gate_fp8 = (q_gate / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    k_fp8 = (k / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    v_fp8 = (v / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    q_gate_heads = (q_gate_fp8.float() * _CUTE_HARNESS_QKV_SCALE).reshape(
        _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HEADS, 2, _CUTE_HARNESS_HEAD_DIM
    )
    query = _cute_harness_rms_norm(q_gate_heads[:, :, 0], q_norm_weight)
    gate = q_gate_heads[:, :, 1]
    key = _cute_harness_rms_norm(
        (k_fp8.float() * _CUTE_HARNESS_QKV_SCALE).reshape(
            _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_KV_HEADS, _CUTE_HARNESS_HEAD_DIM
        ),
        k_norm_weight,
    )
    query = _cute_harness_apply_rope(query, cos_storage, sin_storage)
    key = _cute_harness_apply_rope(key, cos_storage, sin_storage)
    query_fp8 = (query / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    key_fp8 = (key / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    query = query_fp8.float() * _CUTE_HARNESS_QKV_SCALE
    key = key_fp8.float() * _CUTE_HARNESS_QKV_SCALE
    key = key.repeat_interleave(
        _CUTE_HARNESS_HEADS // _CUTE_HARNESS_KV_HEADS, dim=1
    )
    value = (v_fp8.float() * _CUTE_HARNESS_QKV_SCALE).reshape(
        _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_KV_HEADS, _CUTE_HARNESS_HEAD_DIM
    ).repeat_interleave(_CUTE_HARNESS_HEADS // _CUTE_HARNESS_KV_HEADS, dim=1)
    scores = _cute_harness_torch.einsum("thd,shd->hts", query, key) / 16.0
    causal_mask = _cute_harness_torch.triu(
        _cute_harness_torch.ones(
            (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_TOKENS),
            device="cuda",
            dtype=_cute_harness_torch.bool,
        ),
        diagonal=1,
    )
    probabilities = scores.masked_fill(causal_mask, -float("inf")).softmax(dim=-1)
    context = _cute_harness_torch.einsum(
        "hts,shd->thd", probabilities, value
    ) * _cute_harness_torch.sigmoid(gate)
    context_fp8 = (context.reshape(
        _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QUERY
    ) / _CUTE_HARNESS_CONTEXT_SCALE).to(fp8)
    residual = hidden_dequantized + _cute_harness_scaled_linear(
        context_fp8, weights_fp8["out"], _CUTE_HARNESS_CONTEXT_SCALE,
        _CUTE_HARNESS_WEIGHT_CONTEXT_SCALE,
    )
    norm2 = _cute_harness_rms_norm(residual, post_norm_weight)
    norm2_fp8 = (norm2 / _CUTE_HARNESS_NORM_SCALE).to(fp8)
    gate_mlp = _cute_harness_scaled_linear(
        norm2_fp8, weights_fp8["gate"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
    )
    up_mlp = _cute_harness_scaled_linear(
        norm2_fp8, weights_fp8["up"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
    )
    gate_fp8 = (gate_mlp / _CUTE_HARNESS_MLP_PROJECTION_SCALE).to(fp8)
    up_fp8 = (up_mlp / _CUTE_HARNESS_MLP_PROJECTION_SCALE).to(fp8)
    activation = _cute_harness_functional.silu(
        gate_fp8.float() * _CUTE_HARNESS_MLP_PROJECTION_SCALE
    ) * (up_fp8.float() * _CUTE_HARNESS_MLP_PROJECTION_SCALE)
    activation_fp8 = (activation / _CUTE_HARNESS_MLP_ACTIVATION_SCALE).to(fp8)
    reference = residual + _cute_harness_scaled_linear(
        activation_fp8, weights_fp8["down"], _CUTE_HARNESS_MLP_ACTIVATION_SCALE,
        _CUTE_HARNESS_WEIGHT_MLP_SCALE,
    )

    max_abs = (output_storage - reference).abs().max().item()
    if not _cute_harness_torch.isfinite(output_storage).all().item() or max_abs > 0.08:
        raise RuntimeError(f"validation failed: max_abs={max_abs:.6f}")
    print(
        "task=model_qwen35_4b_full_attention_block_fp8 "
        f"max_abs={max_abs:.6f} kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
