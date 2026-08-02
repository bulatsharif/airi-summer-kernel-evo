import cutlass
import cutlass.cute as cute


TOKENS = 128
HIDDEN_SIZE = 1024
NUM_HEADS = 16
HEAD_DIM = 64
INTERMEDIATE_SIZE = 4096
QKV_SIZE = 3 * HIDDEN_SIZE
THREADS = 128
EPSILON = 1.0e-6

INPUT_SCALE = 1.0 / 448.0
NORM_SCALE = 1.0 / 64.0
QKV_SCALE = 1.0 / 64.0
CONTEXT_SCALE = 1.0 / 64.0
MLP_SCALE = 1.0 / 64.0
WEIGHT_H_SCALE = HIDDEN_SIZE ** -0.5 / 448.0
WEIGHT_MLP_SCALE = INTERMEDIATE_SIZE ** -0.5 / 448.0
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.jit
def warp_sum(value):
    # TODO: warp butterfly reduction.
    return value


@cute.kernel
def layer_norm_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def vision_attention_kernel(
    hidden: cute.Tensor,
    qkv_weight: cute.Tensor,
    qkv_bias: cute.Tensor,
    cos: cute.Tensor,
    sin: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def vision_mlp_kernel(
    hidden: cute.Tensor,
    fc1_weight: cute.Tensor,
    fc1_bias: cute.Tensor,
    fc2_weight: cute.Tensor,
    fc2_bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


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
_CUTE_HARNESS_HIDDEN = 1024
_CUTE_HARNESS_HEADS = 16
_CUTE_HARNESS_HEAD_DIM = 64
_CUTE_HARNESS_INTERMEDIATE = 4096
_CUTE_HARNESS_QKV = 3072
_CUTE_HARNESS_EPSILON = 1.0e-6
_CUTE_HARNESS_INPUT_SCALE = 1.0 / 448.0
_CUTE_HARNESS_NORM_SCALE = 1.0 / 64.0
_CUTE_HARNESS_QKV_SCALE = 1.0 / 64.0
_CUTE_HARNESS_CONTEXT_SCALE = 1.0 / 64.0
_CUTE_HARNESS_MLP_SCALE = 1.0 / 64.0
_CUTE_HARNESS_WEIGHT_H_BOUND = _CUTE_HARNESS_HIDDEN ** -0.5
_CUTE_HARNESS_WEIGHT_MLP_BOUND = _CUTE_HARNESS_INTERMEDIATE ** -0.5
_CUTE_HARNESS_WEIGHT_H_SCALE = _CUTE_HARNESS_WEIGHT_H_BOUND / 448.0
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


def _cute_harness_scaled_linear(
    activation, weight, activation_scale, weight_scale, bias
):
    scale_a = _cute_harness_torch.tensor(activation_scale, device="cuda")
    scale_b = _cute_harness_torch.tensor(weight_scale, device="cuda")
    return _cute_harness_torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=_cute_harness_torch.float32,
    ) + bias


def _cute_harness_apply_rope(hidden, cos, sin):
    half = _CUTE_HARNESS_HEAD_DIM // 2
    rotated = _cute_harness_torch.cat(
        (-hidden[..., half:], hidden[..., :half]), dim=-1
    )
    return hidden * cos[:, None, :] + rotated * sin[:, None, :]


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    hidden_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN), device="cuda"
    ).uniform_(-1.0, 1.0)
    norm1_weight = 1.0 + 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    norm1_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    norm2_weight = 1.0 + 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    norm2_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    qkv_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_QKV,), device="cuda"
    )
    out_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    fc1_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_INTERMEDIATE,), device="cuda"
    )
    fc2_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )

    positions = _cute_harness_torch.arange(
        _CUTE_HARNESS_TOKENS, device="cuda", dtype=_cute_harness_torch.float32
    )
    rotary_indices = _cute_harness_torch.arange(
        0,
        _CUTE_HARNESS_HEAD_DIM,
        2,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    inverse_frequency = 1.0 / (
        10_000.0 ** (rotary_indices / _CUTE_HARNESS_HEAD_DIM)
    )
    frequencies = positions[:, None] * inverse_frequency[None, :]
    embeddings = _cute_harness_torch.cat((frequencies, frequencies), dim=-1)
    cos_storage = embeddings.cos()
    sin_storage = embeddings.sin()

    weight_specs = (
        ("qkv", _CUTE_HARNESS_QKV, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("out", _CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("fc1", _CUTE_HARNESS_INTERMEDIATE, _CUTE_HARNESS_HIDDEN,
         _CUTE_HARNESS_WEIGHT_H_BOUND, _CUTE_HARNESS_WEIGHT_H_SCALE),
        ("fc2", _CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_INTERMEDIATE,
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
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QKV),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QKV),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_INTERMEDIATE),
    )
    fp8_workspaces = [_cute_harness_empty_fp8(shape) for shape in fp8_workspace_shapes]
    norm1_pair, qkv_pair, rotary_pair, context_pair, norm2_pair, mlp_pair = (
        fp8_workspaces
    )
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
        _cute_harness_from_dlpack(norm1_weight),
        _cute_harness_from_dlpack(norm1_bias),
        weight_tensors["qkv"],
        _cute_harness_from_dlpack(qkv_bias),
        weight_tensors["out"],
        _cute_harness_from_dlpack(out_bias),
        _cute_harness_from_dlpack(norm2_weight),
        _cute_harness_from_dlpack(norm2_bias),
        weight_tensors["fc1"],
        _cute_harness_from_dlpack(fc1_bias),
        weight_tensors["fc2"],
        _cute_harness_from_dlpack(fc2_bias),
        dynamic(cos_storage),
        dynamic(sin_storage),
        norm1_pair[1],
        qkv_pair[1],
        rotary_pair[1],
        dynamic(score_storage),
        dynamic(probability_storage),
        context_pair[1],
        dynamic(residual_storage),
        norm2_pair[1],
        mlp_pair[1],
        dynamic(output_storage),
    )
    compiled = _cute_harness_cute.compile(qwen35_vision_block, *arguments)
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
    norm1 = _cute_harness_functional.layer_norm(
        hidden_dequantized,
        (_CUTE_HARNESS_HIDDEN,),
        norm1_weight,
        norm1_bias,
        _CUTE_HARNESS_EPSILON,
    )
    norm1_fp8 = (norm1 / _CUTE_HARNESS_NORM_SCALE).to(fp8)
    qkv = _cute_harness_scaled_linear(
        norm1_fp8, weights_fp8["qkv"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE, qkv_bias,
    )
    qkv_fp8 = (qkv / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    query, key, value = (qkv_fp8.float() * _CUTE_HARNESS_QKV_SCALE).reshape(
        _CUTE_HARNESS_TOKENS,
        3,
        _CUTE_HARNESS_HEADS,
        _CUTE_HARNESS_HEAD_DIM,
    ).unbind(dim=1)
    query = _cute_harness_apply_rope(query, cos_storage, sin_storage)
    key = _cute_harness_apply_rope(key, cos_storage, sin_storage)
    rotary_qkv_fp8 = (_cute_harness_torch.stack(
        (query, key, value), dim=1
    ).reshape(_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QKV) / _CUTE_HARNESS_QKV_SCALE).to(fp8)
    query, key, value = (
        rotary_qkv_fp8.float() * _CUTE_HARNESS_QKV_SCALE
    ).reshape(
        _CUTE_HARNESS_TOKENS,
        3,
        _CUTE_HARNESS_HEADS,
        _CUTE_HARNESS_HEAD_DIM,
    ).unbind(dim=1)
    scores = _cute_harness_torch.einsum("thd,shd->hts", query, key) / 8.0
    probabilities = scores.softmax(dim=-1)
    context = _cute_harness_torch.einsum(
        "hts,shd->thd", probabilities, value
    ).reshape(_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    context_fp8 = (context / _CUTE_HARNESS_CONTEXT_SCALE).to(fp8)
    residual = hidden_dequantized + _cute_harness_scaled_linear(
        context_fp8, weights_fp8["out"], _CUTE_HARNESS_CONTEXT_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE, out_bias,
    )
    norm2 = _cute_harness_functional.layer_norm(
        residual,
        (_CUTE_HARNESS_HIDDEN,),
        norm2_weight,
        norm2_bias,
        _CUTE_HARNESS_EPSILON,
    )
    norm2_fp8 = (norm2 / _CUTE_HARNESS_NORM_SCALE).to(fp8)
    fc1 = _cute_harness_scaled_linear(
        norm2_fp8, weights_fp8["fc1"], _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE, fc1_bias,
    )
    mlp_fp8 = (
        _cute_harness_functional.gelu(fc1, approximate="tanh")
        / _CUTE_HARNESS_MLP_SCALE
    ).to(fp8)
    reference = residual + _cute_harness_scaled_linear(
        mlp_fp8, weights_fp8["fc2"], _CUTE_HARNESS_MLP_SCALE,
        _CUTE_HARNESS_WEIGHT_MLP_SCALE, fc2_bias,
    )

    max_abs = (output_storage - reference).abs().max().item()
    if not _cute_harness_torch.isfinite(output_storage).all().item() or max_abs > 0.06:
        raise RuntimeError(f"validation failed: max_abs={max_abs:.6f}")
    print(
        "task=model_qwen35_4b_vision_block_fp8 "
        f"max_abs={max_abs:.6f} kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
