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
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.jit
def warp_sum(value):
    # TODO: warp butterfly reduction.
    return value


@cute.kernel
def layer_norm_fp8_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def qkv_projection_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def attention_score_kernel(qkv: cute.Tensor, scores: cute.Tensor):
    pass


@cute.kernel
def causal_softmax_kernel(
    scores: cute.Tensor,
    probabilities: cute.Tensor,
):
    pass


@cute.kernel
def attention_context_kernel(
    qkv: cute.Tensor,
    probabilities: cute.Tensor,
    context: cute.Tensor,
):
    pass


@cute.kernel
def attention_projection_kernel(
    hidden: cute.Tensor,
    context: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    residual: cute.Tensor,
):
    pass


@cute.kernel
def layer_norm_fp32_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def mlp_fc_gelu_kernel(
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


@cute.kernel
def mlp_projection_kernel(
    residual: cute.Tensor,
    hidden: cute.Tensor,
    weight: cute.Tensor,
    bias: cute.Tensor,
    output: cute.Tensor,
):
    pass


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
    # TODO: launch the complete block in the order specified by TASK.md.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import math as _cute_harness_math

import torch as _cute_harness_torch
import torch.nn.functional as _cute_harness_functional

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import from_dlpack as _cute_harness_from_dlpack
from cutlass.utils import (
    create_cute_tensor_for_fp8 as _cute_harness_create_fp8,
)


_CUTE_HARNESS_TOKENS = 128
_CUTE_HARNESS_HIDDEN = 768
_CUTE_HARNESS_HEADS = 12
_CUTE_HARNESS_HEAD_DIM = 64
_CUTE_HARNESS_MLP = 3072
_CUTE_HARNESS_QKV = 2304
_CUTE_HARNESS_EPSILON = 1.0e-5
_CUTE_HARNESS_INPUT_SCALE = 1.0 / 448.0
_CUTE_HARNESS_NORM_SCALE = 1.0 / 64.0
_CUTE_HARNESS_QKV_SCALE = 1.0 / 64.0
_CUTE_HARNESS_CONTEXT_SCALE = 1.0 / 64.0
_CUTE_HARNESS_MLP_SCALE = 1.0 / 64.0
_CUTE_HARNESS_WEIGHT_H_BOUND = _CUTE_HARNESS_HIDDEN ** -0.5
_CUTE_HARNESS_WEIGHT_MLP_BOUND = _CUTE_HARNESS_MLP ** -0.5
_CUTE_HARNESS_WEIGHT_H_SCALE = _CUTE_HARNESS_WEIGHT_H_BOUND / 448.0
_CUTE_HARNESS_WEIGHT_MLP_SCALE = _CUTE_HARNESS_WEIGHT_MLP_BOUND / 448.0
_CUTE_HARNESS_FP8_DTYPE = _cute_harness_cutlass.Float8E4M3FN


def _cute_harness_fp8_tensor(source, scale):
    storage = _cute_harness_torch.empty(
        source.shape,
        device="cuda",
        dtype=_cute_harness_torch.uint8,
    )
    tensor = _cute_harness_create_fp8(
        storage,
        _CUTE_HARNESS_FP8_DTYPE,
        1,
        source / scale,
    )
    return storage, tensor


def _cute_harness_empty_fp8(shape):
    source = _cute_harness_torch.zeros(
        shape,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    return _cute_harness_fp8_tensor(source, 1.0)


def _cute_harness_scaled_linear(
    activation,
    weight,
    activation_scale,
    weight_scale,
    bias,
):
    scale_a = _cute_harness_torch.tensor(
        activation_scale,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    scale_b = _cute_harness_torch.tensor(
        weight_scale,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    return _cute_harness_torch._scaled_mm(
        activation,
        weight.t(),
        scale_a=scale_a,
        scale_b=scale_b,
        out_dtype=_cute_harness_torch.float32,
    ) + bias


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    hidden_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    ).uniform_(-1.0, 1.0)
    ln1_weight = 1.0 + 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    ln1_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    ln2_weight = 1.0 + 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    ln2_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    qkv_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_QKV,), device="cuda"
    )
    out_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )
    fc_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_MLP,), device="cuda"
    )
    proj_bias = 0.02 * _cute_harness_torch.randn(
        (_CUTE_HARNESS_HIDDEN,), device="cuda"
    )

    qkv_weight_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_QKV, _CUTE_HARNESS_HIDDEN), device="cuda"
    ).uniform_(
        -_CUTE_HARNESS_WEIGHT_H_BOUND,
        _CUTE_HARNESS_WEIGHT_H_BOUND,
    )
    out_weight_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_HIDDEN), device="cuda"
    ).uniform_(
        -_CUTE_HARNESS_WEIGHT_H_BOUND,
        _CUTE_HARNESS_WEIGHT_H_BOUND,
    )
    fc_weight_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_MLP, _CUTE_HARNESS_HIDDEN), device="cuda"
    ).uniform_(
        -_CUTE_HARNESS_WEIGHT_H_BOUND,
        _CUTE_HARNESS_WEIGHT_H_BOUND,
    )
    proj_weight_source = _cute_harness_torch.empty(
        (_CUTE_HARNESS_HIDDEN, _CUTE_HARNESS_MLP), device="cuda"
    ).uniform_(
        -_CUTE_HARNESS_WEIGHT_MLP_BOUND,
        _CUTE_HARNESS_WEIGHT_MLP_BOUND,
    )

    hidden_storage, hidden = _cute_harness_fp8_tensor(
        hidden_source, _CUTE_HARNESS_INPUT_SCALE
    )
    qkv_weight_storage, qkv_weight = _cute_harness_fp8_tensor(
        qkv_weight_source, _CUTE_HARNESS_WEIGHT_H_SCALE
    )
    out_weight_storage, out_weight = _cute_harness_fp8_tensor(
        out_weight_source, _CUTE_HARNESS_WEIGHT_H_SCALE
    )
    fc_weight_storage, fc_weight = _cute_harness_fp8_tensor(
        fc_weight_source, _CUTE_HARNESS_WEIGHT_H_SCALE
    )
    proj_weight_storage, proj_weight = _cute_harness_fp8_tensor(
        proj_weight_source, _CUTE_HARNESS_WEIGHT_MLP_SCALE
    )

    norm1_workspace_storage, norm1_workspace = _cute_harness_empty_fp8(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    )
    qkv_workspace_storage, qkv_workspace = _cute_harness_empty_fp8(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_QKV)
    )
    context_workspace_storage, context_workspace = _cute_harness_empty_fp8(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    )
    norm2_workspace_storage, norm2_workspace = _cute_harness_empty_fp8(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    )
    mlp_workspace_storage, mlp_workspace = _cute_harness_empty_fp8(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_MLP)
    )

    score_workspace_storage = _cute_harness_torch.empty(
        (_CUTE_HARNESS_HEADS * _CUTE_HARNESS_TOKENS, _CUTE_HARNESS_TOKENS),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    probability_workspace_storage = _cute_harness_torch.empty_like(
        score_workspace_storage
    )
    residual_workspace_storage = _cute_harness_torch.empty(
        (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    output_storage = _cute_harness_torch.empty_like(
        residual_workspace_storage
    )
    score_workspace = _cute_harness_from_dlpack(
        score_workspace_storage
    ).mark_layout_dynamic(leading_dim=1)
    probability_workspace = _cute_harness_from_dlpack(
        probability_workspace_storage
    ).mark_layout_dynamic(leading_dim=1)
    residual_workspace = _cute_harness_from_dlpack(
        residual_workspace_storage
    ).mark_layout_dynamic(leading_dim=1)
    output = _cute_harness_from_dlpack(output_storage).mark_layout_dynamic(
        leading_dim=1
    )

    parameter_tensors = (
        _cute_harness_from_dlpack(ln1_weight),
        _cute_harness_from_dlpack(ln1_bias),
        qkv_weight,
        _cute_harness_from_dlpack(qkv_bias),
        out_weight,
        _cute_harness_from_dlpack(out_bias),
        _cute_harness_from_dlpack(ln2_weight),
        _cute_harness_from_dlpack(ln2_bias),
        fc_weight,
        _cute_harness_from_dlpack(fc_bias),
        proj_weight,
        _cute_harness_from_dlpack(proj_bias),
    )
    arguments = (
        hidden,
        *parameter_tensors,
        norm1_workspace,
        qkv_workspace,
        score_workspace,
        probability_workspace,
        context_workspace,
        residual_workspace,
        norm2_workspace,
        mlp_workspace,
        output,
    )
    compiled = _cute_harness_cute.compile(
        gpt2_transformer_block,
        *arguments,
    )
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

    hidden_fp8 = hidden_storage.view(_cute_harness_torch.float8_e4m3fn)
    qkv_weight_fp8 = qkv_weight_storage.view(
        _cute_harness_torch.float8_e4m3fn
    )
    out_weight_fp8 = out_weight_storage.view(
        _cute_harness_torch.float8_e4m3fn
    )
    fc_weight_fp8 = fc_weight_storage.view(
        _cute_harness_torch.float8_e4m3fn
    )
    proj_weight_fp8 = proj_weight_storage.view(
        _cute_harness_torch.float8_e4m3fn
    )
    hidden_dequantized = hidden_fp8.float() * _CUTE_HARNESS_INPUT_SCALE
    norm1_reference = _cute_harness_functional.layer_norm(
        hidden_dequantized,
        (_CUTE_HARNESS_HIDDEN,),
        ln1_weight,
        ln1_bias,
        _CUTE_HARNESS_EPSILON,
    )
    norm1_fp8 = (norm1_reference / _CUTE_HARNESS_NORM_SCALE).to(
        _cute_harness_torch.float8_e4m3fn
    )
    qkv_reference = _cute_harness_scaled_linear(
        norm1_fp8,
        qkv_weight_fp8,
        _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
        qkv_bias,
    )
    qkv_fp8 = (qkv_reference / _CUTE_HARNESS_QKV_SCALE).to(
        _cute_harness_torch.float8_e4m3fn
    )
    qkv_dequantized = qkv_fp8.float() * _CUTE_HARNESS_QKV_SCALE
    query, key, value = qkv_dequantized.reshape(
        _CUTE_HARNESS_TOKENS,
        3,
        _CUTE_HARNESS_HEADS,
        _CUTE_HARNESS_HEAD_DIM,
    ).unbind(dim=1)
    scores = _cute_harness_torch.einsum(
        "thd,shd->hts", query, key
    ) / _cute_harness_math.sqrt(_CUTE_HARNESS_HEAD_DIM)
    causal_mask = _cute_harness_torch.triu(
        _cute_harness_torch.ones(
            (_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_TOKENS),
            device="cuda",
            dtype=_cute_harness_torch.bool,
        ),
        diagonal=1,
    )
    probabilities = scores.masked_fill(causal_mask, -float("inf")).softmax(
        dim=-1
    )
    context_reference = _cute_harness_torch.einsum(
        "hts,shd->thd", probabilities, value
    ).reshape(_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    context_fp8 = (
        context_reference / _CUTE_HARNESS_CONTEXT_SCALE
    ).to(_cute_harness_torch.float8_e4m3fn)
    residual_reference = hidden_dequantized + _cute_harness_scaled_linear(
        context_fp8,
        out_weight_fp8,
        _CUTE_HARNESS_CONTEXT_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
        out_bias,
    )
    norm2_reference = _cute_harness_functional.layer_norm(
        residual_reference,
        (_CUTE_HARNESS_HIDDEN,),
        ln2_weight,
        ln2_bias,
        _CUTE_HARNESS_EPSILON,
    )
    norm2_fp8 = (norm2_reference / _CUTE_HARNESS_NORM_SCALE).to(
        _cute_harness_torch.float8_e4m3fn
    )
    fc_reference = _cute_harness_scaled_linear(
        norm2_fp8,
        fc_weight_fp8,
        _CUTE_HARNESS_NORM_SCALE,
        _CUTE_HARNESS_WEIGHT_H_SCALE,
        fc_bias,
    )
    mlp_fp8 = (
        _cute_harness_functional.gelu(fc_reference, approximate="tanh")
        / _CUTE_HARNESS_MLP_SCALE
    ).to(_cute_harness_torch.float8_e4m3fn)
    quantized_reference = residual_reference + _cute_harness_scaled_linear(
        mlp_fp8,
        proj_weight_fp8,
        _CUTE_HARNESS_MLP_SCALE,
        _CUTE_HARNESS_WEIGHT_MLP_SCALE,
        proj_bias,
    )

    fp32_norm1 = _cute_harness_functional.layer_norm(
        hidden_source,
        (_CUTE_HARNESS_HIDDEN,),
        ln1_weight,
        ln1_bias,
        _CUTE_HARNESS_EPSILON,
    )
    fp32_qkv = fp32_norm1 @ qkv_weight_source.t() + qkv_bias
    fp32_query, fp32_key, fp32_value = fp32_qkv.reshape(
        _CUTE_HARNESS_TOKENS,
        3,
        _CUTE_HARNESS_HEADS,
        _CUTE_HARNESS_HEAD_DIM,
    ).unbind(dim=1)
    fp32_scores = _cute_harness_torch.einsum(
        "thd,shd->hts", fp32_query, fp32_key
    ) / _cute_harness_math.sqrt(_CUTE_HARNESS_HEAD_DIM)
    fp32_probabilities = fp32_scores.masked_fill(
        causal_mask, -float("inf")
    ).softmax(dim=-1)
    fp32_context = _cute_harness_torch.einsum(
        "hts,shd->thd", fp32_probabilities, fp32_value
    ).reshape(_CUTE_HARNESS_TOKENS, _CUTE_HARNESS_HIDDEN)
    fp32_residual = (
        hidden_source + fp32_context @ out_weight_source.t() + out_bias
    )
    fp32_norm2 = _cute_harness_functional.layer_norm(
        fp32_residual,
        (_CUTE_HARNESS_HIDDEN,),
        ln2_weight,
        ln2_bias,
        _CUTE_HARNESS_EPSILON,
    )
    fp32_mlp = _cute_harness_functional.gelu(
        fp32_norm2 @ fc_weight_source.t() + fc_bias,
        approximate="tanh",
    )
    fp32_reference = (
        fp32_residual + fp32_mlp @ proj_weight_source.t() + proj_bias
    )

    quantized_max_abs = (output_storage - quantized_reference).abs().max().item()
    sample_rows = _cute_harness_torch.tensor(
        [0, 1, 31, 63, 64, 95, 127], device="cuda"
    )
    sample_columns = _cute_harness_torch.tensor(
        [0, 63, 64, 255, 511, 767], device="cuda"
    )
    sample_actual = output_storage.index_select(0, sample_rows).index_select(
        1, sample_columns
    )
    sample_reference = fp32_reference.index_select(
        0, sample_rows
    ).index_select(1, sample_columns)
    fp32_sample_max_abs = (sample_actual - sample_reference).abs().max().item()
    if (
        not _cute_harness_torch.isfinite(output_storage).all().item()
        or quantized_max_abs > 0.05
        or fp32_sample_max_abs > 0.5
    ):
        raise RuntimeError(
            f"validation failed: quantized_abs={quantized_max_abs:.6f}, "
            f"fp32_sample_abs={fp32_sample_max_abs:.6f}"
        )

    print(
        "task=model_gpt2_small_transformer_block_fp8 "
        f"quantized_max_abs={quantized_max_abs:.6f} "
        f"fp32_sample_max_abs={fp32_sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
