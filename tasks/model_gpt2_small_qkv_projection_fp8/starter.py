import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import tcgen05


M = 8192
N = 2304
K = 768
FP8_MAX = 448.0
WEIGHT_BOUND = K ** -0.5
SCALE_X = 1.0 / FP8_MAX
SCALE_W = WEIGHT_BOUND / FP8_MAX
FP8_DTYPE = cutlass.Float8E4M3FN


@cute.kernel
def qkv_gemm_kernel(
    hidden_states: cute.Tensor,
    packed_qkv_weight: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: FP8 tcgen05 GEMM with FP32 accumulation and output.
    pass


@cute.kernel
def add_qkv_bias_kernel(output: cute.Tensor, bias_qkv: cute.Tensor):
    # TODO: add FP32 bias by output column.
    pass


@cute.jit
def gpt2_qkv_projection(
    hidden_states: cute.Tensor,
    packed_qkv_weight: cute.Tensor,
    bias_qkv: cute.Tensor,
    output: cute.Tensor,
):
    # TODO: construct and launch GEMM, then apply bias.
    pass


# === CUTE_HARNESS_EVALUATOR_V1 ===
import torch as _cute_harness_torch

import cutlass as _cute_harness_cutlass
import cutlass.cute as _cute_harness_cute
from cutlass.cute.runtime import from_dlpack as _cute_harness_from_dlpack
from cutlass.utils import create_cute_tensor_for_fp8 as _cute_harness_create_fp8


_CUTE_HARNESS_M = 8192
_CUTE_HARNESS_N = 2304
_CUTE_HARNESS_K = 768
_CUTE_HARNESS_FP8_MAX = 448.0
_CUTE_HARNESS_WEIGHT_BOUND = _CUTE_HARNESS_K ** -0.5
_CUTE_HARNESS_SCALE_X = 1.0 / _CUTE_HARNESS_FP8_MAX
_CUTE_HARNESS_SCALE_W = _CUTE_HARNESS_WEIGHT_BOUND / _CUTE_HARNESS_FP8_MAX
_CUTE_HARNESS_FP8_DTYPE = _cute_harness_cutlass.Float8E4M3FN


def main():
    if not _cute_harness_torch.cuda.is_available():
        raise RuntimeError("A Blackwell CUDA device is required")

    _cute_harness_torch.manual_seed(_CUTE_HARNESS_SEED)
    source_x = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_K),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    ).uniform_(-1.0, 1.0)
    source_weight = _cute_harness_torch.empty(
        (_CUTE_HARNESS_N, _CUTE_HARNESS_K),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    ).uniform_(-_CUTE_HARNESS_WEIGHT_BOUND, _CUTE_HARNESS_WEIGHT_BOUND)
    bias_qkv = _cute_harness_torch.randn(
        (_CUTE_HARNESS_N,),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    ) * 0.02
    storage_x = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_K),
        device="cuda",
        dtype=_cute_harness_torch.uint8,
    )
    storage_weight = _cute_harness_torch.empty(
        (_CUTE_HARNESS_N, _CUTE_HARNESS_K),
        device="cuda",
        dtype=_cute_harness_torch.uint8,
    )
    output = _cute_harness_torch.empty(
        (_CUTE_HARNESS_M, _CUTE_HARNESS_N),
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )

    hidden_states = _cute_harness_create_fp8(
        storage_x,
        _CUTE_HARNESS_FP8_DTYPE,
        1,
        source_x * _CUTE_HARNESS_FP8_MAX,
    )
    packed_qkv_weight = _cute_harness_create_fp8(
        storage_weight,
        _CUTE_HARNESS_FP8_DTYPE,
        1,
        source_weight
        * (_CUTE_HARNESS_FP8_MAX / _CUTE_HARNESS_WEIGHT_BOUND),
    )
    bias_tensor = _cute_harness_from_dlpack(bias_qkv)
    output_tensor = _cute_harness_from_dlpack(output).mark_layout_dynamic(
        leading_dim=1
    )

    compiled = _cute_harness_cute.compile(
        gpt2_qkv_projection,
        hidden_states,
        packed_qkv_weight,
        bias_tensor,
        output_tensor,
    )
    for _ in range(_CUTE_HARNESS_WARMUP):
        compiled(hidden_states, packed_qkv_weight, bias_tensor, output_tensor)
    _cute_harness_torch.cuda.synchronize()

    timings_ms = []
    for _ in range(_CUTE_HARNESS_REPEATS):
        start = _cute_harness_torch.cuda.Event(enable_timing=True)
        end = _cute_harness_torch.cuda.Event(enable_timing=True)
        start.record()
        compiled(hidden_states, packed_qkv_weight, bias_tensor, output_tensor)
        end.record()
        end.synchronize()
        timings_ms.append(start.elapsed_time(end))
    kernel_time_ms = sorted(timings_ms)[len(timings_ms) // 2]

    x_fp8 = storage_x.view(_cute_harness_torch.float8_e4m3fn)
    weight_fp8 = storage_weight.view(_cute_harness_torch.float8_e4m3fn)
    scale_x = _cute_harness_torch.tensor(
        _CUTE_HARNESS_SCALE_X,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    scale_w = _cute_harness_torch.tensor(
        _CUTE_HARNESS_SCALE_W,
        device="cuda",
        dtype=_cute_harness_torch.float32,
    )
    reference = _cute_harness_torch._scaled_mm(
        x_fp8,
        weight_fp8.t(),
        scale_a=scale_x,
        scale_b=scale_w,
        out_dtype=_cute_harness_torch.float32,
    ) + bias_qkv
    full_max_abs = (output - reference).abs().max().item()

    rows = _cute_harness_torch.tensor(
        [0, 1, 31, 127, 1023, 4095, 4096, 8191],
        device="cuda",
    )
    columns = _cute_harness_torch.tensor(
        [0, 63, 767, 768, 831, 1535, 1536, 1599, 2303],
        device="cuda",
    )
    fp32_reference = (
        source_x.index_select(0, rows)
        @ source_weight.index_select(0, columns).t()
        + bias_qkv.index_select(0, columns)
    )
    actual = output.index_select(0, rows).index_select(1, columns)
    sample_max_abs = (actual - fp32_reference).abs().max().item()

    if (
        not _cute_harness_torch.isfinite(output).all().item()
        or full_max_abs > 0.01
        or sample_max_abs > 0.2
    ):
        raise RuntimeError(
            f"validation failed: full_abs={full_max_abs:.6f}, "
            f"sample_abs={sample_max_abs:.6f}"
        )

    print(
        "task=model_gpt2_small_qkv_projection_fp8 "
        f"full_max_abs={full_max_abs:.6f} "
        f"sample_max_abs={sample_max_abs:.6f} "
        f"kernel_time_ms={kernel_time_ms:.6f} PASS"
    )
    _cute_harness_torch.cuda.synchronize()


main()
