# FP8 semantics

“FP8 GEMM” is incomplete without formats, scale convention, accumulator,
output, layouts, and epilogue.

## Contents

[Formats](#formats) ·
[Dense versus block-scaled](#dense-versus-block-scaled) ·
[Quantization contract](#quantization-contract) ·
[Dense Blackwell path](#dense-blackwell-path) ·
[Accumulator, epilogue, output](#accumulator-epilogue-output) ·
[Gates](#gates)

## Formats

| Role | Torch | CUTLASS |
|---|---|---|
| E4M3 finite | `torch.float8_e4m3fn` | `cutlass.Float8E4M3FN` |
| E5M2 | `torch.float8_e5m2` | `cutlass.Float8E5M2` |
| E8M0 scale | release-dependent | `cutlass.Float8E8M0FNU` |

Confirm installed spellings. E4M3 and E5M2 differ in precision/range and are not
interchangeable.

## Dense versus block-scaled

Dense MMA consumes FP8 A/B directly:

```text
D = epilogue(A_fp8 @ B_fp8)
```

External tensor/row scales may have produced A/B, but are not MMA inputs unless
the operation explicitly accepts them.

MXFP8 MMA consumes A, B, SFA, and SFB:

```text
D = epilogue((SFA*A_fp8) @ (SFB*B_fp8))
```

Use it only when block scale tensors are in the kernel contract. The local
CUTLASS 4.6.1 baseline uses E4M3 or E5M2 operands, E8M0 scales, 32 K-elements per
scale vector, and FP32 accumulation. Logical scales for A `[M,K,L]` and physical
B `[N,K,L]` are:

```text
SFA [M, ceil_div(K,32), L]
SFB [N, ceil_div(K,32), L]
```

Their physical GMEM/SMEM/TMEM layouts are packed/swizzled by block-scaled
utilities; they are not ordinary row-major arrays. Scale loads contribute to
TMA transaction bytes and pipeline state. SMEM scales are copied into
MMA-compatible TMEM, commonly with `tcgen05.cp`.

## Quantization contract

Write the exact equation, for example:

```text
q = cast(clamp(x / scale)); x_hat = q * scale
```

or:

```text
q = cast(clamp(x * inv_scale)); x_hat = q / inv_scale
```

State for every operand:

- scale versus inverse scale
- rounding/clipping and non-finite policy
- tensor/row/block granularity
- logical and physical scale layout/padding
- where dequantization occurs

Never infer direction from a variable name. Define K tails: unsupported,
padded, or predicated.

## Dense Blackwell path

The version-pinned dense helper constructs an FP8-capable tcgen05 tiled MMA,
conceptually `MmaF8F6F4Op`; MXFP8 uses a block-scaled operation conceptually
`MmaMXF8F6F4Op`. Use the installed helper instead of encoding descriptors.

Baseline dense constraints:

- A/B have the same FP8 type
- contiguous A/B/C modes meet 16-byte alignment (16 FP8 elements)
- FP32 accumulation; FP16 may be supported for dense FP8
- tile/cluster constraints from `examples.md`

Pointer and stride alignment both matter.

## Accumulator, epilogue, output

Specify accumulator type, alpha/beta, bias/activation, intermediate conversion,
output type, and saturation/rounding. FP8 input does not imply FP8 accumulation
or output. For FP8 output, distinguish accumulator error from output
quantization error and test saturation.

Dense oracle:

```python
reference_acc = a_fp8.float() @ b_fp8.float()
reference = apply_exact_epilogue_and_cast(reference_acc)
```

Block-scaled oracle first maps physical scales to logical K blocks and
dequantizes actual FP8 values, then performs the high-precision matmul.

Report separately:

1. kernel versus the quantized-input oracle
2. quantized-input oracle versus original high-precision operation

Only the first is kernel correctness.

## Gates

- Generate ordinary random tests inside useful representable range; use
  `torch.finfo(dtype)` rather than remembered limits.
- Add explicit zero, sign, cancellation, transition, saturation, and
  non-finite cases.
- Fail on unexpected non-finite output.
- Never dequantize before the submitted matmul: that is a reference, not a
  native FP8 implementation.
- Verify CuTe selects the intended tcgen05 FP8 path.
