# FP8 formats, scaling, and kernel semantics

“FP8 GEMM” is incomplete until the operand formats, scale convention,
accumulator, output type, layouts, and epilogue are specified. Write those
choices down before selecting a CuTe MMA operation.

## Contents

- [Format names](#format-names)
- [Dense versus block-scaled FP8](#dense-versus-block-scaled-fp8)
- [Quantization contract](#quantization-contract)
- [Dense FP8 path](#dense-fp8-path)
- [MXFP8 block-scaled path](#mxfp8-block-scaled-path)
- [Scale tensor layout](#scale-tensor-layout)
- [Accumulator and output](#accumulator-and-output)
- [Correctness semantics](#correctness-semantics)
- [Range, saturation, and non-finite values](#range-saturation-and-non-finite-values)
- [Native hardware path](#native-hardware-path)
- [Common mistakes](#common-mistakes)
- [FP8 contract checklist](#fp8-contract-checklist)

## Format names

Use precise names:

| Role | Torch spelling | CUTLASS spelling |
|---|---|---|
| E4M3 finite FP8 | `torch.float8_e4m3fn` | `cutlass.Float8E4M3FN` |
| E5M2 FP8 | `torch.float8_e5m2` | `cutlass.Float8E5M2` |
| E8M0 unsigned scale | release-dependent | `cutlass.Float8E8M0FNU` |
| FP32 accumulator/output | `torch.float32` | `cutlass.Float32` |
| FP16 accumulator/output | `torch.float16` | `cutlass.Float16` |
| BF16 output | `torch.bfloat16` | `cutlass.BFloat16` |

Confirm spellings against the installed release. E4M3FN and E5M2 differ in
range and precision; they are not interchangeable because both occupy one
byte.

State A and B types independently, even when a baseline implementation requires
them to match.

## Dense versus block-scaled FP8

Dense FP8 MMA consumes FP8 A and B operands directly:

```text
D = epilogue(A_fp8 @ B_fp8)
```

An application may have used tensorwise or rowwise scales while producing those
FP8 tensors, but unless scale tensors are kernel inputs, that scaling is outside
the MMA mainloop or represented explicitly in the epilogue.

Block-scaled FP8 MMA consumes A, B, SFA, and SFB:

```text
D = epilogue((SFA * A_fp8) @ (SFB * B_fp8))
```

The scales vary by blocks along K and may also vary along M or N. Hardware and
CuTe layouts determine how the scale elements are packed and delivered.

Do not choose block-scaled MMA merely because inputs were quantized with a
scale. Choose it when scale-factor tensors and their granularity are part of the
kernel contract.

## Quantization contract

For each operand, define:

```text
q = cast_or_round(clamp(x * q_scale))
x_hat = q * dequant_scale
```

or, if inverse scales are supplied:

```text
q = cast_or_round(clamp(x / scale))
x_hat = q * scale
```

Never infer direction from a variable name alone. Record:

- whether each provided value is a quantization scale, dequantization scale, or
  inverse scale
- rounding mode
- clipping/saturation behavior
- treatment of NaN and infinity
- scale granularity
- padding rules
- whether scaling happens before the kernel, in MMA, or in the epilogue

If the task provides already-quantized tensors, their actual stored FP8 values
are the source of truth for kernel correctness.

## Dense FP8 path

The local CUTLASS 4.6.1 baseline dense Blackwell helper uses an FP8-compatible
tcgen05 operation, commonly constructed through:

```python
sm100_utils.make_trivial_tiled_mma(
    a_dtype,
    a_major_mode,
    b_major_mode,
    acc_dtype,
    use_2cta_instrs,
    mma_tiler_mn,
)
```

Conceptually this chooses a dense FP8 tcgen05 operation such as
`tcgen05.MmaF8F6F4Op`. Use the helper from the installed example rather than
hand-encoding the instruction descriptor.

Baseline constraints to retain until verified:

- A and B use the same dense FP8 type
- contiguous A/B/C modes meet at least 16-byte alignment
- FP32 accumulation is supported
- dense FP8 may support FP16 accumulation in this baseline
- tile and cluster choices meet the rules in `examples.md`

Sixteen-byte alignment corresponds to 16 FP8 elements on a contiguous
dimension. Pointer alignment and stride alignment both matter.

## MXFP8 block-scaled path

The local block-scaled baseline uses:

- A/B: E4M3FN or E5M2
- SFA/SFB: E8M0
- scale vector size: 32 K elements
- accumulator: FP32
- a block-scaled tcgen05 MMA operation, conceptually
  `tcgen05.MmaMXF8F6F4Op`

For mathematical matrices A `[M,K]` and B `[K,N]`, the logical scale shapes are:

```text
SFA: [M, ceil_div(K, 32), L]
SFB: [N, ceil_div(K, 32), L]
```

The physical B tensor is commonly represented as `[N,K,L]`, so SFB follows its
physical non-K mode.

During the mainloop:

1. TMA loads A/B and the required scale-factor tiles.
2. a scale-factor copy moves the scale representation from SMEM to TMEM,
   commonly through `tcgen05.cp`
3. the block-scaled MMA consumes A, B, SFA, and SFB for the corresponding K
   block
4. FP32 accumulation continues across K blocks

Scale stages participate in transaction bytes and pipeline lifetime. Adding
scale TMA copies without adjusting the producer/consumer protocol is incorrect.

## Scale tensor layout

Logical scale coordinates do not define physical storage. The block-scaled
instruction expects a release-specific packed/swizzled layout generated by the
CUTLASS block-scaled utilities.

Track these separately:

```text
logical SFA coordinate: (m, k_block, l)
logical SFB coordinate: (n, k_block, l)
physical GMEM layout:   packed/swizzled descriptor layout
SMEM layout:            staged copy layout
TMEM layout:            MMA-consumable scale layout
```

Do not allocate a plain row-major two-dimensional tensor and assume it is
instruction-compatible because it contains the right number of scales.

For a reference calculation, decode physical scales back to the logical
`(row_or_column, k_block, batch)` coordinate. Test scale-indexing with one-hot K
blocks or distinct powers-of-two scales.

When K is not a multiple of the scale vector size, define whether:

- K tails are unsupported
- A/B are padded
- the final scale block covers padded zeros
- a predicated path exists

Do not invent tail support by taking `ceil_div` in the reference alone.

## Accumulator and output

FP8 input does not imply FP8 accumulation or output.

Specify:

- accumulator type used by MMA
- any alpha/beta operation
- bias or activation
- intermediate conversion
- output type
- output saturation/rounding

FP32 accumulation is the safe default for correctness and the block-scaled
baseline. Dense FP16 accumulation changes the numerical contract and generally
needs a looser but still justified tolerance.

If output is FP8:

1. compare a higher-precision view before output quantization when possible
2. model the exact output quantization in the final reference
3. distinguish accumulator error from output-quantization error
4. test saturation and non-finite behavior explicitly

If the epilogue includes `alpha * accumulator + beta * C`, make sure the initial
C tensor and its type are part of the reference.

## Correctness semantics

Dense reference:

```python
a_used = a_fp8.float()
b_used = b_fp8.float()
reference_acc = a_used @ b_used
reference = apply_epilogue_and_output_cast(reference_acc)
```

Block-scaled reference:

```python
a_used = dequantize_a_by_logical_scale_block(a_fp8, sfa)
b_used = dequantize_b_by_logical_scale_block(b_fp8, sfb)
reference_acc = a_used @ b_used
reference = apply_epilogue_and_output_cast(reference_acc)
```

If the mathematical B is supplied physically as `[N,K,L]`, transpose or index
it according to that declared layout before the reference matmul.

Report two different quantities when relevant:

1. **kernel error**: kernel versus reference built from actual quantized inputs
2. **quantization error**: quantized-input reference versus original
   high-precision operation

Only the first decides whether the kernel implemented its contract.

Use `correctness.md` for tolerance policy, structured cases, memory-safety
checks, and acceptance gates.

## Range, saturation, and non-finite values

Derive input ranges from the actual datatype implementation:

```python
info = torch.finfo(dtype)
print(info.min, info.max, info.eps)
```

Generate most random tests comfortably inside the intended scaled range. Add
separate boundary tests for:

- positive and negative maximum intended magnitude
- values that round to zero
- values around representable transitions
- clipping/saturation
- NaN/infinity policy

Do not let random input generation accidentally saturate most elements; it
reduces the test's ability to reveal layout and accumulation errors.

Fail on unexpected non-finite kernel output. If non-finite behavior is part of
the task, compare it with an explicitly defined policy rather than ordinary
`allclose`.

## Native hardware path

A correct FP32 matmul over dequantized FP8 tensors is a reference, not the
implementation.

The implementation must:

- keep the GPU kernel in CuTe DSL Python
- construct a dense or block-scaled FP8-capable tiled MMA
- move operands through layouts compatible with that MMA
- issue the tcgen05 MMA path on the target
- accumulate and store through the intended CuTe kernel

Evidence can come from construction, generated PTX/SASS, or profiler data.
Follow `performance.md` before making a native-path performance claim.

## Common mistakes

| Mistake | Consequence |
|---|---|
| Calling every one-byte float “FP8” | Wrong format/range/operation |
| Comparing with original FP32 inputs only | Quantization error misclassified as kernel error |
| Treating inverse scale as scale | Magnitudes wrong by squared scale ratios |
| Using dense MMA with SFA/SFB ignored | Wrong operation |
| Using block-scaled MMA for tensorwise pre-scaling | Unnecessary/wrong scale layout |
| Treating SFA/SFB as row-major matrices | Wrong scale block selected |
| Dequantizing before the submitted matmul | No native FP8 implementation |
| Assuming FP8 input means FP8 output | Wrong epilogue/reference |
| Applying blanket loose tolerances | Layout and accumulation bugs hidden |
| Letting unsupported tails pass weak tests | Out-of-bounds or partial output |

## FP8 contract checklist

- [ ] A and B formats are named precisely.
- [ ] Dense versus block-scaled path is explicit.
- [ ] Quantization and dequantization equations are written.
- [ ] Every scale's direction and granularity are stated.
- [ ] Logical and physical scale layouts are distinguished.
- [ ] Accumulator, epilogue, and output types are stated.
- [ ] Alignment and tail-shape restrictions are enforced.
- [ ] Reference uses actual quantized operands.
- [ ] Kernel error and quantization error are reported separately.
- [ ] Non-finite and saturation behavior is tested.
- [ ] CuTe selects the intended native FP8 MMA path.
