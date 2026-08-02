# FP8, FP4, scaling, and numerical contracts

Low-precision storage is not a complete numerical specification. A kernel also
needs the format, scale direction, scale granularity, accumulation type,
intermediate conversions, output type, and exceptional-value behavior.

## Floating-point fields

A binary floating-point format encodes:

```text
sign | exponent | significand/fraction
```

More exponent bits increase dynamic range. More fraction bits improve local
precision. Fewer total bits increase rounding and saturation pressure.

## FP8 formats

| Format | Exponent bits | Fraction bits | Main tradeoff |
| --- | ---: | ---: | --- |
| E4M3 | 4 | 3 | greater precision, smaller range |
| E5M2 | 5 | 2 | greater range, lower precision |

Common CuTe type names:

```text
cutlass.Float8E4M3FN
cutlass.Float8E5M2
```

The `FN` suffix denotes a finite-value-oriented E4M3 encoding. E4M3 and E5M2
are different byte interpretations and cannot be exchanged by changing only a
type annotation.

NVIDIA E4M3FN has maximum finite magnitude 448. E5M2 provides a much larger
range at lower precision. Always follow the dtype in the task/runtime tensor
rather than choosing by habit.

## Quantization and dequantization

One common convention:

```text
q = round_and_clamp(real / scale)
real_approx = q * scale
```

An inverse-scale convention:

```text
q = round_and_clamp(real * inverse_scale)
real_approx = q / inverse_scale
```

The variable name alone does not establish direction. Write the actual
equation before implementing arithmetic.

For independently dequantized operands:

```text
A_real ~= A_low * scale_A
B_real ~= B_low * scale_B
A_real @ B_real ~= (A_low @ B_low) * scale_A * scale_B
```

This identity does not determine where scaling is applied. An instruction,
accumulator path, or epilogue may incorporate it. Apply each logical scale
exactly once.

## Scale granularity

Common scale ownership:

- one value per tensor;
- one value per row, column, or channel;
- one value per fixed-size block;
- hierarchical block and tensor scales.

Scale layout is part of the tensor contract. Each data coordinate must select
the scale owned by the same logical group. A wrong mode can produce smooth,
plausible output while systematically scaling the wrong rows or blocks.

## Dense FP8

Dense FP8 tensor-core MMA consumes FP8 operands directly. External scaling may
have been used to create those operands, but scale tensors are not instruction
operands unless the operation explicitly includes them.

Storage dtype, multiply behavior, accumulator dtype, and output dtype remain
separate:

```text
FP8 storage -> narrow multiply -> usually wider accumulation -> output conversion
```

Do not dequantize both full operands to FP32 inside a candidate and call the
result an FP8 tensor-core implementation.

## Block-scaled FP8 and MX formats

Block-scaled instructions consume operand blocks and corresponding scale
factors. A common MXFP8 family uses:

- E4M3 or E5M2 operand values;
- E8M0-like power-of-two scale representation;
- scale ownership over a fixed K-axis block;
- FP32 accumulation.

Logical scale coordinates and physical scale storage are distinct. Instruction
utilities may pack or swizzle scale layouts for GMEM, SMEM, and TMEM.

Adding an arbitrary scale tensor to dense FP8 does not create an MXFP8
contract. Use a block-scaled instruction only when the declared scale type,
granularity, and physical layout match it.

## FP4 and packed storage

E2M1 FP4 uses one sign bit, two exponent bits, and one fraction bit. Its finite
magnitudes are:

```text
0, 0.5, 1, 1.5, 2, 3, 4, 6
```

Two logical FP4 values occupy one byte. A packed-storage contract must specify:

- which nibble holds the first logical element;
- bit interpretation and sign;
- logical element count versus byte count;
- scale ownership;
- padding for an odd logical count.

Scalar dereference of a sub-byte typed tensor is not generally supported in the
4.6.1 DSL path. Elementwise packed work may require typed byte loads and
explicit nibble extraction. Vectorized copy or tensor-core paths can retain a
packed narrow type when their operation supports it.

Plain packed E2M1 is not automatically NVFP4. NVFP4 additionally defines scale
types, block granularity, layouts, and a compatible block-scaled instruction.

## Accumulation

Reduction error depends on:

- accumulator precision;
- reduction length;
- summation order;
- cancellation;
- fast-accumulation modes;
- frequency of conversion or rescaling.

FP32 accumulation often reduces error, but the task contract is authoritative.
Fast accumulation can change both speed and numerical behavior and must be
validated as an optimization.

Initialize the accumulator exactly once per logical output. Distinguish the
first reduction contribution from later accumulation.

## Conversion and nonlinear arithmetic

Apply reductions and nonlinear operations in the declared compute type. Convert
to the output type at the declared boundary.

Public scalar type constructors perform conversion:

```python
converted = cutlass.Float32(value)
```

Simple selection can use generated comparisons:

```python
result = positive_value if predicate else negative_value
```

Whether Python conditional syntax lowers to elementwise selection depends on
the value type and preprocessed region. Preserve type compatibility across
both branches.

## Rounding, saturation, and non-finite values

Specify:

- rounding mode where observable;
- saturation or overflow behavior;
- treatment of NaN and infinity;
- output conversion point.

Do not silently clamp an intermediate unless the mathematical contract says
so. Do not let an output cast hide overflow produced earlier.

## Four error budgets

Separate:

1. input quantization error;
2. kernel arithmetic/accumulation error;
3. scale placement/indexing error;
4. output conversion error.

When the evaluator supplies already-quantized inputs, kernel correctness is
normally measured against a reference derived from those stored values and
their scales. Comparing only with original high-precision pre-quantized values
mixes two different error sources.

## Numerical invariants

General invariants:

- zeros obey the operation's zero behavior;
- signs and monotonicity are preserved where mathematically required;
- changing one scale affects only its owned group;
- scaling does not depend on tile or stage count;
- every output is initialized and written;
- unexpected NaN or infinity fails validation;
- repeated execution is deterministic within the operation's guarantees.

These invariants diagnose numerical contracts without supplying a task
algorithm.
