# Correctness

Correctness means the kernel implements the declared operation over the actual
quantized inputs, scales, layouts, accumulator, epilogue, and output conversion.

## Contents

[Contract](#contract) · [Oracle](#oracle) · [Comparison](#comparison) ·
[Required cases](#required-cases) ·
[Memory and execution safety](#memory-and-execution-safety) ·
[Validation order](#validation-order)

## Contract

Before coding, record:

```text
equation:
A/B/C logical and physical shapes:
dtypes and layouts:
scale direction/granularity/layout:
accumulator and epilogue:
supported alignment/tails:
required cases:
```

Reject unsupported cases rather than weakening tests.

## Oracle

Dense:

```python
a_used = a_fp8.float()
b_used = b_fp8.float()  # transpose/index if stored physically as N,K,L
acc_ref = a_used @ b_used
reference = exact_epilogue_and_output_cast(acc_ref)
```

Block-scaled:

1. decode physical SFA/SFB to logical row/column and K-block coordinates
2. apply the exact scale/inverse-scale direction to actual FP8 A/B values
3. perform FP32 matmul
4. reproduce alpha/beta, bias/activation, rounding/saturation, and output cast

Do not use original FP32 inputs as the kernel oracle. Report their difference
from the quantized-input oracle separately as quantization error.

## Comparison

Use combined absolute/relative tolerance:

```python
torch.testing.assert_close(output, reference, atol=atol, rtol=rtol)
```

Also report maximum absolute error and a guarded relative metric:

```text
abs_err = abs(out-ref)
rel_err = abs_err / max(abs(ref), floor)
```

Choose tolerances from accumulator/output rounding and K reduction depth. FP32
accumulation with FP16/BF16 output is usually governed by output rounding;
FP16 accumulation needs more allowance. FP8 output includes explicit output
quantization. Measure an initial known-correct path, document tolerance, and
never loosen it only because a candidate failed. Relative error alone is
unstable near zero.

## Required cases

Use deterministic seeds and structured inputs:

- zeros, ones, signs, exactly representable values
- identity/one-hot patterns to expose transposes and K-block selection
- row/column ramps to expose coordinate errors
- alternating signs/cancellation
- values near intended range and output conversion boundaries
- every required shape, layout, batch, and scale pattern
- multiple K tiles and multiple output tiles
- tails only when promised

Vary scales by block (distinct powers of two are useful) so wrong scale indexing
cannot pass accidentally.

## Memory and execution safety

- initialize output with a sentinel and verify every promised element changes
- when practical, pad storage with guards and confirm guards are untouched
- test alignment rejection and all supported layout modes
- synchronize immediately after launch during diagnosis
- fail on unexpected NaN/Inf
- repeat a case to expose stale pipeline state or nondeterminism

A candidate passes only if compilation/launch succeed, every required case
meets the fixed comparison rule, no memory/non-finite failure occurs, and the
process exits zero. Print `passed=true/false`, shape/types, `max_abs`,
`max_rel`, and the tolerance used.

## Validation order

1. one exact tile and one K tile
2. multiple K tiles
3. multiple M/N tiles
4. scale variation for MXFP8
5. required layouts/batches
6. supported tails
7. full required sizes

Only then benchmark.
