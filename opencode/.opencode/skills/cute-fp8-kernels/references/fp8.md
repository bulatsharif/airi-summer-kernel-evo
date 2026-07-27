# FP8 kernel rules

## Name the format precisely

Do not use "FP8" as a complete type specification.

- E4M3FN: `torch.float8_e4m3fn` and `cutlass.Float8E4M3FN`
- E5M2: `torch.float8_e5m2` and `cutlass.Float8E5M2`

Confirm the installed spelling before coding. Also specify the accumulator and
output type; FP8 input does not imply FP8 accumulation or output.

## Separate dense FP8 from block-scaled FP8

Dense FP8 MMA consumes FP8 operands directly. An application may still use
tensorwise or rowwise scales around the MMA, but those scales are part of the
operation contract and may be applied before the kernel or in its epilogue.

MXFP8/block-scaled MMA consumes quantized operands plus scale-factor tensors.
The current Blackwell reference example uses E8M0 scale factors with a scale
vector size of 32 for MXFP8. Treat that as a versioned example constraint, not a
universal definition.

For every scaled task, write down:

- whether the supplied value is a scale or inverse scale
- the quantization and dequantization equations
- scale granularity
- logical and physical scale-factor shape
- padding or swizzle rules
- where each scale is applied

## Build the correctness oracle from quantized operands

For a dense FP8 GEMM, use the equivalent of:

```python
a_q = quantize(a_fp32)
b_q = quantize(b_fp32)
reference = a_q.float() @ b_q.float()
reference = apply_epilogue_and_output_conversion(reference)
```

For a scaled GEMM, dequantize using the exact scale convention and scale-factor
layout before the high-precision reference matmul.

Compare the kernel with this reference. Separately report the difference between
the quantized-input reference and `a_fp32 @ b_fp32` if end-to-end quantization
quality matters. The latter is not the kernel correctness error.

Use a combined absolute and relative check. A maximum relative error alone is
unstable near zero. Do not use a blanket "50% is expected for FP8" rule.

## Test deliberately

Use deterministic inputs and include:

- zero and exactly representable values
- positive and negative values
- cancellation-heavy inputs
- values near the intended quantization range
- all required matrix shapes and layouts
- tile-boundary or tail shapes only when promised by the contract

Fail on non-finite output unless the operation explicitly expects it. Do not
catch an unsupported FP8 error and exit successfully.

## Layout and alignment

The CUTLASS 4.6.1 Blackwell dense GEMM example requires at least 16-byte
alignment for contiguous A, B, and C dimensions. That corresponds to a multiple
of 16 FP8 elements. Recheck the installed example before treating this as the
complete contract.

Keep mathematical and physical B layouts distinct. A task may describe
`B[K,N]`, while the CuTe implementation accepts an `N,K,L` tensor plus a layout
that represents the mathematical operand.

## Verify the intended hardware path

Converting a PyTorch tensor to FP8 and immediately converting it back to FP32 is
not an FP8 kernel. Neither is running `torch.matmul` on the dequantized values.

Use CuTe tiled MMA configuration that selects the intended FP8 operation. When
possible, inspect generated PTX/SASS or useful profiler data for the relevant MMA
path. Keep PyTorch matmul confined to the reference calculation.
