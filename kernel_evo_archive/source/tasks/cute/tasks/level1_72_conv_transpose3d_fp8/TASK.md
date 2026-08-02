# Task: grouped asymmetric ConvTranspose3d, CuTe FP8

Adapt KernelBench Level 1 task 72 to a correctness-first CuTe DSL kernel:

```text
input:          [2, 8, 7, 9, 11]
logical weight: [8, 6, 3, 5, 7]
output:         [2, 12, 14, 18, 33]
stride:         (2, 2, 3)
padding:        (1, 2, 3)
output_padding: (1, 1, 2)
groups:         2
bias:           false
```

The operation follows `torch.nn.functional.conv_transpose3d`. PyTorch stores a
transposed-convolution weight as
`[in_channels, out_channels / groups, kernel_d, kernel_h, kernel_w]`.

For the candidate ABI, the evaluator exposes contiguous flattened views:

```text
input_tensor:  [BATCH, IN_CHANNELS * IN_D * IN_H * IN_W]
weight_tensor: [IN_CHANNELS, WEIGHT_STORAGE_ROW_SIZE]
output_tensor: [BATCH, OUT_CHANNELS * OUT_D * OUT_H * OUT_W]
```

Derive the transposed-convolution indexing from the public parameters above.
Use output-centric inversion: one thread owns one output coordinate and sums
only source coordinates whose inverse stride numerators are divisible. Use
`cutlass.range`, nested `if` predicates, and `.to(cutlass.Float32)`. Do not use
`continue`, `while`, `cute.convert`, `cute.cast`, `float(...)`,
`thread_rank()`, or `num_threads()`.

`WEIGHT_LOGICAL_ROW_SIZE = 6 * 3 * 5 * 7 = 630`. The physical FP8 tensor uses
`WEIGHT_STORAGE_ROW_SIZE = 640` for converter alignment. Only columns `0:630`
hold the logical `[out_channel_in_group, kd, kh, kw]` weight; columns `630:640`
are zero padding and must be ignored.

The tensors remain two-dimensional inside the kernel:

```text
input_tensor[batch, input_flat]
weight_tensor[ic, weight_column]
output_tensor[batch, output_flat]

input_flat =
    ic * IN_D * IN_H * IN_W
    + id * IN_H * IN_W
    + ih * IN_W
    + iw

weight_column =
    oc_in_group * KD * KH * KW
    + kd * KH * KW
    + kh * KW
    + kw
```

For every valid kernel coordinate, reduce over all four
`ic_in_group` values; using only the first channel of the group is incorrect.

## Mathematical contract

The operation is defined by scattering every input/kernel product into its
output coordinate. For input channel `ic`, let
`group = ic // IN_CHANNELS_PER_GROUP`. For every
`oc_in_group`, `kd`, `kh`, and `kw`:

```text
oc = group * OUT_CHANNELS_PER_GROUP + oc_in_group
od = id * STRIDE_D - PAD_D + kd
oh = ih * STRIDE_H - PAD_H + kh
ow = iw * STRIDE_W - PAD_W + kw

output[b, oc, od, oh, ow] +=
    input[b, ic, id, ih, iw]
    * weight[ic, oc_in_group, kd, kh, kw]
```

Only contributions whose `(od, oh, ow)` are inside the declared output shape
are included. Dilation is one. Every output starts from zero. Output padding
only determines the declared output extent; it does not alter the coordinate
equations above.

## Precision contract

- Input and weight storage are FP8 E4M3FN.
- Convert every loaded input and weight element to FP32 before multiplication.
- Accumulate each output in FP32.
- The raw FP8 dot product must be multiplied by
  `INPUT_SCALE * WEIGHT_SCALE` before storing FP32 output.
- `INPUT_SCALE = 1 / 448`.
- `WEIGHT_SCALE = 0.25 / 448`.
- Output padding changes the output shape; it does not insert additional input
  samples.
- Correctness is scored; speed is not.

## Candidate ABI

- Edit only the prepared `submission.py`.
- Keep at least one `@cute.kernel` and one `@cute.jit` function.
- `conv_transpose3d(input_tensor, weight_tensor, output_tensor)` is the
  evaluator entrypoint and must be decorated with `@cute.jit`.
- A launched CuTe GPU kernel must write every output element.
- Do not define/call `main()`, create evaluator inputs, call PyTorch, compute a
  reference, or print a PASS marker.

## Acceptance

- Full max absolute error versus PyTorch applied to the dequantized FP8 input
  and weight: `<= 0.01`.
- Every output value must be finite.
