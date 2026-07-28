# Task reference: grouped ConvTranspose3d FP8

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`. For this
correctness-first task, use its scalar FP8 indexing guidance; a tensor declared
as `[rows, columns]` is indexed with two coordinates.

For a fixed output coordinate, invert the public scatter equation only when
each numerator is divisible by its stride. Keep `oc_in_group = oc %
OUT_CHANNELS_PER_GROUP` fixed and reduce over the four `ic_in_group` values.
The physical weight row has 640 columns, but only the first 630 are logical.

The task intentionally provides no convolution implementation. The reference
only disambiguates the public ABI and channel roles.

Keep compilation bounded: use dynamic `cutlass.range` loops for the
`4 * 3 * 5 * 7` reduction rather than fully unrolling the entire convolution
with `cutlass.range_constexpr`. Do not use `continue` inside CuTe loops. Express
divisibility and bounds as nested `if` predicates, accumulate only when all
predicates hold, and issue one final output store per thread.

## Exact two-dimensional storage indexing

The CuTe tensors are already two-dimensional views. Never fold the batch or
weight row into the column and then access row zero:

```python
input_flat = (
    ic * IN_D * IN_H * IN_W
    + id_value * IN_H * IN_W
    + ih_value * IN_W
    + iw_value
)
kernel_flat = kd * KH * KW + kh * KW + kw
weight_column = oc_in_group * KD * KH * KW + kernel_flat

input_value = input_tensor[batch, input_flat].to(cutlass.Float32)
weight_value = weight_tensor[ic, weight_column].to(cutlass.Float32)
output_tensor[batch, output_flat] = accumulator * INPUT_SCALE * WEIGHT_SCALE
```

For every valid `(kd, kh, kw)`, sum all four channels:

```python
for ic_in_group in cutlass.range(IN_CHANNELS_PER_GROUP):
    ic = group * IN_CHANNELS_PER_GROUP + ic_in_group
    # load input_tensor[batch, input_flat]
    # load weight_tensor[ic, weight_column]
```

Do not use only `group * IN_CHANNELS_PER_GROUP`; that drops three quarters of
the input channels. Do not multiply `oc_in_group` by
`WEIGHT_LOGICAL_ROW_SIZE`; each output-channel slice has only
`KD * KH * KW` entries.
