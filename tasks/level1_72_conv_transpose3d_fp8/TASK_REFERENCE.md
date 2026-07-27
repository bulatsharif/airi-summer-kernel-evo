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
