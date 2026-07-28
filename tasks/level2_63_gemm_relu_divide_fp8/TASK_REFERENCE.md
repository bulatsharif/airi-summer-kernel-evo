# Task reference: KernelBench Level 2/63

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-gemm-api.md`. Reuse its version-pinned dense FP8 GEMM
flow; this task owns only the operation-specific epilogue.

The exact mathematical order is `Y = relu(X @ W.T + bias) / 2.0`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.

Use the common reference's exact `(128, 256, 128)` TMEM epilogue to write the
correctly dequantized FP32 GEMM matrix first. Keep ReLU/divide in the starter's
separate elementwise kernel with `output[row, column]` and `bias[column]`.
Do not invent `get_slice_in_stage`, call `partition_D` on an MMA object, or
replace the documented destination partition with a scalar tensor slice.
