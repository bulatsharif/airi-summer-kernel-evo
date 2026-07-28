# Task reference: KernelBench Level 2/12

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-gemm-api.md`. Reuse its version-pinned dense FP8 GEMM
flow; this task owns only the operation-specific epilogue.

The exact mathematical order is `Y = leaky_relu((X @ W.T + bias) * 2.0, 0.1)`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.

Use the common reference's exact `(128, 256, 128)` TMEM epilogue to write the
correctly dequantized FP32 GEMM matrix first. Then launch the starter's
elementwise kernel and index output as `output[row, column]` and bias as
`bias[column]`. In-place global read/modify/write is valid after the GEMM
launch; do not allocate an unavailable temporary tensor or fuse the activation
while basic correctness is still failing.
