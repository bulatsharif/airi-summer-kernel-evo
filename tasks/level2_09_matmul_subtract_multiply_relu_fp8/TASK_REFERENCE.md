# Task reference: KernelBench Level 2/9

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-gemm-api.md`. Reuse its version-pinned dense FP8 GEMM
flow; this task owns only the operation-specific epilogue.

The exact mathematical order is `Y = relu((X @ W.T + bias - 2.0) * 1.5)`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.

Use the common reference's exact `(128, 256, 128)` TMEM epilogue to write the
correctly dequantized FP32 GEMM matrix first. Then launch the starter's simple
elementwise kernel over the two-dimensional output to add
`bias[column]`, subtract, multiply, and clamp. In-place global read/modify/write
is valid after the GEMM launch completes; a temporary tensor is not required.
Do not rewrite the working GEMM pipeline merely because the elementwise result
is numerically wrong.
