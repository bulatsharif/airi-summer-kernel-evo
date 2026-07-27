# Task reference: KernelBench Level 2/63

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-gemm-api.md`. Reuse its version-pinned dense FP8 GEMM
flow; this task owns only the operation-specific epilogue.

The exact mathematical order is `Y = relu(X @ W.T + bias) / 2.0`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.
