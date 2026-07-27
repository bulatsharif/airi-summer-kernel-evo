# Task reference: KernelBench Level 2/14

Load `cute-fp8-kernels`, read `references/candidate-gemm-api.md` for the dense
GEMM, and read `references/reductions.md` for a one-warp FP32 row reduction.
The provided scratch tensor deliberately permits a correctness-first two-kernel
decomposition. Divide each GEMM value by `2.0`, sum all 8192 columns, then
multiply the completed row sum by `1.5`.
