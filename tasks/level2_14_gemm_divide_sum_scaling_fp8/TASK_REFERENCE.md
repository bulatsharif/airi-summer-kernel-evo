# Task reference: KernelBench Level 2/14

Load `cute-fp8-kernels`, read
`references/candidate-dense-gemm-template.py` for the compile-verified dense
GEMM, and read `references/reductions.md` for a one-warp FP32 row reduction.
Preserve the template's TMA/pipeline/TMEM core. The provided scratch tensor
deliberately permits a correctness-first two-kernel decomposition. Divide each
GEMM value by `2.0`, sum all 8192 columns, then multiply the completed row sum
by `1.5`.
