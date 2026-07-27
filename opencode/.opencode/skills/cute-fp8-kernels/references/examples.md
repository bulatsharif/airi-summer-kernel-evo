# Versioned NVIDIA examples

Use the example closest to the requested operation. Read its constraints,
command-line runner, reference check, and kernel class before extracting code.
Retain NVIDIA's BSD-3-Clause notice when copying source.

The links below are pinned to CUTLASS `v4.6.1`, commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`. If the remote environment uses a
different version, prefer that version's files.

## Dense Blackwell GEMM

<https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py>

Use as the first reference for a non-persistent dense FP8 GEMM. Its documented
inputs include E4M3FN and E5M2, and it demonstrates TMA, `tcgen05.mma`, tensor
memory accumulation, and the epilogue path.

<https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py>

Use only when persistent scheduling is required or the simpler version is
already correct. Persistent scheduling introduces additional coordination and
is a poor first debugging surface.

## Block-scaled Blackwell GEMM

<https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/dense_blockscaled_gemm_persistent.py>

Use for SM100-style MXFP8 or other supported block-scaled formats. Read the
scale-factor layout utilities and reference generation; do not replace them
with a flat scale array.

<https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/sm103_dense_blockscaled_gemm_persistent.py>

This is a specialized SM103 FP4 Ultra example. Use it to understand SM103
mechanics, not as proof that the same type combinations are available for an
FP8 task.

## General DSL documentation

- Overview: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/overview.html>
- CuTe DSL guide: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>
- API reference: <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html>

Do not mix CuTe C++ examples with CuTe DSL Python unless the task is explicitly
about translating an algorithm and every API is re-established from Python DSL
documentation.
