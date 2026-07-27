---
name: cute-fp8-kernels
description: Write, debug, validate, and optimize NVIDIA CuTe DSL Python kernels that use FP8 or block-scaled FP8 on Blackwell GPUs. Use for dense or scaled FP8 GEMM, FP8 data movement and conversion, tcgen05 MMA, TMA/TMEM pipelines, FP8 numerical validation, or performance work on the remote B300 runner.
---

# CuTe DSL FP8 kernels

Implement the requested kernel in the task's existing Python submission. Keep the
submission self-contained apart from installed CUDA, PyTorch, and CUTLASS/CuTe
packages. Do not create a separate harness, specification, or candidate tree
unless the user requests one.

## Establish the contract

Before editing, extract these facts from the prompt and existing code:

- operation and epilogue
- logical and physical shapes and layouts
- FP8 format and scaling convention
- accumulator and output types
- required shapes, strides, alignment, and tail behavior
- correctness criterion and performance objective

Treat the prompt as the specification. Do not invent a new YAML or Markdown
specification file. If a missing fact changes the mathematical operation, ask
for it. Otherwise choose a conservative default and state it in the final
report.

## Read the applicable references

Read:

- [references/fp8.md](references/fp8.md) for every task.
- [references/b300.md](references/b300.md) before selecting an architecture or
  benchmarking.
- [references/examples.md](references/examples.md) before authoring a new GEMM
  or copying an NVIDIA example.
- [references/cute-dsl.md](references/cute-dsl.md) when authoring or debugging
  CuTe layouts, JIT staging, copies, MMA, or pipelines.

Prefer the installed CUTLASS package and version-matched NVIDIA examples over
remembered APIs or examples from another release.

## Implement

1. Inspect the current submission and preserve its required entry point.
2. Start from the closest version-compatible CuTe DSL example when practical.
3. Keep the GPU implementation in CuTe DSL Python. PyTorch operations are
   allowed for input preparation and the reference, not as the implementation.
4. Keep compile-time configuration explicit. Compile once and reuse the
   executor during validation and timing.
5. Preserve required alignment, layout, pipeline, barrier, and cluster
   invariants. Do not make an unsupported tail shape appear supported by
   weakening its test.
6. Make errors fail the process. Do not catch an implementation failure and
   print a successful result.

## Validate correctness

Build the reference from the actual quantized inputs and their specified scale
factors. Compare kernel output to that reference. Report end-to-end quantization
error against the original high-precision inputs separately.

Use deterministic seeds and test every required shape. Include boundary or
irregular shapes only when the operation contract promises them. Use both
absolute and relative tolerances appropriate to the accumulator and output
type. Never loosen tolerances solely because a candidate failed.

Submit the self-contained file through the remote runner described in
`AGENTS.md`. Correctness must pass remotely before optimization.

## Measure performance

Exclude JIT compilation, input generation, allocation, and reference computation
from kernel timing. Warm up the compiled kernel, synchronize correctly, collect
multiple measurements, and report a robust statistic such as the median.

Treat the service's single `device_time_ms` as directional because its PyTorch
Profiler measurement has no controlled warmup. Repeat final candidates modestly.
Do not run an uncontrolled tuning loop.

Before claiming native FP8 acceleration, verify that the implementation selects
the intended FP8 MMA path through CuTe configuration, generated code, or useful
profile evidence. FP8 tensor storage alone is not proof.

## Finish

Leave the requested submission in place and report:

- the implemented operation and chosen assumptions
- remote correctness evidence
- measured latency and measurement method
- remaining limitations or unsupported shapes
