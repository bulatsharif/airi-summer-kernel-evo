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

For every task, read:

- [references/b300.md](references/b300.md) for the target and compatibility
  contract.
- [references/fp8.md](references/fp8.md) for datatype and scaling semantics.
- [references/correctness.md](references/correctness.md) for the oracle and
  acceptance gates.
- [references/submission.md](references/submission.md) for the required
  single-file structure and remote protocol.

Before creating or structurally changing a kernel, also read:

- [references/cute-dsl.md](references/cute-dsl.md) for language/JIT semantics
  and current limitations.
- [references/layouts.md](references/layouts.md) for tensor and layout algebra.
- [references/memory-pipelines.md](references/memory-pipelines.md) for TMA,
  shared memory, TMEM, barriers, and warp roles.
- [references/examples.md](references/examples.md) for dense FP8 and MXFP8
  implementation recipes.

When a run fails, read [references/debugging.md](references/debugging.md).
After correctness passes, read
[references/performance.md](references/performance.md) before tuning or making
performance claims.

Use these local references for the initial implementation. Prefer examples
shipped with the installed CUTLASS package when exact API detail is still
needed. Do not require web access for the normal workflow.

## Implement

1. Inspect the current submission and preserve its required entry point.
2. Write down the operation contract from `correctness.md`; keep it in the
   submission or working notes rather than creating a new specification tree.
3. Follow the local dense or block-scaled recipe. If the installed package
   contains a matching example, use it to confirm exact API signatures.
4. Keep the GPU implementation in CuTe DSL Python. PyTorch operations are
   allowed for input preparation and the reference, not as the implementation.
5. Keep compile-time configuration explicit. Compile once and reuse the
   executor during validation and timing.
6. Preserve required alignment, layout, pipeline, barrier, and cluster
   invariants. Do not make an unsupported tail shape appear supported by
   weakening its test.
7. Make errors fail the process. Do not catch an implementation failure and
   print a successful result.

## Validate correctness

Follow every applicable gate in `correctness.md`. Submit the self-contained file
through the remote runner described in `AGENTS.md`. Correctness must pass
remotely before optimization.

## Measure performance

Follow the measurement protocol and tuning order in `performance.md`. Do not
optimize a failing candidate or run an uncontrolled search.

## Finish

Leave the requested submission in place and report:

- the implemented operation and chosen assumptions
- remote correctness evidence
- measured latency and measurement method
- remaining limitations or unsupported shapes
