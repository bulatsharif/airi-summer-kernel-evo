---
name: cute-fp8-kernels
description: Write, debug, validate, and optimize NVIDIA CuTe DSL Python kernels using dense or block-scaled FP8 on Blackwell GPUs. Use for FP8 GEMM, layouts, TMA/TMEM pipelines, tcgen05 MMA, numerical checks, or remote B300 performance work.
---

# CuTe DSL FP8 kernels

Implement the requested kernel in the task's existing self-contained Python
submission. Do not create a harness/spec/candidate tree unless requested.
PyTorch may allocate inputs and compute the reference; the implementation must
be CuTe DSL Python.

## Establish the contract

Before editing, identify:

- operation, epilogue, and mathematical equation
- logical/physical shapes, layouts, strides, alignment, and tail behavior
- FP8 formats, scale direction/granularity/layout, accumulator, and output type
- required cases, correctness rule, and performance target

Ask only when a missing fact changes the operation. Otherwise choose a
conservative default and report it.

## Load references progressively

- Every FP8 task: [fp8.md](references/fp8.md).
- New/changed kernel: [cute-dsl.md](references/cute-dsl.md),
  [layouts.md](references/layouts.md),
  [memory-pipelines.md](references/memory-pipelines.md), and
  [examples.md](references/examples.md).
- Correctness implementation: [correctness.md](references/correctness.md).
- Remote compatibility/submission: [b300.md](references/b300.md) and
  [submission.md](references/submission.md).
- Failed run: [debugging.md](references/debugging.md).
- Correct kernel being tuned: [performance.md](references/performance.md).

These references are sufficient for design. Inspect examples shipped with the
installed CUTLASS package only to confirm release-specific APIs.

## Workflow

1. Preserve the required entry point and existing contract.
2. Start from the closest local dense/MXFP8 recipe.
3. Keep static configuration explicit; compile once per specialization.
4. Enforce dtype/layout/alignment/tail restrictions in `can_implement`.
5. Build the oracle from actual quantized inputs and exact scale semantics.
6. Submit remotely and make any failure exit nonzero.
7. Optimize only after every required correctness case passes.
8. Warm up, measure repeated kernel-only execution, and verify native FP8 MMA.

Never replace the implementation with PyTorch/Triton/CUDA C++, weaken tests,
hide an error, or run an uncontrolled tuning loop.

## Report

Leave the submission in place and report the operation, assumptions, remote
correctness evidence, timing method/latency, native-path evidence, and remaining
limitations.
