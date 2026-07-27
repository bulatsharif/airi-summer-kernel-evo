---
name: cute-fp8-kernels
description: Write, debug, validate, and optimize NVIDIA CuTe DSL Python kernels using dense or block-scaled FP8 on Blackwell GPUs. Use for FP8 GEMM, layouts, TMA/TMEM pipelines, tcgen05 MMA, numerical checks, or remote B300 performance work.
---

# CuTe DSL FP8 kernels

Implement the requested kernel in the task's existing Python submission. The
task decides which of two modes applies:

- **Harness-owned evaluator:** edit only candidate code; do not define/call
  `main()`, allocate inputs, compute a PyTorch reference, or print `PASS`.
  Preserve the task's candidate entry points. The harness appends validation.
- **Standalone submission:** keep the file self-contained with allocation,
  oracle, checks, and `main()` as requested by that task.

The explicit `TASK.md` contract always overrides generic handbook examples.
Do not create another harness/spec/candidate tree unless requested. The GPU
implementation must be CuTe DSL Python in both modes.

## Establish the contract

Before editing, identify:

- operation, epilogue, and mathematical equation
- logical/physical shapes, layouts, strides, alignment, and tail behavior
- FP8 formats, scale direction/granularity/layout, accumulator, and output type
- required cases, correctness rule, and performance target

Ask only when a missing fact changes the operation. Otherwise choose a
conservative default and report it.

## Load references progressively

Read the smallest task-specific route; do not preload the whole handbook:

- Elementwise or reduction task: [reductions.md](references/reductions.md).
- Dense or block-scaled GEMM: [examples.md](references/examples.md), then
  [api-cutlass-4.6.1.md](references/api-cutlass-4.6.1.md).
- FP8 scale/format ambiguity: [fp8.md](references/fp8.md).
- Only when building a TMA/TMEM GEMM pipeline:
  [layouts.md](references/layouts.md) and
  [memory-pipelines.md](references/memory-pipelines.md).
- General DSL tracing question: [cute-dsl.md](references/cute-dsl.md).
- Correctness implementation: [correctness.md](references/correctness.md).
- Remote compatibility/submission: [b300.md](references/b300.md) and
  [submission.md](references/submission.md).
- Failed run: [debugging.md](references/debugging.md).
- Correct kernel being tuned: [performance.md](references/performance.md).

Start editing after the task-specific reference. Load a second reference only
for a concrete unresolved question or a compiler diagnostic. Runtime feedback
is authoritative when an installed signature still differs.

## Workflow

1. Identify harness-owned versus standalone mode and preserve its entry point.
2. Start from the closest local dense/MXFP8 recipe.
3. Keep static configuration explicit; compile once per specialization.
4. Enforce dtype/layout/alignment/tail restrictions in `can_implement`.
5. Build the oracle from actual quantized inputs and exact scale semantics.
6. In harness-owned mode use the exact `cute_harness check/run` commands from
   the prompt. In standalone mode submit as directed by the task.
7. Optimize only after every required correctness case passes.
8. Warm up, measure repeated kernel-only execution, and verify native FP8 MMA.

Never replace the implementation with PyTorch/Triton/CUDA C++, weaken tests,
hide an error, or run an uncontrolled tuning loop.

## Report

Leave the submission in place and report the operation, assumptions, remote
correctness evidence, timing method/latency, native-path evidence, and remaining
limitations.
