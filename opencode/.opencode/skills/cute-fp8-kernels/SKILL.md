---
name: cute-fp8-kernels
description: Write, debug, validate, and optimize NVIDIA CuTe DSL Python kernels using dense or block-scaled FP8 on Blackwell GPUs. Use for FP8 GEMM, layouts, TMA/TMEM pipelines, tcgen05 MMA, numerical checks, or remote B300 performance work.
---

# CuTe DSL FP8 kernels

Implement the requested kernel in the task's existing Python submission.
PyTorch may support evaluation, but the GPU implementation must be CuTe DSL
Python.

## Select the submission contract

If the workspace contains `task.json` with
`validation.mode = local_owned_evaluator_v1`, use candidate-only mode:

- edit only `submission.py`;
- preserve the starter's JIT entrypoint;
- do not add `main()`, inputs, an oracle, timing, or PASS reporting;
- use `python -m cute_harness check` and `python -m cute_harness run`;
- treat the harness result as authoritative.

Otherwise use standalone mode: keep the complete self-contained submission,
including `main()`, input creation, validation, and bounded timing. Do not
create a harness/spec/candidate tree unless requested.

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
- Standalone correctness implementation:
  [correctness.md](references/correctness.md).
- Standalone remote compatibility/submission:
  [b300.md](references/b300.md) and
  [submission.md](references/submission.md). In candidate-only mode the local
  harness owns these concerns.
- Failed run: [debugging.md](references/debugging.md).
- Correct kernel being tuned: [performance.md](references/performance.md).

These references are sufficient for design. Inspect examples shipped with the
installed CUTLASS package only to confirm release-specific APIs.

## Workflow

1. Select candidate-only or standalone mode and preserve that contract.
2. Start from the closest local dense/MXFP8 recipe.
3. Keep static configuration explicit; compile once per specialization.
4. Enforce dtype/layout/alignment/tail restrictions, using `can_implement`
   when the submission exposes it.
5. In standalone mode, build the oracle from actual quantized inputs and exact
   scale semantics. In candidate-only mode, do not duplicate the harness oracle.
6. Run the evaluator for the selected mode and treat any failure as a failure.
7. Optimize only after every required correctness case passes.
8. In standalone mode, own warmup and repeated kernel-only timing. In
   candidate-only mode, rely on the harness measurement. Verify native FP8 MMA
   in both modes.

Never replace the implementation with PyTorch/Triton/CUDA C++, weaken tests,
hide an error, or run an uncontrolled tuning loop.

## Report

Leave the submission in place and report the operation, assumptions, remote
correctness evidence, timing method/latency, native-path evidence, and remaining
limitations.
