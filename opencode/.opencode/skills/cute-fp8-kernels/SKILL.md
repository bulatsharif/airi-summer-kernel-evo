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

- for dense FP8 GEMM, start from the compile-verified
  [candidate-dense-gemm-template.py](references/candidate-dense-gemm-template.py);
- for a separate bias/activation pass over `[M,N]`, adapt
  [candidate-elementwise-template.py](references/candidate-elementwise-template.py)
  without changing its indexing or launch;
- preserve that template's TMA, pipeline, TMEM, and output-copy flow until the
  dense GEMM reaches remote execution;
- use [candidate-gemm-api.md](references/candidate-gemm-api.md) only to explain
  a concrete template line, not to reconstruct the whole kernel;
- after a failure, route its first diagnostic through
  [candidate-error-atlas.md](references/candidate-error-atlas.md);
- edit only `submission.py`;
- preserve the starter's JIT entrypoint;
- do not add `main()`, inputs, an oracle, timing, or PASS reporting;
- use `python3 -m cute_harness check` and `python3 -m cute_harness run`;
- run those commands plainly, without pipes, redirects, command chaining, or
  added shell utilities;
- use the read tool for workspace/skill files; arbitrary shell and
  `python3 -c` probes are intentionally unavailable;
- treat the remote harness compiler as the installed CuTe API oracle and fix
  its first concrete diagnostic before delegating exploration;
- treat the harness result as authoritative.

Before the first edit in candidate-only mode, read only the task reference and
the single template or scalar/reduction chapter selected by it. Do not
proactively read the API atlas, general examples, layouts, memory-pipeline,
correctness, performance, or standalone chapters. They are fallback material
for a concrete diagnostic, not required startup context. After those required
reads, implement immediately and run the local check; do not spend another turn
restating the task contract.

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

- Standalone FP8 format/scale work: [fp8.md](references/fp8.md). Candidate-only
  tasks get their exact scale contract from `TASK.md` and the selected task
  reference; do not load this general chapter before the first implementation.
- Candidate-only dense FP8 GEMM:
  [candidate-dense-gemm-template.py](references/candidate-dense-gemm-template.py).
- Candidate-only two-dimensional epilogue:
  [candidate-elementwise-template.py](references/candidate-elementwise-template.py).
- Candidate-only dense FP8 GEMM failure:
  [candidate-error-atlas.md](references/candidate-error-atlas.md), then the
  narrow relevant section of
  [candidate-gemm-api.md](references/candidate-gemm-api.md).
- New/changed standalone kernel: [cute-dsl.md](references/cute-dsl.md),
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

In candidate-only mode, do not read `correctness.md`, `b300.md`, or
`submission.md`; they describe the standalone contract. The version-pinned
candidate API chapter plus the remote compiler are sufficient. In standalone
mode, inspect examples shipped with the installed CUTLASS package only to
confirm release-specific APIs.

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

The local candidate check is an AST/policy check, not a CuTe import or compile
test. Interpret its diagnostics literally. For example, “found 0
`@cute.kernel` functions” means the launchable device function lost its exact
`@cute.kernel` decorator; changing imports cannot fix it. Keep helper functions
as `@cute.jit`, but never change the starter's launchable kernel to
`@cute.jit`.

Use short diagnostic iterations. Change only the first rejected symbol or
layout, then rerun the same evaluator. Never reintroduce a spelling already
rejected by the worker, rewrite the whole pipeline after a late epilogue
failure, or spend repeated turns re-explaining shapes and scaling.

Never retry an identical candidate after a worker timeout or CUDA launch
failure. If the same diagnostic occurs twice, restore the compile-verified
template and reapply only the task constants and separate epilogue. Stop
immediately after the first harness `PASS`.

Never replace the implementation with PyTorch/Triton/CUDA C++, weaken tests,
hide an error, or run an uncontrolled tuning loop.

## Report

Leave the submission in place and report the operation, assumptions, remote
correctness evidence, timing method/latency, native-path evidence, and remaining
limitations.
