---
name: cute-fp8-kernels
description: General documentation for adapting to NVIDIA CuTe DSL and low-precision kernel development on Blackwell.
---

# CuTe DSL adaptation documentation

This skill contains the frozen, task-independent local documentation used by
the four-tier adaptation study. It teaches the framework without providing a
task implementation or selecting task-specific material.

1. Tier I — task and starter only; no documentation.
2. Tier II — local CuTe API, layouts, Blackwell, asynchronous execution,
   FP8/FP4, correctness, and performance foundations.
3. Tier III — Tier II plus incomplete, task-neutral code fragments.
4. Tier IV — Tier III plus common error explanations and diagnostic hints.

The experiment runner materializes the allowed files into each run. Read only
the files listed in the prepared packet. The task statement is authoritative
for the operation, shapes, physical layouts, precision, scales, entry point,
and acceptance contract.

## Working rules

- Preserve the supplied candidate ABI and edit only the candidate file.
- Use CuTe DSL for candidate GPU work.
- Do not add inputs, a reference implementation, compilation, timing, `main()`,
  or success output when the harness owns them.
- Treat the installed CUTLASS release and its compiler feedback as
  authoritative for exact API compatibility.
- Use only the materialized local documentation; do not browse for missing
  implementation details.
- Establish correctness before optimization.
- Change one subsystem at a time and act on the earliest deterministic error.
- Never weaken validation, hide an error, or replace the kernel with a
  framework reference operation.
