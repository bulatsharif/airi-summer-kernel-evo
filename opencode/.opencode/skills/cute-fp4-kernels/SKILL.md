---
name: cute-fp4-kernels
description: Write, debug, and validate NVIDIA CuTe DSL Python kernels for packed E2M1, NVFP4, or MXFP4 data on Blackwell GPUs. Use for FP4 unpack/dequantize kernels, native block-scaled FP4 MMA, packed-nibble indexing, FP4 scale contracts, or B300 FP4 compiler and correctness failures.
---

# CuTe DSL FP4 kernels

Implement the requested operation with CuTe DSL Python. Treat `TASK.md`,
`task.json`, and the harness compiler as authoritative for shapes, storage,
scale direction, accumulator/output types, entrypoint, and required primitives.

## Select the FP4 contract

Identify the contract before writing code:

- **Packed scalar E2M1:** bytes contain two independent four-bit values. Read
  [packed-e2m1.md](references/packed-e2m1.md), then adapt
  [candidate-packed-e2m1-template.py](references/candidate-packed-e2m1-template.py).
- **Native block-scaled MMA:** FP4 A/B and scale tensors are MMA operands. Read
  [block-scaled-fp4.md](references/block-scaled-fp4.md). Do not unpack to FP16
  when the task requires native FP4 tensor-core execution.
- **Failed attempt:** classify the first diagnostic with
  [error-atlas.md](references/error-atlas.md). Fix only that failure before
  changing the algorithm.

Do not infer native block-scaled helper names from dense FP8 APIs. The exact
FP4 MMA atom, scale-layout utilities, and supported tiles are release-specific.

## Preserve the submission boundary

For a prepared harness candidate:

- edit only `submission.py`;
- preserve the starter's public JIT entrypoint and argument types;
- keep launchable device functions decorated with exactly `@cute.kernel`;
- do not add `main()`, inputs, a reference, timing, or PASS reporting;
- run the task's exact `python3 -m cute_harness check` and `run` commands.

For a standalone probe, compile once, warm up, validate exhaustively, and time
only repeated kernel execution.

## Establish correctness

Write down:

1. the four-bit value format and nibble order;
2. logical element count versus packed byte count;
3. scale versus inverse scale, granularity, and application point;
4. accumulator and output dtype;
5. odd-tail behavior and required alignment;
6. whether native FP4 MMA is mandatory.

For packed E2M1, test all 16 bit patterns and both positions in a byte before
using random inputs. For block-scaled MMA, use distinct adjacent scale blocks
so a wrong physical scale layout cannot accidentally pass.

## Iteration rules

- Convert a packed byte to an integer before bitwise operations.
- Mask the low nibble with `& 0xF`; obtain the high nibble with `>> 4`.
- Apply the declared dequantization scale exactly once.
- Keep large loops dynamic with `cutlass.range`; use
  `cutlass.range_constexpr` only for small static loops.
- Treat the first remote compiler/runtime diagnostic as the API oracle.
- Never retry an identical timeout, launch failure, or illegal access.
- Stop after the first harness PASS unless optimization was requested.

Do not replace CuTe with PyTorch, Triton, CUDA C++, or a reference operator.
Do not weaken tolerances or claim native FP4 acceleration without native-path
evidence.
