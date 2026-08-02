# Correctness and performance method

Kernel development is a sequence of separate gates. Passing an earlier gate
does not imply passing a later one.

## Development gates

1. Candidate policy and public interface.
2. Python parsing and imports.
3. JIT argument binding.
4. tracing and API object construction.
5. IR/PTX compilation.
6. legal launch and resource configuration.
7. memory and synchronization safety.
8. numerical validation.
9. repeatability.
10. performance measurement.

Optimize only after gate eight passes for every required case.

## Write the contract before the implementation

Record:

```text
mathematical equation
logical tensor modes
physical shapes and strides
storage, compute, accumulator, and output dtypes
scale direction, granularity, and layout
alignment and divisibility guarantees
tail behavior
entry point and output ownership
```

These facts come from the task, not from the documentation tier.

## Correctness oracle principles

An independent oracle should:

- use the actual stored low-precision inputs;
- decode physical layouts and packed formats correctly;
- apply scale direction and granularity exactly;
- reproduce accumulator and epilogue semantics;
- convert to the requested output type at the same logical boundary.

The candidate must not contain or weaken the harness-owned oracle.

## Structured validation cases

Useful generic patterns:

- zeros and exactly representable small values;
- positive/negative signs;
- identity or one-hot structure;
- row/column ramps;
- alternating signs and cancellation;
- distinct scale values per group;
- multiple reduction tiles;
- multiple output tiles;
- supported boundary cases.

Each pattern isolates a class of mistakes. It does not determine an
operation's implementation.

## Output coverage and safety

Initialize the output with a sentinel when the evaluator supports it. Verify
that every promised element is written and no guard region changes.

For every access, reason about:

- coordinate bounds;
- physical stride;
- allocation span;
- alignment;
- address space;
- asynchronous completion;
- lifetime.

An unspecified launch failure is not a numerical mismatch and should not be
handled by changing tolerance.

## Error metrics

Combined absolute and relative comparison handles both near-zero and large
values:

```text
abs_error = abs(output - reference)
relative_error = abs_error / max(abs(reference), relative_floor)
pass if abs_error <= atol + rtol * abs(reference)
```

Tolerance follows from the declared numerical contract. Never loosen it only
because one candidate fails.

Also inspect:

- maximum and mean absolute error;
- guarded relative error;
- location and pattern of the worst errors;
- count of non-finite values;
- repeat-to-repeat variation.

## Measurement boundary

Kernel-only timing excludes:

- compilation and cache misses;
- tensor allocation;
- input generation;
- reference computation;
- descriptor construction owned outside the launch;
- profiling startup;
- device printing.

Warm up the exact executor, inputs, specialization, and stream used for timing.
Synchronize through a valid event or stream boundary.

## Latency and throughput

Report a robust statistic such as median latency over bounded repetitions.
Minimum latency can reveal the best uncontended run but should not replace the
distribution.

For a matrix product with batch count `L`:

```text
work = 2 * M * N * K * L floating-point operations
throughput = work / seconds
```

Use an operation-specific work definition for non-GEMM kernels. State the
definition rather than comparing unlike metrics.

## Native-path evidence

Low-precision storage alone does not prove low-precision tensor-core execution.
Evidence can come from:

- the selected documented instruction object;
- generated PTX/SASS instruction family;
- profiler counters consistent with the intended engine.

The measured interval must execute candidate CuTe code, not a framework
fallback.

## Resource-aware optimization

Potential dimensions:

- tile and vector width;
- block/warp role structure;
- shared-memory staging;
- pipeline depth;
- cluster cooperation;
- epilogue path;
- persistent scheduling;
- specialization.

Change one family at a time. Revalidate correctness and resources after every
change. A larger tile or deeper pipeline can reduce memory latency while also
reducing occupancy or exceeding resource limits.

## Comparing candidates

Keep:

- exact candidate hash/configuration;
- correctness status;
- median and dispersion;
- resource observations;
- instruction-path evidence;
- compiler/runtime diagnostics.

Treat differences within evaluator noise as ties. Retain a known-correct
candidate while exploring.
