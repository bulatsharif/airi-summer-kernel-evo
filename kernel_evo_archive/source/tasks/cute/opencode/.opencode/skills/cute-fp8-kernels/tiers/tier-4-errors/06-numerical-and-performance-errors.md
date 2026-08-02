# Numerical and performance errors

## Classify the numerical pattern first

| Pattern | First hypotheses |
| --- | --- |
| nearly constant global ratio | scale direction or duplicate/missing scale |
| one row/column/group wrong | scale ownership or layout mode |
| edge-only error | tail predicate or output coverage |
| error grows with reduction length | accumulator initialization/precision |
| alternating or periodic error | partition/value layout |
| stale-looking regions | missing store or use-before-completion |
| NaN/Inf | overflow, uninitialized state, invalid conversion |
| nondeterministic output | race, phase error, incomplete synchronization |

Test the hypothesis with structured inputs before changing the implementation.

## Scale direction error

Write both producer and consumer equations. Determine whether the stored
metadata is a dequantization scale or reciprocal quantization multiplier.

A constant ratio across output often indicates:

- multiplying where division was required;
- applying A or B scale twice;
- omitting one operand scale;
- applying output scale at the wrong boundary.

Do not rename the variable without changing the equation.

## Scale indexing error

Use distinct, easily distinguishable scale values for each logical group in a
diagnostic case. Map:

```text
data coordinate
-> logical scale group
-> physical scale coordinate
-> loaded scale
```

If only selected groups fail, inspect mode mapping before accumulation
precision.

## Packed-format decode error

Symptoms:

- every pair of values swapped;
- signs wrong in alternating positions;
- magnitude set limited to unexpected values;
- odd final element incorrect.

Check nibble order, sign/exponent/fraction decode, byte versus logical index,
and odd-length padding separately.

## Accumulation precision error

Error that grows smoothly with reduction length can come from a narrower
accumulator or a changed summation order. Confirm accumulator type in the
actual MMA and intermediate epilogue, not only in a Python constant.

Fast-accumulation modes require a separate correctness comparison.

## Output conversion error

If the accumulator is correct but stored output is wrong:

- inspect register fragment values before conversion;
- confirm output dtype constructor;
- check saturation/rounding expectations;
- verify store partition and physical output layout;
- check epilogue arithmetic executes before conversion.

Do not mask overflow by increasing tolerance.

## Correct but unexpectedly slow

First verify measurement:

- compilation is excluded;
- allocation and reference work are excluded;
- warmup uses the same specialization;
- events and kernel use the same stream;
- synchronization is correct;
- device printing/debug flags are disabled;
- all output work is still performed.

Then verify generated instruction-path evidence.

## FP8 storage without FP8 tensor-core execution

Possible causes:

- operands converted to wider values before the core operation;
- scalar code implements the product;
- selected MMA operation does not match types/layouts;
- framework fallback performs the work.

Inspect construction and generated code. Correctness alone does not identify
the execution engine.

## Resource regression

A seemingly useful change can slow down by:

- reducing occupancy;
- increasing register spills;
- adding shared-memory traffic;
- increasing barrier count;
- extending prologue/drain;
- adding cluster coordination;
- increasing tail waste.

Compare one resource family at a time.

## Timing noise

If candidate differences are near evaluator variation:

- repeat bounded measurements;
- compare medians and dispersion;
- rerun the known-correct reference under the same conditions;
- treat overlapping noise as a tie.

Do not infer a speedup from a single minimum.

## Optimization repair loop

1. retain the correct candidate;
2. state one bottleneck hypothesis;
3. change one configuration family;
4. rerun correctness;
5. measure with the same protocol;
6. keep only reproducible improvements.

Tier IV does not recommend a tile, stage count, cluster, or epilogue. Those
choices require task-specific evidence and profiling.

## Kernel runs but the result is far outside tolerance

Symptoms:

```text
RuntimeError: validation failed: full_abs=..., sample_abs=...
```

The kernel compiled, launched, and completed: only the numbers are wrong. This
is a better position than any binding error, and the reported magnitudes tell
you which kind of mistake it is. An absolute error of the same order as the
output means a systematic fault — a missing or doubled scale factor, an
accumulator in the wrong precision, a transposed operand, or an epilogue applied
in the wrong order. An error a few multiples of the tolerance means rounding or
accumulation order, not structure.

Inspect:

- every scale the task specifies, applied exactly once and on the intended side;
- accumulation precision, which must not inherit the narrow input type;
- operand layouts and which mode is contiguous;
- the order of epilogue operations, which is fixed by the task statement.

Change one of these at a time. Re-running an unchanged kernel reproduces the
same magnitudes to the digit and tells you nothing new.
