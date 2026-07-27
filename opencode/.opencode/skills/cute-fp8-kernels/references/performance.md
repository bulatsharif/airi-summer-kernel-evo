# Performance measurement and tuning

Performance work begins only after the required correctness cases pass. Record
the measurement contract before comparing candidates; otherwise tile changes,
warmup effects, profiler aggregation, and launch overhead become
indistinguishable.

## Contents

- [Define the metric](#define-the-metric)
- [Compile and warm up](#compile-and-warm-up)
- [Measure correctly](#measure-correctly)
- [Report robust statistics](#report-robust-statistics)
- [Compute useful rates](#compute-useful-rates)
- [Interpret the remote metric](#interpret-the-remote-metric)
- [Evidence for the native FP8 path](#evidence-for-the-native-fp8-path)
- [Tuning order](#tuning-order)
- [Resource tradeoffs](#resource-tradeoffs)
- [Diagnose the bottleneck](#diagnose-the-bottleneck)
- [Run a bounded experiment](#run-a-bounded-experiment)
- [Final performance checklist](#final-performance-checklist)

## Define the metric

State:

- exact `M`, `N`, `K`, and batch `L`
- A/B FP8 formats
- dense or block-scaled operation
- accumulator and output type
- epilogue operations
- input/output layouts
- whether tails are present
- one-CTA or two-CTA MMA
- kernel-only or end-to-end latency
- warmup and measured iteration counts
- timing mechanism

Do not compare:

- kernel-only time from one candidate with end-to-end time from another
- a persistent kernel with a non-persistent kernel on different shapes
- a debug build with a release build
- a cold first launch with a warmed steady-state launch
- a correct candidate with one that skipped output work

## Compile and warm up

Compile once for each specialization:

```python
executor = cute.compile(jit_entry, a, b, c, stream)
```

Compilation, descriptor construction, memory allocation, input generation, and
reference calculation are outside kernel-only timing.

Warmup must exercise the same executor, shape, stream, and data path used by the
measurement. Synchronize after warmup before starting the timer. Ten warmups are
a reasonable starting point, but the correct count is the smallest count after
which measurements stabilize.

If the operation overwrites output, reuse the same output allocation. If it
accumulates into existing output, reset that state outside the interval or
include the reset consistently and label the metric.

## Measure correctly

CUDA-event timing is appropriate for a bounded kernel sequence on one stream:

```python
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for _ in range(iterations):
    executor(a, b, c, stream)
end.record()
end.synchronize()
average_ms = start.elapsed_time(end) / iterations
```

Confirm that the CuTe launch uses the same underlying stream on which the
events are recorded. A stream mismatch can make the interval meaningless.

For per-iteration distribution data, record event pairs or use
`cutlass.cute.testing.benchmark` when available in the installed release. Avoid
CPU wall-clock timing without explicit CUDA synchronization.

If the kernel is very short, time a loop of launches and divide by the count.
Keep the loop bounded and make sure repeated launches represent the intended
workload.

## Report robust statistics

For a distribution of repeated timings, report:

- median
- minimum or a low percentile
- a high percentile when variability matters
- repetition count

The median is the default comparison statistic. The minimum can approximate an
uncontended run but is sensitive to timing artifacts. A mean alone is easily
distorted by one cold or contended iteration.

Repeat final candidates in separate remote submissions. Shared-GPU contention
and service-level profiler behavior can vary between runs.

Keep raw samples in the submission only when they are few; otherwise print a
compact summary.

## Compute useful rates

For a dense batched GEMM:

```text
FLOPs = 2 * M * N * K * L
TFLOP/s = FLOPs / (median_ms * 1e-3) / 1e12
```

This convention counts multiply and add as two operations. State it when
reporting throughput.

For block-scaled GEMM, the GEMM FLOP count is still useful for comparison, but
it does not count scale movement or conversion work. Do not present it as a
complete instruction count.

For bandwidth-oriented epilogues, compute bytes from the actual tensors read
and written. Do not count cache reuse as repeated DRAM traffic without profiler
evidence.

## Interpret the remote metric

The service response includes `device_time_ms` from PyTorch Profiler. The
current service does not provide a controlled warmup for that field, so:

- use it as a directional signal
- inspect whether it aggregates one or several GPU activities
- do not mix it with the submission's per-launch event statistic
- repeat promising candidates
- prefer a controlled timing loop inside the file for final comparison

`profile_error` can occur even when the process and kernel succeed. Conversely,
a profile ID does not prove correctness.

## Evidence for the native FP8 path

FP8 tensor storage is not proof of FP8 Tensor Core execution. Obtain at least
one of:

- CuTe construction selecting the intended dense or block-scaled tcgen05 MMA
  operation
- generated PTX/SASS containing the expected MMA instruction family and FP8
  operand encoding
- profiler evidence consistent with the intended Tensor Core path

Also verify that the measured interval invokes the CuTe executor. PyTorch
matmul must remain confined to reference calculation.

Keep the evidence proportional to the claim. Construction-level evidence is
usually sufficient during development; a final hardware-path claim benefits
from generated-code or profiler confirmation.

## Tuning order

Change one group at a time:

1. **Baseline** — one-CTA, non-persistent, simple epilogue, correct native path.
2. **CTA/MMA tile** — improve work per CTA and Tensor Core utilization.
3. **Cluster and two-CTA mode** — use multicast/cooperation where the shape can
   amortize coordination.
4. **Pipeline stages** — overlap TMA and MMA without exceeding shared memory.
5. **Epilogue** — vectorize conversion/store; evaluate TMA store where
   compatible.
6. **Persistent scheduling** — reduce launch/scheduling overhead for shapes that
   provide enough work and predictable tile distribution.
7. **Specialized shape handling** — only for measured, required shapes.

For block-scaled kernels, tune dense A/B movement and MMA first, then scale
movement/layout and scale-TMEM staging. Keep scale transaction accounting
correct at every step.

Do not search all tile, stage, and cluster combinations blindly. Use shape and
resource constraints to eliminate invalid candidates before remote execution.

## Resource tradeoffs

Larger tiles may:

- increase Tensor Core work per CTA
- improve data reuse
- reduce scheduling overhead
- increase registers per thread
- increase shared-memory and TMEM use
- reduce resident CTAs
- amplify tail waste

More stages may improve latency hiding but consume more shared memory and can
increase prologue/drain cost. A stage increase that prevents useful residency
can regress performance.

Two-CTA MMA and larger clusters introduce coordination and multicast
opportunities. They are not automatic improvements for small or narrow
problems.

Persistent scheduling can reduce repeated tile scheduling overhead, but adds a
work-tile protocol and may worsen load balance when tile counts are small or
irregular.

Measure the whole epilogue. A faster MMA mainloop can be hidden by TMEM-to-RMEM
loads, type conversion, or global output stores.

## Diagnose the bottleneck

Use shape reasoning before a profiler:

- small `M*N`: launch/scheduling and epilogue overhead may dominate
- large `K`: mainloop and pipeline utilization matter more
- narrow `N`: tile waste or MMA shape mismatch may dominate
- many batches: grid scheduling and batch stride become important
- block scaling: scale traffic and extra SMEM/TMEM copies may matter

Then inspect available evidence:

- achieved Tensor Core activity
- active warps/CTAs and occupancy limits
- register, shared-memory, and TMEM usage
- TMA/global-memory throughput
- barrier and dependency stalls
- generated instruction mix
- output-store efficiency

Do not optimize a single counter in isolation. Lower occupancy can be acceptable
when a larger tile improves reuse and throughput.

## Run a bounded experiment

For each candidate, record:

```text
name:
operation:
shape:
tile_mn:
cluster_mn:
one_or_two_cta:
stages:
epilogue:
correctness:
median_ms:
min_ms:
tflops:
notes:
```

Set a small experiment budget before submitting. A practical first pass is one
baseline plus two or three changes supported by a concrete hypothesis. Stop
when:

- candidates fail the resource or implementation contract
- gains are below run-to-run noise
- a new bottleneck requires a different design
- the task's performance target is met

Preserve the last known-correct baseline. Re-run it if service conditions appear
to change.

## Final performance checklist

- [ ] All required correctness cases pass.
- [ ] The measured implementation uses the intended FP8 MMA path.
- [ ] Exact shape, types, scaling, layouts, and epilogue are stated.
- [ ] Compilation, allocation, input generation, and reference are excluded
      from kernel-only timing.
- [ ] Warmup and timing use the same executor and stream.
- [ ] CUDA work is synchronized at valid boundaries.
- [ ] Median and repetition count are reported.
- [ ] Throughput uses a stated formula.
- [ ] Debug flags and device prints are disabled.
- [ ] Final candidates are repeated remotely.
- [ ] Tuning stayed bounded and hypothesis-driven.
