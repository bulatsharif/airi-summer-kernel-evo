# Performance

Tune only after every required correctness case passes.

## Measurement contract

Record exact `M,N,K,L`, formats/scales, accumulator/output, layouts, epilogue,
tile/cluster/stages, one- or two-CTA mode, and whether latency is kernel-only or
end-to-end.

Compile once, warm up the same executor/shape/stream, synchronize, then measure
bounded repetitions. CUDA events are suitable:

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

Confirm events and CuTe use the same underlying stream. For distributions, use
event pairs or `cutlass.cute.testing.benchmark` when installed. Report median,
minimum/low percentile, repetition count, and timing method. Repeat final
candidates in separate remote submissions.

Dense GEMM throughput:

```text
FLOPs = 2*M*N*K*L
TFLOP/s = FLOPs / (median_ms*1e-3) / 1e12
```

The service's `device_time_ms` is directional because it lacks controlled
warmup and may aggregate activities. Prefer the in-file controlled statistic.

## Native-path evidence

FP8 storage is not FP8 Tensor Core execution. Verify at least one:

- construction selects the intended dense/block-scaled tcgen05 MMA
- generated PTX/SASS has the expected MMA family/operand encoding
- profiler evidence supports the intended Tensor Core path

The measured interval must call the CuTe executor, not the reference.

## Tuning order

Change one family at a time:

1. correct one-CTA, non-persistent baseline
2. CTA/MMA tile
3. cluster and two-CTA mode
4. pipeline stages
5. epilogue/store
6. persistent scheduling
7. required shape specialization

Larger tiles improve reuse but consume registers, SMEM, and TMEM and may waste
tails. More stages can hide TMA latency but reduce residency. Two-CTA/clusters
add coordination; persistent scheduling adds load-balancing cost. Measure the
epilogue as well as MMA.

Use a small hypothesis-driven budget: baseline plus two or three valid
candidates. Log configuration, correctness, median/min latency, throughput, and
notes. Stop when gains are within noise or a new design is required. Preserve
and rerun the known-correct baseline if service conditions change.

Final measurements must exclude compilation/allocation/reference work, use
correct synchronization, disable debug output, state the FLOP convention, and
repeat the winner.
