# CuTe DSL debugging guide

Debug from the outside inward. First identify whether the failure belongs to
the environment, tracing/JIT, code generation, launch, synchronization,
numerics, or measurement. Changing tile sizes before classifying the failure
usually hides the useful evidence.

## Contents

- [Diagnosis ladder](#diagnosis-ladder)
- [Tracing versus device execution](#tracing-versus-device-execution)
- [Useful diagnostics](#useful-diagnostics)
- [Environment and import failures](#environment-and-import-failures)
- [Tracing and type failures](#tracing-and-type-failures)
- [Layout and copy failures](#layout-and-copy-failures)
- [Compilation and assembly failures](#compilation-and-assembly-failures)
- [Launch and memory failures](#launch-and-memory-failures)
- [Pipeline hangs](#pipeline-hangs)
- [Numerical failures](#numerical-failures)
- [Performance anomalies](#performance-anomalies)
- [Remote-run evidence](#remote-run-evidence)
- [Minimal reproducer checklist](#minimal-reproducer-checklist)

## Diagnosis ladder

Use this order:

1. **Environment** — device, compute capability, CUDA, Torch, and CUTLASS import.
2. **Trace/JIT** — Python executes far enough to construct the compiled graph.
3. **Code generation** — MLIR/PTX generation and `ptxas` complete.
4. **Launch** — grid, block, cluster, shared memory, arguments, and stream are
   legal.
5. **Liveness** — every CTA and warp exits; no barrier or pipeline deadlock.
6. **Memory safety** — no misalignment, invalid address, or out-of-bounds tile.
7. **Numerics** — output matches the quantized-input oracle.
8. **Performance** — timing excludes setup and represents steady state.

Preserve the earliest useful error. A later “illegal memory access” reported at
synchronization may originate in a preceding launch.

## Tracing versus device execution

Normal Python `print(...)` in a JIT function runs while CuTe traces or compiles
the function. It is useful for static values such as layouts, types, tiles, and
compile-time branch choices. It does not print once per GPU thread.

Use the DSL/device printing facility, commonly `cute.printf(...)`, for runtime
device values. Limit it to one CTA and a small set of threads; device printing
changes timing and can obscure races.

When uncertain whether a value is static:

- print its Python type during tracing
- inspect whether it is a `cutlass.Constexpr`/compile-time object or an IR value
- avoid using an IR value to index a Python container
- avoid mutating a Python list or tuple based on runtime control flow

Remove debugging prints before performance measurement.

## Useful diagnostics

The current CuTe DSL family recognizes diagnostic environment variables such
as:

```bash
CUTE_DSL_DEBUG=1
CUTE_DSL_LINEINFO=1
CUTE_DSL_LOG_TO_CONSOLE=1
CUTE_DSL_LOG_TO_FILE=1
CUTE_DSL_LOG_LEVEL=DEBUG
CUTE_DSL_PRINT_IR=1
CUTE_DSL_KEEP=ir,ptx,cubin,sass
CUTE_DSL_DUMP_DIR=/tmp/cute-dsl-dump
```

Use only the minimum needed. Exact accepted values can vary by installed
release. `CUTE_DSL_KEEP=all` is convenient but can produce large artifacts.

Common compile options exposed by CuTe DSL tools include:

```text
--opt-level
--enable-assertions
--generate-line-info
--keep-ptx
--keep-cubin
--ptxas-options
--gpu-arch
```

An executor may expose generated artifacts through attributes such as
`__mlir__`, `__ptx__`, or `__cubin__`. Confirm attributes before depending on
them. Treat artifacts as diagnostics, not source-controlled build products.

Debug flags, line information, assertions, and device prints can alter code
generation, register pressure, cache keys, and timing. Disable them and rebuild
before recording final performance.

## Environment and import failures

For an import or compatibility failure, record:

```python
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
print(torch.__version__)
print(torch.version.cuda)
print(getattr(cutlass, "version", "unknown"))
print(getattr(cutlass, "CUDA_VERSION", "unknown"))
```

Then inspect the installed package:

- locate the symbol instead of guessing a new module path
- find the closest shipped Blackwell example
- compare that example's imports and configuration with the submission
- confirm that the selected operation supports the reported architecture

Do not install another CUTLASS version inside the submission to work around an
API mismatch. That makes the run non-representative and often exceeds service
limits.

## Tracing and type failures

Typical causes:

- a configuration value that should be `cutlass.Constexpr` is runtime
- a tuple/list changes length across control-flow paths
- runtime code mutates a composite Python value
- a DSL value is used as a Python list index or dictionary key
- a helper returns incompatible types on different branches
- a nested function captures unsupported `global` or `nonlocal` state
- runtime `break`, `continue`, or early `return` is used in a dynamic region
- a layout depends on a runtime value where a static layout type is required

Repair the static/dynamic boundary rather than casting values blindly.

CuTe DSL uses static typing and does not support arbitrary dependent types. If
an output layout type depends on a runtime argument, either specialize that
argument, choose a common static representation, or move the dynamic decision
into supported runtime indexing.

## Layout and copy failures

For a failed copy or partition:

1. print the source and destination tensors and layouts during tracing
2. state their logical coordinates and physical storage order
3. verify `cute.size`, `cute.cosize`, and byte size
4. verify the tiled copy and thread slice partition both tensors congruently
5. verify required vector alignment
6. reduce to one CTA tile without tails
7. initialize the destination with a sentinel and inspect missing stripes

Common visual symptoms:

| Symptom | Likely cause |
|---|---|
| Transposed result | mathematical B and physical `N,K,L` B were confused |
| Repeated rows/columns | partition or tile coordinate omitted a mode |
| Every second vector wrong | vector width or alignment mismatch |
| One stage stale | pipeline state advanced before copy completion |
| Only edge tiles wrong | unsupported tail or missing predicate |
| Whole tile zero | warp role, multicast mask, or descriptor not active |

For TMA, validate the whole descriptor contract: source tensor, SMEM
destination layout, tile shape, transaction bytes, multicast count/mask,
barrier, and participating CTA.

## Compilation and assembly failures

Separate the layer reporting the error:

- Python exception: host setup or tracing
- CuTe/MLIR diagnostic: unsupported DSL operation, type, or layout
- PTX generation error: target or lowering issue
- `ptxas` error: instruction, register, shared-memory, or architecture issue
- launch-time “invalid argument”: launch resource or cluster mismatch

When `ptxas` reports excessive resources, first inspect stage count, tile size,
epilogue fragments, register-resident temporaries, and static shared memory.
Do not treat resource failure as an invitation to remove required
synchronization.

When an instruction is unsupported, verify the actual compute capability and
the chosen MMA operation class. FP8 storage types do not guarantee that an
FP8-capable MMA descriptor was created.

## Launch and memory failures

Check:

- block thread count matches warp-role assumptions
- cluster dimensions satisfy the operation and hardware limits
- grid dimensions cover the output exactly as promised
- dynamic shared-memory byte count matches the shared-memory allocator
- every pointer uses the correct element type and address space
- every assumed alignment is true for base pointer and stride
- Torch storage remains alive until the launch completes
- the current CUDA stream is passed through correctly
- output is synchronized before Torch reads it

An illegal access may be reported on a later API call. Synchronize immediately
after the suspect launch during diagnosis.

Initialize output and guard regions with recognizable sentinel values. For
memory-safety tests, allocate padding around a logical tensor and confirm it is
unchanged after execution.

## Pipeline hangs

A hang is usually a protocol mismatch, not a slow kernel.

Reduce to:

- one problem tile
- one K tile
- one pipeline stage if supported
- one-CTA instruction mode
- no persistent scheduler
- the simplest epilogue

Then compare, as a single protocol:

- producer acquire/commit count
- consumer wait/release count
- barrier expected arrival count
- TMA transaction bytes
- stage index and phase transitions
- cluster multicast senders and receivers
- warp/warp-group participation
- prologue and drain behavior

Every acquired stage must eventually be committed or canceled according to the
pipeline API. Every consumed stage must eventually be released. All
participating threads must agree on dynamic control flow around collective
operations.

Do not fix a hang by increasing the remote timeout.

## Numerical failures

Follow this isolation order:

1. compare the actual FP8 tensor bytes/values with the intended quantization
2. test a single non-scaled tile
3. compare accumulator values before output conversion if accessible
4. test the epilogue independently
5. add block scales only after dense dataflow is correct
6. add multiple K tiles and pipeline stages
7. add multiple CTA tiles
8. add tails, clusters, two-CTA MMA, or persistent scheduling

Use structured inputs:

- identity-like patterns reveal transposes
- row/column ramps reveal coordinate errors
- one-hot K positions reveal scale-block selection
- all ones reveal reduction length errors
- alternating signs reveal dropped K tiles and accumulation order

If error grows sharply with K, inspect accumulation initialization and stage
reuse. The first K MMA must initialize/overwrite the accumulator; subsequent K
MMAs must accumulate.

If only block boundaries fail, inspect scale index, K-block size, and physical
scale-factor layout.

Never increase `atol`/`rtol` until the error distribution has been explained.

## Performance anomalies

Before changing the kernel, confirm:

- compilation is outside the interval
- inputs and outputs are already allocated
- warmups completed
- synchronization surrounds the measured region correctly
- the same specialization is being reused
- debug flags and device prints are disabled
- the timed shape is the claimed shape
- the implementation, not the reference, is timed

A fast result with wrong output is not a candidate. An unexpectedly tiny time
can indicate an asynchronous timing bug, an empty grid, a skipped branch, or a
kernel that does not write the full result.

Inspect generated PTX/SASS or profiler counters only after the measurement
contract is sound.

## Remote-run evidence

Read remote fields independently:

- `success`: service/process result
- `exit_code`: submission process status
- `stdout`: explicit checks and metrics
- `stderr`: Python, compiler, launch, and assertion diagnostics
- `timed_out`: possible hang or excessive work
- `device_time_ms`: profiler aggregation, directional unless controlled
- `profile_id`: downloadable artifact identifier
- `profile_error`: profiling failure that may be separate from kernel execution

Preserve the complete first useful compiler or runtime error in working notes.
When iterating, change one contract dimension at a time.

## Minimal reproducer checklist

- [ ] One file with the same imports and architecture target.
- [ ] Smallest shape that still fails.
- [ ] One CTA and one K tile where possible.
- [ ] Deterministic structured inputs.
- [ ] No benchmark loop, persistent scheduler, or optional epilogue.
- [ ] Explicit synchronization after the suspect launch.
- [ ] Static layouts/types printed during tracing.
- [ ] Output and guard sentinels checked.
- [ ] Failure remains a nonzero exit.
- [ ] The reproducer still uses CuTe DSL rather than a fallback.
