# CuTe DSL working model

Use this reference for the mechanics that are easy to confuse with ordinary
Python or CuTe C++.

## Contents

- [Compilation boundary](#compilation-boundary)
- [JIT arguments and specialization](#jit-arguments-and-specialization)
- [Compilation, executors, and caching](#compilation-executors-and-caching)
- [Supported Python model and limitations](#supported-python-model-and-limitations)
- [Control flow](#control-flow)
- [Functions, structs, and configuration](#functions-structs-and-configuration)
- [Tensors and layouts](#tensors-and-layouts)
- [Torch and DLPack boundary](#torch-and-dlpack-boundary)
- [Data movement and ownership](#data-movement-and-ownership)
- [Blackwell MMA path](#blackwell-mma-path)
- [Pipelines](#pipelines)
- [Debug and timing](#debug-and-timing)
- [Authoring checklist](#authoring-checklist)

## Compilation boundary

- `@cute.kernel` defines device code and is launched with explicit grid, block,
  cluster, shared-memory, and stream settings.
- `@cute.jit` defines a JIT-compiled host or device function. A host JIT wrapper
  commonly builds descriptors and launches a kernel.
- Runtime arguments are dynamic by default. Annotate configuration that must be
  known during compilation with `cutlass.Constexpr`.
- Python objects and Python control flow may execute while tracing. DSL values
  represent generated code. Do not feed a runtime DSL value into logic that must
  execute as native Python.
- Compile once with the intended argument types and static configuration. Reuse
  the returned executor for correctness runs and timing.

The normal structure is:

```python
import cutlass
import cutlass.cute as cute


@cute.kernel
def device_kernel(inputs: cute.Tensor, output: cute.Tensor):
    # Device-side CuTe operations.
    ...


@cute.jit
def launch(inputs: cute.Tensor, output: cute.Tensor, stream):
    device_kernel(inputs, output).launch(
        grid=(grid_x, grid_y, grid_z),
        block=(threads, 1, 1),
        stream=stream,
    )


compiled = cute.compile(launch, inputs, output, stream)
compiled(inputs, output, stream)
```

This is a structural example, not a complete kernel. Values such as grid size,
block size, cluster shape, layouts, and launch arguments must come from the
operation.

## JIT arguments and specialization

Classify every argument:

| Kind | Examples | Effect |
|---|---|---|
| Runtime scalar | M, N, K when dynamic | Passed to generated code |
| Runtime tensor/pointer | A, B, C | Storage address plus declared tensor type/layout |
| Compile-time scalar | stages, tile sizes, feature flags | Specializes generated code |
| Compile-time object | layouts, tiled MMA, tiled copy | Determines static IR types and structure |
| Stream | current CUDA stream handle | Controls launch ordering |

Use `cutlass.Constexpr` for arguments that must be available during tracing.
Compile-time values can control:

- tile and cluster construction
- static layout modes
- pipeline stage allocation
- unrolled loops
- operation class selection
- one-CTA/two-CTA and persistent feature branches

Do not mark every shape compile-time by habit. Static values produce specialized
executors and can improve generated code, but each distinct specialization has
compile and cache cost.

Runtime tensor arguments still have static type information such as element
type, rank, address space, and sometimes layout/shape properties. A JIT function
cannot return an arbitrary new static type chosen from an unconstrained runtime
value.

## Compilation, executors, and caching

`cute.compile(...)` performs compilation for the provided JIT entry and example
arguments, then returns a reusable executor:

```python
executor = cute.compile(jit_entry, a, b, c, stream)
executor(a, b, c, stream)
executor(a2, b2, c2, stream)
```

All later arguments must be compatible with the compiled signature and static
configuration.

Calling a decorated JIT function through supported runtime integration may use
an implicit cache. Explicit `cute.compile` is preferable in submissions because
it makes compilation placement and executor reuse visible.

CuTe DSL also maintains a file cache for compiled artifacts. The default is a
user-specific directory under `/tmp`; current releases provide controls such as:

```bash
CUTE_DSL_CACHE_DIR=/chosen/cache
CUTE_DSL_DISABLE_FILE_CACHING=1
```

Do not depend on a warm cache for correctness or report cache lookup as kernel
latency. Debug flags and compile options can create a distinct cache key.

Compilation options can control optimization level, assertions, target
architecture, line information, PTX assembler options, and retained artifacts.
Use the installed release's help/signatures for exact option spelling.

## Supported Python model and limitations

CuTe DSL is a statically typed compiled subset of Python, not a Python
interpreter running on the GPU.

Safe mental model:

- Python builds and specializes IR.
- DSL values represent runtime device/host values in that IR.
- static Python containers describe compile-time structure.
- generated device code follows CUDA execution and synchronization rules.

Current important limitations:

- dynamic code cannot arbitrarily change the structure or type of a tuple,
  list, object, tensor, or layout
- a Python container generally cannot be indexed with an arbitrary runtime DSL
  value
- dependent types are not supported; a runtime value cannot freely determine a
  returned static type
- `global` and `nonlocal` state are not a sound mechanism for compiled device
  behavior
- dynamic regions do not support arbitrary early `return`, `break`, or
  `continue`
- exceptions are a tracing/host concern, not general device control flow
- Python reflection, dynamic imports, generators, coroutines, and arbitrary
  object mutation are outside the kernel language model
- layout algebra uses 32-bit coordinates; layouts whose coordinate calculations
  overflow that range are unsupported

When a normal Python idiom fails, identify whether it changes static structure
or crosses the trace/runtime boundary. Refactor into a compile-time branch, a
runtime scalar operation, or separate specializations.

## Control flow

- `range(...)` emits a runtime IR loop even when its bounds are Python values.
- `cutlass.range(...)` also emits a runtime loop and supports options such as
  unrolling and software-pipeline hints.
- `cutlass.range_constexpr(...)` executes during compilation and fully unrolls.
- Use `cutlass.const_expr(condition)` for a compile-time branch.
- A normal `if` over a DSL predicate emits runtime control flow.
- Dynamic regions do not support early `break`, `continue`, or `return`.

Use this decision table:

| Need | Construct |
|---|---|
| Generate a fixed number of copies/instructions | `cutlass.range_constexpr(...)` |
| Runtime loop with hints | `cutlass.range(...)` |
| Ordinary runtime loop | `range(...)` under JIT semantics |
| Specialize away a feature | `cutlass.const_expr(...)` |
| Runtime per-thread predicate | normal `if` over a DSL predicate |

All threads participating in a collective copy, MMA, pipeline, or barrier must
take compatible control-flow paths. A syntactically valid runtime branch can
still deadlock if it causes only part of a warp/CTA/cluster to participate.

## Functions, structs, and configuration

Use small `@cute.jit` helpers for repeated compiled behavior. Pass tensors,
layouts, tiled-copy objects, and pipeline state explicitly enough that ownership
is clear.

A kernel class is useful for compile-time configuration:

```python
class Gemm:
    def __init__(self, tile_mn, cluster_mn, stages, use_2cta):
        self.tile_mn = tile_mn
        self.cluster_mn = cluster_mn
        self.stages = stages
        self.use_2cta = use_2cta

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, c: cute.Tensor):
        ...

    @cute.jit
    def __call__(self, a, b, c, stream):
        ...
```

Keep fields immutable during compiled execution. Validate configuration in
ordinary host code or an explicit `can_implement` method before compilation.

When the release supports JIT-compatible struct types, use them for stable,
typed groups of values—not as mutable general Python dictionaries.

## Tensors and layouts

- A CuTe tensor is an engine or pointer combined with a layout.
- A layout maps logical coordinates to storage coordinates. Shape alone does not
  establish row-major or column-major storage.
- Preserve hierarchical modes instead of flattening them casually; tiling,
  partitioning, TMA, and MMA depend on their structure.
- State both the mathematical operand shape and the physical CuTe shape. In
  NVIDIA GEMM examples, mathematical `B[K,N]` is frequently represented
  physically as `B[N,K,L]`.
- Check that tensor layouts are congruent with the tiled copy or tiled MMA before
  debugging individual indexes.

Frequently used layout operations:

| Operation | Purpose |
|---|---|
| `cute.make_shape` / tuples | Describe hierarchical extents |
| `cute.make_layout` | Combine a shape and optional stride |
| `cute.make_tensor` | Combine pointer/storage with a layout |
| `cute.local_tile` | Select a CTA-sized tile from a tensor |
| `cute.slice_` | Select modes or pipeline stages |
| `cute.tiled_divide` | Divide a layout into repeated tiles |
| `cute.size` / `cute.cosize` | Logical size / required backing storage |
| `cute.size_in_bytes` | Transaction or allocation size |

See `layouts.md` for the complete local layout working model.

## Torch and DLPack boundary

Torch tensors can cross the JIT boundary through DLPack conversion.

There are two useful modes:

- **implicit/dynamic conversion**: convenient for a runtime tensor whose
  compatible type/layout can be inferred
- **explicit/static conversion**: supplies an explicit CuTe layout and assumed
  alignment, enabling stronger specialization and avoiding ambiguity

Explicit conversion remains zero-copy: CuTe views the Torch allocation. Keep
the owning Torch tensor alive through execution.

Use explicit layout/alignment when:

- TMA descriptor construction needs a specific stride model
- alignment affects vectorization or `can_implement`
- the mathematical operand and physical tensor order differ
- static shape/layout improves code generation

Crossing the framework/JIT boundary adds microsecond-scale host overhead even
without a data copy. For a kernel-latency benchmark, compile once and reuse the
executor rather than rebuilding wrappers per iteration.

If a tensor dimension must remain runtime-dynamic, mark or represent it
according to the installed DLPack helper instead of accidentally baking its
example size into a static layout.

## Data movement and ownership

- Decide which warp or warp group owns each operation: descriptor setup, TMA
  load, MMA, epilogue, and store.
- Partition global and shared tensors with the same tiled-copy object used by
  the participating threads.
- Keep TMA transaction bytes, multicast masks, and barrier arrival counts
  consistent.
- Predicate boundary accesses only when the kernel contract permits tail
  shapes. Do not add a partial predicate to one side of a pipeline while leaving
  its barrier accounting unchanged.

Common operations are `cute.copy`, `cute.make_copy_atom`,
`cute.make_tiled_copy`, and `partition_S`/`partition_D` on a thread slice.
Blackwell TMA setup uses tiled TMA atoms whose source tensor, shared-memory
layout, tile, multicast configuration, and barrier transaction bytes agree.

## Blackwell MMA path

Blackwell examples use `cutlass.cute.nvgpu.tcgen05`. The usual high-level flow
is:

1. Move A and B from global memory to shared memory, commonly with TMA.
2. Execute `tcgen05.mma` from shared-memory operands.
3. Accumulate in tensor memory.
4. Load accumulator fragments from tensor memory to registers.
5. Convert, apply the epilogue, and store the output.

Reuse the construction and partitioning pattern from the closest installed
NVIDIA example. Do not reconstruct instruction descriptors from memory.

The dense Blackwell helper flow used by the pinned knowledge baseline is:

```text
make_trivial_tiled_mma
    -> make staged A/B shared-memory layouts
    -> make tiled TMA atoms for A and B
    -> launch @cute.kernel
    -> partition A/B/C through the tiled MMA
    -> TMA GMEM-to-SMEM pipeline
    -> tcgen05 MMA into TMEM
    -> TMEM-to-register accumulator load
    -> output conversion and GMEM store
```

## Pipelines

- Treat producer and consumer state as a protocol, not ordinary loop counters.
- Preserve initialization, arrive, wait, commit, release, and phase transitions
  together.
- A hang generally indicates mismatched participation, transaction bytes,
  barrier counts, or phases. Reduce to one tile and inspect ownership before
  changing stage counts.

## Debug and timing

- `CUTE_DSL_DEBUG=1` enables broad diagnostics.
- `CUTE_DSL_LINEINFO=1` enables Python-to-PTX/SASS line correlation.
- Debug and line-info settings can change generated code and cache keys. Disable
  them before final performance measurement.
- Exclude compilation and descriptor construction from kernel latency unless the
  task explicitly measures end-to-end invocation.

This reference summarizes the CuTe DSL behavior used by the project. When an API
signature differs, inspect the installed package rather than switching to a web
example from another release.

## Authoring checklist

- [ ] Device code is under `@cute.kernel` or a device-compatible `@cute.jit`
      helper.
- [ ] Launch setup is under a JIT host entry with explicit stream and resources.
- [ ] Compile-time and runtime arguments are deliberately classified.
- [ ] The executor is compiled once per specialization and reused.
- [ ] Dynamic control flow does not mutate static structure or split
      collectives.
- [ ] Tensor element type, logical shape, physical layout, and alignment are
      known.
- [ ] Torch/DLPack conversion preserves storage lifetime and stream ordering.
- [ ] Installed examples decide exact release-specific API signatures.
- [ ] Debug options are disabled for final measurement.
