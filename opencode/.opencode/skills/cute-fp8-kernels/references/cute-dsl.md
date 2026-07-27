# CuTe DSL working model

Use this reference for the mechanics that are easy to confuse with ordinary
Python or CuTe C++.

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

Official documentation:

- <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl.html>
- <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/dsl_jit_arg_generation.html>
- <https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_general/debugging.html>
