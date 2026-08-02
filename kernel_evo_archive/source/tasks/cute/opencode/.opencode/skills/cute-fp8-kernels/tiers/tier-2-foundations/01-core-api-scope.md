# Tier II: local CuTe DSL foundations

Tier II is the first documentation arm. Tier I is deliberately bare and
contains only the task statement.

This tier contains the local CuTe DSL execution model, public namespaces,
function signatures, parameter roles, value types, layout operations, copies,
MMA calls, launches, Blackwell execution, asynchronous pipelines, TMEM,
low-precision numerics, and validation methodology. It does not contain:

- complete kernel implementations;
- recommended tiles, stages, block sizes, or cluster shapes;
- a failure atlas or repair advice.

Higher tiers add those categories cumulatively. The task statement remains the
only authority for the requested operation, tensors, formats, shapes, scales,
entry point, and validation contract.

## Version and evidence boundary

The API descriptions are curated from NVIDIA CUTLASS CuTe DSL 4.6.1
documentation and neutral probes against the study worker. Public documented
spelling is used throughout. Where a function is release-sensitive, the text
states the object and parameter contract without inventing convenience
wrappers.

There are four distinct compatibility levels:

1. a Python symbol exists;
2. a decorated function traces;
3. generated IR and device code compile;
4. a launch with concrete layouts, address spaces, and resources succeeds.

An API signature establishes only the first level. The task's own compiler and
evaluation feedback establish the later levels.

## Namespace map

| Namespace | Contents |
| --- | --- |
| `cutlass` | scalar DSL types, `Constexpr`, compile-time helpers, DSL loop helpers |
| `cutlass.cute` | decorators, layouts, tensors, copy/MMA abstractions, layout algebra |
| `cutlass.cute.arch` | device indices, barriers, synchronization, SMEM/TMEM allocation |
| `cutlass.cute.runtime` | DLPack conversion, raw pointer construction, runtime adapters |
| `cutlass.cute.nvgpu` | NVIDIA GPU instruction and copy families |
| `cutlass.pipeline` | cooperative groups, barriers, and asynchronous pipelines |
| `cutlass.utils` | allocators and layout/operation construction utilities |
| `cutlass.utils.blackwell_helpers` | release-specific Blackwell construction helpers |

Import the namespace that owns an abstraction. Do not infer that an equally
plausible spelling exists under another namespace.

## Reading order

Start with the language, type, layout, copy/MMA, and control-flow API chapters.
Then read the architecture, TMA, pipeline, TMEM, numerical, correctness, and
performance chapters as the candidate requires.

All files in this directory are part of Tier II and are copied into the
agent's isolated workspace. No internet or repository lookup is required.
