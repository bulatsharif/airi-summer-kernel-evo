# Blackwell execution architecture

Blackwell kernels combine several independently scheduled engines. Performance
comes from overlap, but correctness comes from explicit ownership and
synchronization among those engines.

## Execution hierarchy

```text
grid
  -> optional clusters of CTAs
    -> CTA / thread block
      -> warps
        -> lanes
```

- A lane is one thread within a warp.
- A warp executes in SIMT fashion.
- A CTA shares shared memory and CTA barriers.
- A cluster permits limited cross-CTA coordination and distributed shared
  memory.
- A grid contains all CTAs launched for the kernel.

Collectives require the documented participant set. A branch that excludes one
required lane, warp, or CTA can deadlock even when excluded participants do no
arithmetic.

## Memory hierarchy

| Space | Scope | Typical role |
| --- | --- | --- |
| Global memory | device/application | input and output tensors |
| Shared memory | CTA or cluster | staged tiles and barriers |
| Registers | thread | scalar work and register fragments |
| Tensor memory | CTA group on Blackwell | `tcgen05` accumulators and related fragments |

Address space is part of type compatibility. A layout suitable for one space
does not make a pointer from another space legal.

## Independent engines

A high-throughput Blackwell dataflow may involve:

- TMA for multidimensional global/shared transfers;
- ordinary or asynchronous copy instructions;
- `tcgen05` tensor-core MMA;
- TMEM copy instructions;
- CUDA cores for scalar/vector epilogues;
- barriers that communicate completion.

These engines can run asynchronously relative to issuing threads and to each
other. Program order alone is not a completion guarantee.

## Conceptual dense dataflow

```text
global operands
  -> TMA or copy
  -> staged shared-memory operands
  -> tensor-core MMA
  -> tensor-memory accumulator
  -> register fragment
  -> conversion or epilogue
  -> global output
```

This is a vocabulary map, not a required implementation. Elementwise and
reduction kernels may use only global memory, registers, and shared memory.
Choose engines from the operation contract.

## Warp specialization

Warp-specialized kernels give different roles to different warps, such as:

- producer role: issue data movement;
- compute role: issue tensor-core operations;
- consumer role: load accumulators and store results.

Role assignment must agree with:

- the block thread count;
- cooperative-group participant counts;
- named barriers;
- pipeline producer/consumer groups;
- TMEM allocation ownership;
- cluster participation.

The role structure is a design invariant. It is not safe to change the block
size independently.

## CTA clusters

A cluster provides:

- a cluster coordinate and shape;
- cross-CTA synchronization;
- distributed shared-memory addressing;
- TMA multicast opportunities;
- instruction modes that cooperate across CTAs.

Cluster shape affects the grid mapping: grid dimensions count CTAs, while a
work tile may be owned by a cluster. Every dimension must divide or cover the
intended launch space under the chosen scheduling policy.

Multicast requires a mask derived from cluster coordinates and an identical
receiver protocol. Sending data to a CTA that does not wait or omitting a CTA
that does wait breaks completion.

## One-CTA and multi-CTA tensor-core work

Blackwell tensor-core instructions may be issued by one CTA or a cooperating
CTA group. Changing that mode affects:

- legal instruction shapes;
- cluster constraints;
- operand sharing;
- TMEM allocation and ownership;
- synchronization;
- which participants store output.

It is not merely a boolean performance option.

## Resource coupling

The following values form one resource equation:

```text
operand tile
pipeline stages
shared-memory layouts
barrier storage
tensor-memory columns
register fragments
threads and roles
cluster shape
```

Increasing a tile or stage count can raise reuse and overlap while also
increasing shared memory, TMEM, registers, and live barriers. A configuration
can trace correctly but fail launch-resource validation.

## Target checks

The study device is a B300-class Blackwell target. Exact compute capability,
CUDA toolchain, and CUTLASS build decide instruction availability. Distinguish:

1. architecture family supports a concept;
2. the installed package exposes an API;
3. the selected operation supports the concrete dtypes/layouts;
4. generated code targets the device;
5. launch resources fit.

Do not infer native tensor-core execution from an FP8 storage type alone.

## Ownership table

For each asynchronous object, write:

| Question | Required answer |
| --- | --- |
| Who initializes it? | one lane, warp, CTA, or cluster |
| Who issues work? | exact producer participant set |
| Who observes completion? | exact consumer participant set |
| What storage does it protect? | tensor/layout/stage |
| What event marks completion? | byte arrival, commit, wait, or barrier phase |
| When is it reused or released? | after all consumers finish |

This table makes architectural protocols explicit without selecting any
task-specific implementation.
