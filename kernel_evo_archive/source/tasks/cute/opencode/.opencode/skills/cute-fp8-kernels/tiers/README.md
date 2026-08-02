# CuTe adaptation tiers

These folders define the frozen, cumulative documentation arms used by the
CuTe adaptation study. Every arm receives its own task statement and starter
candidate. The general files are identical for every task.

| Study tier | Local information |
| --- | --- |
| Tier I — Bare (`bare`) | No documentation; only the task statement and starter |
| Tier II — Foundations (`docs`) | CuTe execution model, API signatures and parameters, layouts, Blackwell architecture, TMA, pipelines, TMEM, low-precision numerics, correctness, and performance method |
| Tier III — Examples (`examples`) | Tier II plus incomplete, task-neutral code fragments for each major API boundary |
| Tier IV — Error guidance (`errors`) | Tier III plus failure classification and diagnostic hints derived from pre-study runs |

Tier I has no documentation directory and no tier files.
`--disable-documentation` resolves to the same task-only packet.

## Fairness boundary

All tier files are identical for every task. They must not contain:

- task IDs, task shapes, task constants, or task-specific indexing;
- a verified candidate, complete GEMM scaffold, or runnable benchmark;
- a recommended tile, stage count, thread count, cluster shape, or epilogue;
- text copied from a candidate generated for one of the evaluated tasks;
- evaluator implementation, acceptance thresholds, or hidden reference data.

The documentation may explain public framework concepts, API signatures and
parameters, small neutral syntax examples, and failure categories. A task
statement remains the only source of operation-specific semantics.

## Source policy

The local source corpus was frozen before the study:

- NVIDIA CUTLASS CuTe DSL 4.6.1 RST documentation;
- NVIDIA-generated CuTe API pages for `cute`, `cute.arch`, `cute.runtime`, and
  `cute.nvgpu`;
- NVIDIA CUDA/PTX architectural descriptions represented in the local
  Blackwell notes;
- pre-study AIRI and KernelEvo compiler/runtime diagnostics.

Agent-facing tier files contain no external URLs and never instruct the model
to browse. Release-sensitive spelling is described locally; compiler feedback
remains the final compatibility check for a concrete object combination.
