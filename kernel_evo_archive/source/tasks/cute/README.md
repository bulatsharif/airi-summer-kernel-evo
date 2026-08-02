# Writing CuTe DSL FP8 kernels on B300

KernelEvo runs an agent that writes Python CuTe DSL kernels against task-owned
correctness and timing code on the remote B300 worker. The harness compiles each
manifest's named `@cute.jit` entry point directly; `ModelNew.forward` remains a
compatibility alias. Candidates never contain inputs, reference computation,
compilation, timing, or result parsing — the harness appends all of that.

Ten repository tasks are available:

```bash
PYTHONPATH=src uv run python -m kernel_evo cute task-list
```

Nine use FP8 inputs. `level1_02_vector_scale_fp4` uses packed E2M1 storage to
measure unfamiliar-format handling; it is not an NVFP4 tensor-core benchmark.

Each task ships two candidate files. `starter.py` is the public skeleton —
imports and constants with `pass` in every kernel body — and is what the agent
starts from. `cute_kernels/<task>/submission.py` is a verified reference used
only as the speedup denominator; it is never shown to the model, and
`opencode.json` denies reading it.

## Requirements

- Python 3.12 and `uv sync`;
- `opencode` on PATH and GNU `timeout` (`brew install coreutils` on macOS);
- an OpenAI-compatible model endpoint;
- `QWEN_BASE_URL`, `QWEN_API_KEY`, and `CUTE_HARNESS_API_KEY`;
- network access to the B300 evaluator.

API keys are read from the environment and are not written to run artifacts.

## Run one task

```bash
export QWEN_BASE_URL=<openai-compatible-v1-url>
export QWEN_API_KEY=<model-endpoint-key>
export CUTE_HARNESS_API_KEY=<b300-key>
```

```bash
PYTHONPATH=src uv run python -m kernel_evo run init \
  --config examples/agent/airi_cute_b300.yaml --run-id my-run
```

Then drive the barrier loop — `iter prepare`, author, `iter evaluate`,
`iter report`, `iter advance` — or let the study runner do it for every tier.
Check a candidate locally before it reaches the GPU:

```bash
PYTHONPATH=src uv run python -m kernel_evo cute task-check \
  level1_01_square_matrix_multiplication_fp8 submission.py
```

## Documentation ablation

`--documentation-tier` selects cumulative, frozen context:

| Tier | Additional context |
| --- | --- |
| `bare` | None beyond the task statement and starter |
| `docs` | Local CuTe API plus layout, Blackwell, asynchronous execution, FP8/FP4, correctness, and performance foundations |
| `examples` | Foundations plus fixed, incomplete, task-neutral code fragments |
| `errors` | Examples plus general explanations of recurring failure classes |

Every tier receives the same task, skeleton, evaluator, and budget. Within a
tier, every task receives the same general files: there are no task-selected
examples or references. Tests reject task IDs, titles, tuning answers,
executable templates, and per-task reference files in the documentation. The
files are self-contained and contain no external URLs. The model never receives
the evaluator suffix; the harness appends inputs, correctness checks,
CUDA-event timing, and result parsing immediately before remote evaluation.

Tier I is deliberately bare: its materialized packet contains `TASK.md` and no
tier documentation. `--disable-documentation` is equivalent to selecting
`bare`.

The agent writes the kernel from `starter.py` through the KernelEvo barrier loop
with OpenCode as the author:

```bash
PYTHONPATH=src uv run python experiments/cute_ablation/run_iter_matrix.py results/iter --dry-run
```

No Redis is needed. KernelEvo owns the one-island archive, parent selection,
evaluation, and promotion; the runner delegates only the authoring turn. All
local B300 clients share `.kernelevo/b300.lock` to prevent overlapping GPU
measurements, so every arm must be launched from one checkout.

Protocol and results are in
[`REPORT.md`](../../experiments/cute_ablation/REPORT.md).

## Outputs

Each run directory contains:

- `documentation/<tier>/`: the materialized bundle the author could read;
- `b300/baseline/result.json`: verified reference timing, the speedup denominator;
- `iter_*/island_0/candidate/submission.py`: the authored kernel;
- `iter_*/island_0/agent/`: OpenCode event stream, stderr, and token usage;
- `samples/iteration-*.json` and `summary.json`: per-turn and per-arm results.

The matrix analyzer writes `aggregate.json` and `REPORT.md`. It reports pass
rate, independent-replication pass rate with a 95% interval, geomean speedup
with incorrect or slower units floored at 1×, static documentation size, and
provider-observed requests, repairs, cap hits, input, cached-input, output,
reasoning, and total tokens.

Evaluator stability evidence and the complete methodology are in
[`experiments/cute_ablation`](../../experiments/cute_ablation/README.md).
