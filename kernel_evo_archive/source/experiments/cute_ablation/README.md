# Can a model write CuTe DSL FP8 kernels from documentation?

One task per run, one model, one question: given the public skeleton and a
controlled amount of CuTe documentation, can the agent write a kernel that
compiles and is numerically correct on B300?

The agent never sees a working implementation. It starts from `starter.py` —
imports and constants, with `pass` in every kernel body — and writes the rest.
The verified reference is timed separately as the speedup denominator and is
never shown to the model. This is the same shape as KernelBench and as the
`airi-summer-kernel-evo` harness.

## Tiers

Cumulative, four levels:

1. `bare` — no documentation beyond the task statement and starter;
2. `docs` — local CuTe API plus layout reasoning, Blackwell architecture, TMA,
   pipelines, TMEM, FP8/FP4, correctness, and performance foundations;
3. `examples` — plus fixed, incomplete, task-neutral code fragments;
4. `errors` — plus general explanations and diagnostic hints for recurring
   failure classes.

The hard rule: a tier may teach framework mechanics but may never supply an
evaluated task's answer. Every task receives the same files at a given tier.
Tests reject task IDs, titles, tuning answers, executable templates, per-task
references, operation-dependent routing, and external URLs.

`--disable-documentation` and `--documentation-tier bare` both select the
task-only Tier I packet.

Inspect exactly what a tier sends:

```bash
PYTHONPATH=src uv run python -m kernel_evo cute ablation-context \
  level1_01_square_matrix_multiplication_fp8 --tier docs --output /tmp/bundle.md
```

## Running

Needs `opencode` on PATH, GNU `timeout` (`brew install coreutils` on macOS), and
`QWEN_BASE_URL`, `QWEN_API_KEY`, `CUTE_HARNESS_API_KEY` exported.

Preview — validates tooling, config, and tier digests, starts nothing:

```bash
PYTHONPATH=src uv run python experiments/cute_ablation/run_iter_matrix.py results/iter --dry-run
```

One arm end to end, before committing the full matrix:

```bash
PYTHONPATH=src uv run python experiments/cute_ablation/run_iter_matrix.py results/smoke --tier bare --replications 1 --steps 2
```

Full matrix:

```bash
PYTHONPATH=src uv run python experiments/cute_ablation/run_iter_matrix.py results/iter --concurrency 4
```

Then aggregate:

```bash
PYTHONPATH=src uv run python experiments/cute_ablation/analyze.py results/iter
```

## How a run works

Each arm is one independent KernelEvo run pinned to one tier:

```
prepare  ──► materialize tier docs into <run>/documentation/<tier>/
             copy the current candidate (turn 1: the skeleton)
   │
author   ──► one OpenCode session, agent `cute-fp8-author`,
             reads only the packet's listed files, edits only the candidate
   │
evaluate ──► local policy gate, then remote B300 under .kernelevo/b300.lock
   │
critique ──► optional: one read-only session turns the diagnostic into hints
   │
advance  ──► promote if better, next barrier
```

Six turns per arm, closed-loop and with no reference implementation. Each turn
sees the B300 diagnostic from the previous turn and the best candidate so far —
which is the skeleton again until one turn produces a valid kernel, since an
invalid candidate is never promoted.

## Delivery (`documentation.delivery`)

A tier says *what* the author knows; delivery says *how much work it is to get*.
Both arms send the same bytes with the same digest — only the retrieval burden
differs, so a difference between them is a retrieval result, not a content one.

```yaml
documentation:
  delivery: prompt   # or: files (default)
```

`files` is the frozen protocol: the tier is materialized into
`<run>/documentation/<tier>/`, listed in the packet, and the agent spends tool
calls opening what it judges relevant — it may read all of it, some, or none.
`prompt` materializes the same files but lists none of them; the whole bundle is
prepended to the authoring session prompt, so it is in context before the agent
acts and no tool call can add to it. The packet says so, to stop the agent
hunting for files that are not listed.

Cost differs accordingly: `prompt` pays the full tier on every turn (up to ~27k
cl100k tokens at `errors`), `files` pays only what the agent chose to open. Read
per-arm token totals from `summary.json` before comparing pass rates.

## Critic (`feedback.critic`)

```yaml
feedback:
  critic: true
```

One bounded read-only OpenCode session (`cute-fp8-critic`) runs between turns. It
reads the just-evaluated candidate and the harness diagnostic — error, stderr and
stdout tails, profiler summary when enabled — and returns at most three
one-sentence hints. Those lead the next turn's `FEEDBACK.md`, labelled
`Critic on turn N:`.

The critic writes nothing. Its hints are parsed from its session transcript, so
it needs no write permission near a candidate and cannot edit one. It is skipped
on the final turn, where nothing would consume it: an arm of `steps: 6` costs
five extra sessions, reported per turn as `critic_tokens` and `critic_seconds`.

Hints are keyed by island, not attached to an archive entry, because a failing
candidate is never promoted — its critique would otherwise be dropped exactly
when it matters most.

## Profiler feedback (`profiling.enabled`)

Every B300 run returns a PyTorch Chrome trace. Each evaluation now compacts its
own trace into `profile_summary.md`, written beside `profile.json` in the
evaluation directory: device, total GPU busy time, memcpy/memset totals, and the
top ten kernels by device time. The table is capped at 2 KB and is deterministic
for a given trace, so an arm can be replayed from its artifacts. A missing or
malformed trace records `profile_summary_error` on the evaluation record and
changes nothing else — the summary never fails or slows a timed run.

Whether the author ever sees it is the flag, off by default:

```yaml
profiling:
  enabled: true
```

With it on, the next turn's packet carries the previous candidate's table as
`PARENT_PROFILE.md` — closing the loop on *why* a kernel was slow, not just that
it was. With it off, the study runs exactly as frozen: the summary is written to
disk and no packet mentions it.

The legacy top-k table remains the default. To additionally select the complete
timeline formatter, enable the explicit second option:

```yaml
profiling:
  enabled: true
  timeline: true
```

For direct CLI initialization, the equivalent flags are `--profile
--profile-timeline`.

The timeline retains every GPU kernel, memcpy, and memset in timestamp order,
including an exact `gap before` each activity. It does not median-filter or
sample launches. CPU/Python trace records are omitted because they are not GPU
activity. Kernel names declared with `@cute.kernel` in the candidate are used
only as attribution hints; the packet separately lists candidate symbols that
were observed and symbols that never launched. The legacy aggregate is included
above the timeline, and the original Chrome trace remains the replay authority.

Only the island's own trace is ever delivered. The reference baseline's summary
stays in `<run>/b300/baseline/`, unreachable from any packet, for the same reason
the verified kernel is: an author must never see it.

## Parallelism

Author sessions run concurrently; B300 evaluation serializes itself through the
repository-wide `.kernelevo/b300.lock`. Wall time is bounded by
`total_evaluations × evaluation_seconds` almost regardless of `--concurrency`;
above 4–6 streams you are only queueing on the lock.

> **Launch every arm from this one checkout.** The lock is resolved per
> checkout, so separate clones or `git worktree`s each take their own and would
> overlap timed runs on the single remote B300, corrupting every measurement.

Arms are isolated by `--runs-dir` under the results tree. Completed arms are
skipped on rerun; an interrupted arm resumes at its last barrier.

An arm is keyed by tier and replication only, so **one results root holds one
protocol**. Give each delivery/feedback configuration its own `--results`
directory; pointing an existing root at a changed protocol is refused rather
than resumed, because it would rerun some arms and skip others into a single
analysis. Adding replications or narrowing `--tier` within a root is fine.

## Tier enforcement

`iter prepare` materializes the tier bundle into `<run>/documentation/<tier>/`
and points the packet at those copies, so a packet never references the shared
skill directory. Under `delivery: prompt` the copies are still made but nothing
is listed, and the bundle reaches the author in the session prompt instead.
[`../../opencode.json`](../../opencode.json) then denies `glob`, `grep`, and
every read under `tasks/**`, `src/kernel_evo/**`, `tests/**`, and
`experiments/**`.

Together these make the tier a hard bound: a Tier I author receives no
documentation and cannot reach Tier II material by guessing a path, because no
such path exists inside its run and the shared original is denied. The same denial keeps
`tasks/cute/cute_kernels/**` — the verified solutions — unreachable, which
matters far more now that the agent is writing kernels rather than editing one.

Verified by `test_prepared_documentation_is_isolated_inside_the_run`.

## Evaluator stability

Confirm the timing floor before interpreting any speedup:

```bash
PYTHONPATH=src uv run python -m kernel_evo cute stability \
  level1_01_square_matrix_multiplication_fp8 --output results/stability --runs 10
```

Frozen acceptance is coefficient of variation ≤ 2%. Measured previously at
0.181% (FP8 GEMM), 0.135% (packed FP4), 0.015% (FP8 LayerNorm).

API keys are read from the environment and never written to artifacts.

## Results

Full results for all five tasks and three models — pass rate, speedup, input and
output tokens per arm — are in [`REPORT.md`](REPORT.md), section
*Results — five tasks, three models, 46 arms*. Machine-readable tables:
[`results/final-multi/cross_experiment.json`](results/final-multi/cross_experiment.json)
and [`results/final-multi/failure_taxonomy.json`](results/final-multi/failure_taxonomy.json).

Read pass rates from `cross_experiment.py`, not `analyze.py`: the harness scores
a kernel that never launches as PASS, and only the former checks the device
profile for a candidate-authored kernel before counting a solve. Procedure and
the failure modes to guard against are in [`RUNBOOK.md`](RUNBOOK.md).
