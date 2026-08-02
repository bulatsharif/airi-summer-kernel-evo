---
name: kernelevo
description: Orchestrate visible multi-island GPU kernel optimization through the local KernelEvo harness. Use when Codex or OpenCode is asked to optimize, evolve, benchmark, or improve Triton, CUDA-inline, or CuTe kernels with KernelEvo; run the barrier loop and delegate only bounded candidate patches while KernelEvo owns scheduling, evaluation, profiling, archive state, and reports.
---

# Orchestrate KernelEvo

Use `kernel-evo` as the source of truth. Do not recreate the evolutionary algorithm in the coordinator.

## Start or resume

For a new run, initialize from a config:

```bash
kernel-evo run init --config evo.yaml
```

Or pass `--problem-path`/`--baseline`, `--backend`, `--steps`, and `--islands` directly. Save the returned
`run_id`. For an existing run, call `kernel-evo run status --run-id RUN`. Follow `next_action`; run commands
are safe to resume at a barrier.

## Run one barrier

1. Call `kernel-evo iter prepare --run-id RUN`.
2. Spawn exactly one bounded author for every returned task. Give each author only its `task_file`; authors
   may read the files listed there and edit only its `candidate_path`. Run independent islands concurrently.
3. Wait for every author. If an author returned rationale metadata, record it with `kernel-evo island submit`.
   Submission is optional when the author edited the prepared candidate in place.
4. Call `kernel-evo iter evaluate --run-id RUN`. Do not run candidate tests or benchmarks in author turns.
5. Relay `kernel-evo iter report --run-id RUN --format markdown` without reading raw logs.
6. Call `kernel-evo iter review-profiles --run-id RUN`. For every returned task, spawn exactly one
   bounded `kernel-profile-reviewer` with only its `task_file`. The reviewer reads the compact profile
   and its island candidate, edits only `output_file`, and never opens raw traces/logs. Submit each result
   with `kernel-evo island review-submit --run-id RUN --iter ITER --island ISLAND --review OUTPUT`.
   These reviews may add new optimization ideas; configured ideas are seeds, not a closed search space.
7. If the compact report lists a repairable island and the failure is localized,
   call `kernel-evo island repair --run-id RUN --iter ITER --island ISLAND`, give
   the returned `repair_file` to one bounded repair author, resubmit, then call
   `iter evaluate` and `iter report` again. Never repair a performance regression.
8. Call `kernel-evo iter advance --run-id RUN`. Repeat until status is `complete`.

Keep each author invocation equivalent to one candidate patch and each reviewer invocation equivalent to one
compact profile analysis. Use only the stateful `island repair` transition;
never edit an evaluated/archived candidate without reopening it. A performance regression is negative archive
evidence, not a repair task.

Read [references/interactive_loop.md](references/interactive_loop.md) for transition and recovery details.
Read [references/subagent_contract.md](references/subagent_contract.md) before delegating authors or repairs.
