# Barrier loop reference

The legal lifecycle is:

```text
ready -> authoring -> evaluating -> evaluated -> profile review -> ready|complete
          prepare      evaluate       advance
              ^             |
              |-- island repair --|
```

- `iter prepare` is idempotent and returns existing packets when already in `authoring`.
- Authors edit prepared candidates in place. `island submit` is useful for copying an external candidate or
  preserving rationale; `iter evaluate` also accepts the prepared files as implicit submissions.
- `iter evaluate` resumes unfinished islands after interruption. It evaluates every island, then atomically
  updates island elites and the global best at the barrier.
- `iter advance` is invalid before evaluation. On the final step it marks the run complete.
- `iter review-profiles` emits one isolated compact-trace task per profiled candidate. When review is required,
  `iter advance` remains invalid until every emitted review is submitted.
- `island repair` is valid only for a compactly reported localized invalid result, is bounded by
  `max_repairs_per_island`, and reopens the current authoring barrier.
- Never edit an evaluated candidate directly. Archives are immutable snapshots; repair the reopened working
  candidate, resubmit it, and reevaluate the barrier.
- Pass the same `--runs-dir` to every command when using a non-default run directory.

Use `island context --format json` to recover one packet. Use `iter report --format json` for machine-readable
metrics. Inspect raw logs only when the compact result explicitly lacks enough information for a bounded repair.

Direct Python callers can use:

```python
from kernel_evo.agent import KernelEvoAgent

evo = KernelEvoAgent(".kernelevo/runs")
status = evo.init_run("evo.yaml")
tasks = evo.prepare_iteration(status["run_id"])
# Authors edit task.candidate_path.
report = evo.evaluate_iteration(status["run_id"])
```
