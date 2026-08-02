# Experiment memory

Mutable experiment records live outside the installed package, by default at `.kernelevo/cute-experiments.jsonl`.

KernelEvo agent runs use `<run>/cute/experiments.jsonl` and append one record only after the evaluation barrier owns the correctness, timing, profile, evidence, and promotion decision. Later packets retrieve at most `cute.context_lessons` matching one-line lessons; they never replay logs or entire records into author context.

Each JSON record must contain `task`, `hypothesis`, `change`, and `decision`. Recommended fields are `base_example`, `correctness`, `resources`, `performance`, `profile`, and `lesson_tags`.

```bash
kernel-evo cute record --record experiment.json
kernel-evo cute history --task attention --tag pipeline_stages
```
