# Task format

Each task directory contains:

- `task.json` — the machine-readable contract;
- `TASK.md` — the task statement shown to the coding agent;
- `starter.py` — the incomplete candidate followed by the task-owned evaluator,
  separated by the harness marker.

The manifest's `baseline` points to a verified implementation used for
infrastructure checks and timing. It is never included in the evaluated
agent's context.

The authoring unit is:

```text
TASK.md + fixed general tier files + candidate prefix
  -> agent submission.py
```

The agent does not receive the evaluator suffix. Before a remote run, the
harness combines a policy-compliant candidate with that suffix into one
standalone file.

The shared skill location is declared in `task.json.agent_skills`. Tier files
are selected only by documentation level and are identical for every task:
there are no per-task references or operation-specific routing rules. Baselines,
previous runs, and other workspaces remain outside the readable packet.
