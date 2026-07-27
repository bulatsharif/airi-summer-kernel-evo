# API-only agent experiment

The documentation arm tests whether a compact, task-selected CuTe pack is
enough to improve agent correctness without giving the agent a known solution
or general repository access.

## Context boundary

The prepared workspace contains exactly:

```text
TASK.md
task.json
submission.py
docs/
  INDEX.md
  architecture-and-dataflow.md
  tma-and-pipelines.md
  tmem-and-epilogue.md
  server-api-deltas.md
  task-adaptation.md
  examples/README.md
```

The pack contains compressed, version-pinned CUTLASS 4.6.1 documentation,
task routing, server-verified API deltas, and an explicit directory for future
code examples. It does not contain a complete task kernel or an optimization
schedule.

The Qwen agent may read only this prepared directory and may edit only
`submission.py`. Web access, repository search, known answers, previous
submissions, profiles, and evaluator internals remain unavailable.

## Prepare the Level 2 task

```powershell
python -m cute_harness prepare `
  level2_76_gemm_add_relu_fp8 `
  --output work/level2-api-only-001 `
  --with-api-context
```

The public `task.json` names only the copied `docs/INDEX.md`; it does not expose the
source documentation path or the private baseline.

## Run the API-only arm

Keep the SSH tunnel and `CUTE_HARNESS_API_KEY` in the parent environment, then
run:

```powershell
powershell -ExecutionPolicy Bypass -File tmp/run-token-pilot-arm.ps1 `
  -Arm harness `
  -TaskId level2_76_gemm_add_relu_fp8 `
  -Workspace work/level2-api-only-001 `
  -ResultRoot tmp/token-pilot-level2-api-only `
  -LabelPrefix level2-api-only `
  -HarnessDocs api-only `
  -AttemptBudget 8
```

The runner gives the agent a 12-minute wall-clock budget. Every arm uses the
same local compatibility wrapper and the same remote-attempt budget. The
wrapper records checks, remote submissions, timestamps, and evaluator results.

## Fair comparison

For the primary ablation compare:

1. `harness` with `-HarnessDocs api-only`;
2. `web`, with web search/fetch enabled and local CuTe documentation denied.

Use a freshly prepared workspace for each arm, the same task, model, output
limit, 12-minute deadline, attempt budget, and remote evaluator. Correctness is
the primary result; token count, time to first edit, time to first remote run,
attempt count, and final device/profile metrics are secondary measurements.

Do not compare a run that lacked remote API access against one that had it.
