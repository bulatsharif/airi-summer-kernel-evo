# Qwen 35B CuTe coverage run, 2026-07-28

This is a compact, reproducible summary of the first full 12-task coverage run.
Raw OpenCode event streams, generated candidates, downloaded profiles, secrets,
and evaluator workspaces are intentionally not committed.

## Protocol

- Model: `qwen-server/qwen3.6-35b-a3b` (Q8_0)
- Tasks: all 12 tasks discovered by `cute_harness`
- Agent sessions: one independent session per task
- Agent timeout: 600 seconds per task
- Remote development evaluations: unlimited inside the timeout; stop after PASS
- Authoritative evaluation: one separate evaluator run after every session
- Seed: `20260728`
- Benchmark: one warmup and three measured repetitions
- Context: curated local CuTe skill, task references, B300-verified math recipes,
  and compile-verified public templates; no web access

Command:

```text
python -m experiment run --all \
  --model qwen-server/qwen3.6-35b-a3b \
  --attempts 1 \
  --agent-timeout 600 \
  --gpu-timeout 600 \
  --seed 20260728 \
  --warmup 1 \
  --repeats 3
```

## Result

The authoritative evaluator accepted 10 of 12 final candidates (83.3%). Seven
tasks passed on the first agent-controlled B300 call. Three more passed after
diagnostic retries. There were 18 agent-controlled B300 calls in total.

| Task | Status | Agent B300 calls | Baseline ms | Candidate ms | Agent seconds |
|---|---:|---:|---:|---:|---:|
| Square GEMM | PASS | 1 | 0.143680 | 0.145952 | 231.8 |
| LayerNorm | PASS | 1 | 8.189888 | 35.785954 | 197.7 |
| ConvTranspose3d | TIMEOUT | 2 | - | - | 600.7 |
| L2-09 subtract/multiply/ReLU | PASS | 1 | 0.383616 | 0.151616 | 249.0 |
| L2-12 multiply/LeakyReLU | PASS | 1 | 0.387648 | 0.151264 | 267.4 |
| L2-14 divide/sum/scale | PASS | 1 | 0.366464 | 0.134016 | 302.1 |
| L2-29 Mish/Mish | PASS | 1 | 0.167520 | 0.230848 | 496.3 |
| L2-40 scale/residual | PASS | 2 | 0.387744 | 0.151488 | 287.8 |
| L2-55 max-pool/sum/scale | PASS | 3 | 0.261792 | 0.255392 | 539.0 |
| L2-63 ReLU/divide | PASS | 1 | 0.386560 | 0.160832 | 313.2 |
| L2-76 add/ReLU | PASS | 2 | 0.126688 | 0.152512 | 277.1 |
| L2-99 GELU/softmax | TIMEOUT | 2 | 0.269248 | - | 600.5 |

Token totals from the durable OpenCode event streams:

- uncached input: 440,189
- cached input: 4,411,779
- output: 75,688
- aggregate agent wall time: 4,362.546 seconds

## Observed gaps

- ConvTranspose3d compiled and ran twice, but both candidates were numerically
  wrong. The first had `max_abs=3.879138`; the attempted correction made the
  indexing error much worse. This is a task-logic/indexing gap rather than an
  unfamiliar CuTe API gap.
- GELU/softmax first exposed an SSA control-flow error (`row_max` was unbound),
  then compiled and ran with an incorrect cross-warp softmax reduction
  (`full_abs=0.544980`, `row_sum_abs=89.087303`). The scalar math recipes were
  sufficient; the missing context is a precise shared-memory cross-warp
  reduction recipe.
- L2-29 and L2-99 spent too much of the wall budget editing before the first
  evaluator call. L2-55 also repeated one failed candidate without a recorded
  mutation. The next runner should record candidate hashes and enforce clearer
  feedback-loop milestones.

## Proposed larger evaluation

Before spending the larger budget, add a hard, separately reported limit of
five agent-controlled B300 calls per session and record time-to-first-checker
plus a candidate hash for every call. Then run five independent sessions per
task: 12 tasks x 5 sessions = 60 sessions, with the same 600-second timeout.

Primary metrics should be pass@1, pass@5, pass by remote-call index, total and
cached tokens, time to first checker call, and repeated-identical-candidate
rate. Run the suite detached because the worst-case wall time is about ten
hours. After that reliability run, use the hardest tasks for an A/B comparison
between curated local context and the external-search baseline.
