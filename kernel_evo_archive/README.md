# KernelEvo development archive

This directory preserves the KernelEvo work that produced the GPT-2 and Qwen
profiler ablations. It is intentionally self-contained: code, commit patches,
best kernels, and complete raw runs live together in this repository.

## Provenance

- Source repository: `svtdanny/kernel-evo`
- Source branch: `bulatsharif/integrate-cute-dsl-harness`
- Frozen source commit: `e8bffa5`
- Included development commits: `6292ff6` through `e8bffa5` (eight commits)

`source/` is a browsable snapshot of `e8bffa5`. It also contains the final Qwen
RoPE/RMSNorm convention clarification and `prof-qwen-standard.yaml`, which were
created during the follow-up runs. `patches/` contains one `git format-patch`
file per source commit so the original history can be replayed.

## Contents

- `source/` — KernelEvo implementation, tests, task definitions, and configs.
- `patches/` — the eight feature commits as portable patches.
- `best_kernels/` — winning submission and evaluation artifacts for each arm.
- `results/` — compressed full run trees, including contexts, candidates,
  numerical feedback, profiles, and per-turn results.
- `CHECKSUMS.sha256` — SHA-256 checksums for the raw run archives.

Extract a run with, for example:

```bash
tar -xzf results/qwen-timeline.tar.gz
```

The archives contain no API credentials.

## Nine-turn profiler ablation

Times are candidate kernel times in milliseconds. `fail` means the candidate
did not pass correctness for that turn.

| Task | Profiler | Turn 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | Best speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GPT-2 | off (6 turns) | fail | fail | 78.298 | 5.450 | 5.450 | **0.520** | — | — | — | 6.903× |
| GPT-2 | standard | fail | 4.082 | 4.085 | 4.084 | 3.614 | 3.626 | 3.627 | 0.831 | **0.829** | 4.333× |
| GPT-2 | timeline | 4.036 | fail | 1.193 | 1.193 | 0.510 | 0.348 | 0.348 | 0.347 | **0.316** | 11.361× |
| Qwen | off | fail | fail | 160.125 | 5.847 | 5.847 | fail | 3.215 | **0.626** | 0.627 | 79.662× |
| Qwen | standard | fail | 279.856 | 51.185 | 51.182 | 51.181 | 51.246 | **51.181** | 51.186 | 73.539 | 0.975× |
| Qwen | timeline | fail | 219.986 | 220.287 | 50.320 | 9.415 | 51.516 | 9.413 | 9.417 | **7.356** | 6.781× |

## Exclusive paired remeasurement

The latest GPT-2 timeline winner and Qwen no-profiler winner were remeasured
against their baselines in alternating order under the same exclusive server-B
lock. This rules out cross-run GPU contention as the source of their speedups.

| Task | Baseline, ms | Winner, ms | Median paired speedup |
| --- | --- | --- | ---: |
| GPT-2 | 3.589568 / 3.589536 / 3.593632 | 0.313856 / 0.313568 / 0.313312 | 11.447× |
| Qwen | 49.882526 / 49.882080 / 49.881023 | 0.627456 / 0.626432 / 0.626752 | 79.587× |

All remeasured candidates passed. Qwen's observed maximum absolute error was
0.0271–0.0391 against a tolerance of 0.08.
