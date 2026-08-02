# CuTe DSL FP8 documentation ablation

Status: **complete.** 46 arms across five tasks and three models, run
2026-08-01. Raw artifacts in `results/{ds,gem,qwen}-e{1,2,3,4}-*` and
`results/e{1,2,3,4}-l2-63`; cross-experiment tables and the failure taxonomy in
[`results/final-multi/`](results/final-multi/).

## Summary

An LLM agent reaches a correct CuTe DSL FP8 kernel without retraining when it is
given a **worked example** and an **executable feedback loop**, and the effect is
a threshold on correctness rather than a gradient on quality: across five tasks
the `bare` tier -- task text only -- produced **zero** working kernels in 30
turns, failing every time on types and signatures rather than on concepts, while
the tiers carrying a worked example solved all five, usually on turn 1. The rung
between them, `docs`, is where the interesting variance lives: it crosses the
correctness threshold on three of five tasks, but the kernels it yields range
from 0.05x (correct and twenty times slower than the reference) to 38.94x, so
conceptual prose can be sufficient for correctness and is never sufficient for
performance. Of the three knobs varied one at a time, the critic (E3) was neutral
-to-negative on every task and the profiler (E4) was null -- better on two tasks,
worse on three -- and both cost tokens. The strongest methodological result is
that **pass rate replicates and speedup does not**: re-running one arm under an
uncontended evaluator reproduced 6/6 exactly and moved its speedup from 74.77x to
38.94x, so every tier ordering by milliseconds in this report sits inside the
noise. Two model comparisons are far weaker than DeepSeek -- Gemma 4 31B managed
one real solve in 42 turns and it was *slower* than the reference, Qwen3.6-35B
none in 24 -- but that comparison covers only `level1_40`.

## Question

Can an LLM agent write a correct CuTe DSL FP8 kernel for B300, and how much does
supplied documentation change that? The agent starts from the public skeleton
and writes the kernel; the verified reference is only ever a timing denominator.

Primary metric is **solved runs** — did any turn in the run produce a correct
kernel. Secondary: turns to first pass, tokens to first pass, failure taxonomy.
Speedup is reported only for correct kernels and is not the headline; a model
that writes any correct CuTe FP8 GEMM has already cleared the interesting bar.

## The FP16 → FP8 position

The task is KernelBench level2/63, whose reference model is dense FP32/FP16
torch. The deliverable here is the FP8 kernel, so **the torch reference → FP8
kernel is the precision transition under study**. `task.json` fixes it:
`gemm_inputs: float8_e4m3fn`, `accumulator: float32`, `bias: float32`,
`output: float32`. Correctness is anchored twice, which is what makes the
transition measurable rather than merely asserted — tightly against a torch FP8
reference (`full_max_absolute_error_vs_torch_fp8 ≤ 0.01`) and loosely against the
FP32 reference (`sample_max_absolute_error_vs_fp32 ≤ 0.1`). The agent must
restore `SCALE_A * SCALE_B` exactly once before the FP32 bias add, then ReLU,
then divide — the scale-restoration step is where FP8 kernels usually go wrong,
and it is checked by the tight bound.

## Protocol

Frozen in [`study-iter.yaml`](study-iter.yaml); the executed grid is
[`e1-l2-63.yaml`](e1-l2-63.yaml) plus one file per knob.

- Five tasks x four cumulative tiers x one replication = 4 E1 arms per task,
  plus three one-knob arms (E2-E4) per task. Seven arms per model-task cell;
  46 arms in total across DeepSeek (5 tasks), Gemma and Qwen (level1_40 only).
- E2/E3/E4 pin the tier to `errors` rather than selecting it per task, so all
  tasks could run in one wave. Pinned, not selected -- except on level2_63,
  where `errors` was the honest best-tier selection.
- Six barrier turns per arm; one OpenCode session per turn; 1 island.
- Model `deepseek/deepseek-v4-flash` for author and critic alike;
  `reasoning_effort: "none"`, 32,768 output cap, 900 s session timeout.
- In-turn evaluation budget `CUTE_AGENT_EVAL_BUDGET=6`.
- Start from `starter.py`; `seed_preflight` disabled because the skeleton is
  intentionally incomplete.
- B300 device time, seed 0, five warmups, 50 timed repetitions, 900 s timeout.
- Tier bundles materialized per arm and SHA-256 pinned: bare 195 cl100k tokens
  (1 file), docs 17,860 (14), examples 26,416 (21), errors 33,146 (27).

**As executed, two deviations from the frozen text.** Replications are 1, not 3,
so no tier ordering below is statistically separated. And arms did not share a
checkout: each ran in its own git worktree (see *Isolation*), which replaces the
"author concurrency 4" line — evaluation is still serialized by a single
`b300.lock` shared through a symlink, verified by an `flock` test.

## Isolation, and why the first attempt was discarded

A first pass at E1 produced a flat ladder — every tier solving on turn 1,
including `bare`. It was false. The `bare` arm had listed `results/` with
`find`, read the `examples` arm's promoted kernel from
`<run>/archive/iter-006-island-0/submission.py`, and emitted it back: 329 lines,
identical SHA-256, zero diff. The existing deny of `**/candidate/*.py` held —
both `read` and `cat` of the candidate were refused — but the archive copy of the
same kernel was not covered. `docs` had read the other arms'
`context/PREVIOUS_ATTEMPT.py`, `errors` leaked from turn 4 and `examples` from
turn 5.

No permission glob can fix this: every arm shares one path shape, so a deny that
blocks a sibling's `PREVIOUS_ATTEMPT.py` also blocks the arm's own — the file the
protocol requires it to see. Three further controls were tested and **measured**
rather than assumed:

| control | outcome |
| --- | --- |
| `**/candidate/*.py`, `**/agent-evals/**` deny | holds, but does not cover `archive/` |
| `external_directory: deny` | **does not** block absolute paths — a probe read a passing kernel from another checkout under it |
| one git worktree per arm | `results/` is untracked, so a fresh worktree has no sibling data on disk |
| `chmod 0111` on the worktree parent | blocks enumeration; `read` lists directories, so `ls` was never the only way |
| `chmod 0000` on historical run trees | the only control that stopped absolute-path reads |

Deny globs bind **relative to the session directory**. The same file denied
inside a worktree stayed readable through the main checkout's absolute path,
including `src/kernel_evo/cute_harness/examples/hopper_wgmma_gemm/kernel.py` — a
complete verified FP8 GEMM, i.e. the `examples` tier — which would have handed
`bare` and `docs` the answer. Final state, each verified by probe: 7 worktrees at
one commit, randomized names, non-listable parents, and ten answer-bearing
directories in the main checkout at mode 0000. The tripwire that watched every
session for cross-arm references fired zero times on the run reported here.

## A passing kernel that never ran

`bare` recorded four harness PASSes at 0.0128 ms — "29.9×". They are not solves.
Its candidate calls `fp8_gemm_kernel(...)` with no `.launch(...)`, so **no kernel
reaches the device**; the device profile for those turns contains no
`kernel_cutlass_*` entry at all, only torch init, the fp32→fp8 convert, and the
harness's own cuBLAS reference. The output buffer it was handed already held the
reference result, so validation read `full_max_abs=0.000000` and the timer
measured an empty region. The arithmetic is what gave it away: 137.4 GFLOP in
12.8 µs is ~10.7 PFLOP/s, above the device's dense FP8 peak.

So in this harness **launching nothing passes, while launching something wrong
fails** — which is exactly why `bare` turns 4–5 failed with `out_abs=0.000000`
while the no-op turns "succeeded". The evaluator was not modified. Instead
[`cross_experiment.py`](cross_experiment.py) requires a candidate-authored kernel
in the profile before counting a solve. Every table below uses that stricter
criterion, so **they disagree with `results/*/REPORT.md`**, which reports the
harness verdict unchanged: `analyze.py` still scores Bare 4/6 at 29.88×.

## Results — five tasks, three models, 46 arms

Every pass below is profile-verified: a turn counts as solved only if the device
trace contains a candidate-authored `kernel_cutlass_*` kernel. Arms marked
`WARN N no-op` had N harness PASSes that launched nothing; those are excluded
from the pass count. See *A passing kernel that never ran*.

### The documentation ladder, DeepSeek, all five tasks

| tier | `l1-02` | `l1-40` | `l2-12` | `l2-14` | `l2-63` |
| --- | :-: | :-: | :-: | :-: | :-: |
| `bare` | 0/6 | 0/6 | 0/6 | 0/6 | 0/6 |
| `docs` | 4/6 | 6/6 | 3/6 | 0/6 | 0/6 |
| `examples` | 6/6 | 5/6 | 5/6 | 6/6 | 5/6 |
| `errors` | 6/6 | 5/6 | 5/6 | 6/6 | 6/6 |

`bare` is **0/6 on every task** -- 30 turns, no working kernel, and the failures
are always types and signatures (`Did you mean: 'dtype'?`,
`module 'cutlass.cute' has no attribute 'launch'`), never concepts. That is the
one result that never moved across re-runs, models, or tasks.

`docs` is the interesting rung: it solves 3 of 5 tasks, and where it solves, the
quality varies enormously -- 0.05x on level2_12 (a correct kernel 20x slower
than the reference) against 38.94x on level1_40. Conceptual documentation is
sufficient for correctness on some tasks and never sufficient for performance.

### Per-task detail

#### DeepSeek V4 Flash — `level1_02_vector_scale_fp4`

Baseline 0.1324 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 | — | 16,495,557 | 86,142 |
| `docs` | 4/6 | 2.35× | 11,552,872 | 88,905 |
| `examples` | 6/6 | 3.15× | 9,826,937 | 80,826 |
| `errors` | 6/6 | 3.31× | 9,262,849 | 104,053 |
| E2 retrieval (prompt) | 6/6 | 3.25× | 10,285,436 | 63,699 |
| E3 feedback (critic) | 3/6 | 1.48× | 10,425,892 | 77,005 |
| E4 profiling | 5/6 | 2.92× | 7,744,227 | 60,316 |

#### DeepSeek V4 Flash — `level1_40_layer_norm_fp8`

Baseline 8.4256 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 ⚠1 no-op | — | 13,139,038 | 82,843 |
| `docs` | 6/6 | 38.94× | 6,260,481 | 66,207 |
| `examples` | 5/6 | 23.68× | 9,742,208 | 92,604 |
| `errors` | 5/6 | 40.34× | 5,055,034 | 48,432 |
| E2 retrieval (prompt) | 5/6 | 68.50× | 9,267,260 | 76,541 |
| E3 feedback (critic) | 5/6 | 70.42× | 10,362,636 | 94,924 |
| E4 profiling | 6/6 | 16.37× | 8,144,977 | 97,779 |

#### DeepSeek V4 Flash — `level2_12_gemm_multiply_leaky_relu_fp8`

Baseline 0.3845 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 | — | 6,460,575 | 41,222 |
| `docs` | 3/6 | 0.05× | 6,216,678 | 62,964 |
| `examples` | 5/6 | 1.87× | 9,837,921 | 70,801 |
| `errors` | 5/6 | 2.75× | 5,654,093 | 65,832 |
| E2 retrieval (prompt) | 6/6 | 2.91× | 5,681,663 | 49,941 |
| E3 feedback (critic) | 6/6 | 1.36× | 2,437,685 | 18,089 |
| E4 profiling | 6/6 | 1.39× | 1,912,398 | 15,027 |

#### DeepSeek V4 Flash — `level2_14_gemm_divide_sum_scaling_fp8`

Baseline 0.3652 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 | — | 13,308,106 | 91,264 |
| `docs` | 0/6 | — | 11,578,927 | 73,782 |
| `examples` | 6/6 | 3.22× | 10,161,676 | 69,667 |
| `errors` | 6/6 | 3.10× | 6,047,599 | 55,805 |
| E2 retrieval (prompt) | 6/6 | 3.16× | 6,757,695 | 49,414 |
| E3 feedback (critic) | 5/6 | 1.36× | 6,900,279 | 47,815 |
| E4 profiling | 6/6 | 3.15× | 4,815,114 | 51,621 |

#### DeepSeek V4 Flash — `level2_63_gemm_relu_divide_fp8`

Baseline 0.3815 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 ⚠4 no-op | — | 10,323,585 | 61,096 |
| `docs` | 0/6 | — | 14,519,559 | 92,626 |
| `examples` | 5/6 | 3.23× | 8,762,557 | 72,828 |
| `errors` | 6/6 | 3.29× | 7,434,221 | 60,662 |
| E2 retrieval (prompt) | 4/6 | 3.54× | 12,986,542 | 102,068 |
| E3 feedback (critic) | 4/6 | 3.29× | 12,519,376 | 96,038 |
| E4 profiling | 5/6 | 3.38× | 11,168,385 | 83,196 |

#### Gemma 4 31B — `level1_40_layer_norm_fp8`

Baseline 8.2845 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 | — | 3,042,836 | 79,320 |
| `docs` | 1/6 | 0.88× | 4,567,945 | 80,080 |
| `examples` | 0/6 | — | 3,723,755 | 94,383 |
| `errors` | 0/6 | — | 2,505,678 | 59,196 |
| E2 retrieval (prompt) | 0/6 | — | 2,167,343 | 52,049 |
| E3 feedback (critic) | 0/6 ⚠4 no-op | — | 1,820,578 | 43,536 |
| E4 profiling | 0/6 ⚠3 no-op | — | 1,868,948 | 55,387 |

#### Qwen3.6-35B — `level1_40_layer_norm_fp8`

Baseline 8.2865 ms. Speedup is best passing turn against that baseline.

| arm | pass rate | speedup | input tokens | output tokens |
| --- | :-: | ---: | ---: | ---: |
| `bare` | 0/6 ⚠1 no-op | — | 3,911,916 | 113,081 |
| `docs` | 0/6 | — | 6,111,695 | 108,063 |
| `examples` | 0/6 | — | 4,084,244 | 68,422 |
| `errors` | 0/6 | — | 750,984 | 17,049 |


### Failure taxonomy (E1 arms, all tiers pooled per task)

| arm set | evals | passed | failed | import/API | compile | layout/shape | numeric | other | zero-output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ds-e1-l1-02 | 119 | 65 | 50 | 23 | 12 | 0 | 0 | 15 | 0 |
| ds-e1-l1-40 | 85 | 51 | 31 | 6 | 2 | 0 | 0 | 23 | 15 |
| ds-e1-l2-12 | 59 | 44 | 15 | 7 | 0 | 0 | 8 | 0 | 4 |
| ds-e1-l2-14 | 93 | 46 | 46 | 25 | 7 | 1 | 0 | 13 | 0 |
| e1-l2-63 | 86 | 52 | 34 | 12 | 10 | 1 | 5 | 6 | 4 |
| gem-e1-l1-40 | 167 | 2 | 143 | 92 | 32 | 0 | 0 | 19 | 15 |
| qwen-e1-l1-40 | 72 | 2 | 58 | 11 | 7 | 2 | 0 | 31 | 23 |

`zero-output` counts numeric failures with `out_abs=0.000000` -- the kernel
launched and wrote nothing, counted apart from a scaling error. Gemma and Qwen
dominate it, which is the same defect as the no-op passes: both models
frequently emit CuTe that traces but never launches.

### What replicates and what does not

| finding | evidence | verdict |
| --- | --- | --- |
| `bare` cannot write a working kernel | 0/6 on 5 tasks, 3 models | **holds** |
| a worked example crosses the correctness threshold | `examples`/`errors` solve all 5 tasks | **holds** |
| the critic (E3) does not help | 3/6, 5/6, 6/6, 5/6, 4/6 vs controls 6/6, 5/6, 5/6, 6/6, 6/6 | **holds, negative** |
| the profiler (E4) improves speedup | 2.92x, 16.37x, 1.39x, 3.15x, 3.38x vs 3.31x, 40.34x, 2.75x, 3.10x, 3.29x | **null** |
| tier ordering by speedup | same arm re-run gave 74.77x then 38.94x | **does not replicate** |

The last row is the important caveat. Re-running `ds-e1-l1-40-docs` under an
uncontended evaluator reproduced its pass rate exactly (6/6 both times) and its
speedup not at all -- 74.77x then 38.94x, a factor of 1.9. **Pass rate
replicates; speedup does not.** Every ordering of tiers by milliseconds in this
report sits inside that noise band and must not be read as an effect.

A second re-run changed a conclusion outright: `ds-e1-l1-02-docs` scored 0/6
originally with 5 of its 6 turns killed at the author cap, and 4/6 when re-run
with the evaluator uncontended. Its original null was an artifact of evaluation
queueing, not a capability limit. Arms whose turns are truncated by evaluation
queueing understate capability, so cap-kill rate belongs beside any null.

### Multi-model

| model | tasks | turns | real solves | best speedup |
| --- | --- | ---: | ---: | ---: |
| DeepSeek V4 Flash | 5 | 210 | majority | 3.31x (l1-02), 38.94x (l1-40) |
| Gemma 4 31B | level1_40 | 42 | **1** | 0.88x -- slower than the reference |
| Qwen3.6-35B | level1_40 | 24 | **0** | -- |

Gemma's single solve came at `docs`, on turn 6, and was slower than the
reference it was asked to beat. Seven of its harness passes across E3/E4 were
no-ops. Qwen never produced a working kernel; its `errors` arm collapsed to
750,984 input tokens because the 33,146-token bundle exhausts its 50k context.
Both sit far below DeepSeek, but the comparison covers only `level1_40` -- the
Gemma and Qwen `level2_63` grids were cancelled before running, so nothing here
speaks to model dependence on the GEMM family.

## Single-task appendix: level2_63

The original seven-arm grid, kept in full because it is the only task with a
complete DeepSeek ladder *and* all three knobs, and because the isolation and
no-op defects above were both found in it. Its headline numbers appear in the
per-task table; everything below is the deeper read of that one task.

### E1 — documentation ladder

Baseline 0.3815–0.3835 ms per arm (each arm times its own).

| Tier | Bundle | Pass | Turns→1st | Best ms | Speedup | Input | Cached | Output | Reasoning | Tokens→1st | Device evals |
| --- | ---: | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bare | 195 | did not pass 0/6 | — | — | — | 10,323,585 | 9,936,512 | 61,096 | 198,552 | — | 8 |
| Documentation | 17,860 | did not pass 0/6 | — | — | — | 14,519,559 | 13,687,552 | 92,626 | 470,913 | — | 15 |
| Examples | 26,416 | **pass 5/6** | 1 | 0.1181 | 3.23× | 8,762,557 | 8,308,864 | 72,828 | 231,642 | 2,805,178 | 25 |
| Error hints | 33,146 | **pass 6/6** | 1 | **0.1160** | **3.29×** | 7,434,221 | 6,977,792 | 60,662 | 179,156 | 1,604,327 | 14 |

`cached_input_tokens` is genuinely reported here — the DeepSeek API returns it,
unlike SGLang where a structural 0 means "not reported". Caching carries 94–96%
of input in every arm.

**The step that matters is `docs` → `examples`.** Conceptual prose bought
nothing: `docs` spent the most of any arm in the grid — 14.5M input, 92.6K
output, 15 device evaluations — and solved nothing. The worked example converts
the task from unsolved to solved on the first turn. Because the tiers are
cumulative, E1 cannot separate "worked example helped" from "error hints
helped"; they are adjacent rungs, and `examples` 5/6 vs `errors` 6/6 is a
one-turn difference at 1 replication that should not be read as a separation.

**Adaptation vs from-scratch.** The `examples` tier contains a complete verified
FP8 GEMM building block, so results at `examples`/`errors` measure *adaptation*
of a supplied kernel. `bare` and `docs` are the genuine from-scratch condition —
and both are 0/6. Nothing in this study shows the model writing a correct CuTe
FP8 GEMM without one to work from.

### Failure taxonomy

Every failed evaluation, barrier and in-turn alike, by stderr class.

| Arm | Evals | Passed | Failed | import/API | compile | layout/shape | numeric | timeout | other | zero-output |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 bare | 14 | 10¹ | 4 | 0 | 0 | 0 | 4 | 0 | 0 | **4** |
| E1 docs | 21 | 0 | 21 | 8 | 7 | 1 | 0 | 0 | 5 | 0 |
| E1 examples | 31 | 22 | 9 | 4 | 3 | 0 | 1 | 0 | 1 | 0 |
| E1 errors | 20 | 20 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| E2 prompt | 27 | 19 | 8 | 0 | 5 | 0 | 2 | 0 | 1 | 0 |
| E3 critic | 25 | 14 | 11 | **10** | 1 | 0 | 0 | 0 | 0 | 0 |
| E4 profiler | 25 | 23 | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |

¹ spurious — no candidate kernel ran; see above.

The taxonomy shifts exactly with the ladder. `bare` never produces a kernel at
all: all four of its failures are `out_abs=0.000000`, an output tensor of zeros,
counted separately from a scaling error. `docs` reaches the device and fails on
API reality — 8 import/API and 7 compile, with signatures the prose names but
does not demonstrate: `TmemAllocator.__init__() missing 1 required positional
argument`, `'_Pointer' object has no attribute 'arrive_and_wait'`,
`module 'cutlass.cute' has no attribute 'range_constexpr'`. `examples` cuts
failures to 9 of 31, and `errors` eliminates them: **20 of 20 evaluations passed,
the only clean arm in the grid.**

### E2 — retrieval (delivery)

Same tier and bytes; only how they reach the author differs.

| Arm | Delivery | Pass | Turns→1st | Best ms | Speedup | Input | Input per solve | Output | Tokens→1st | Timeouts |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 errors | `files` | **6/6** | 1 | 0.1160 | 3.29× | 7,434,221 | **1,239,037** | 60,662 | 1,604,327 | 0 |
| E2 | `prompt` | 4/6 | 3 | **0.1080** | **3.54×** | 12,986,542 | 3,246,636 | 102,068 | 5,648,113 | **3** |

Eager delivery is worse on every measure that matters and better on the one that
does not. It cost **2.6× the input tokens per solve**, delayed first pass from
turn 1 to turn 3, and needed 3.5× the tokens to get there — yet produced the
fastest kernel in the grid at 0.1080 ms once it did.

**This arm is confounded and should not be read as a clean delivery comparison.**
Three of its six sessions were killed at the 900 s cap (exit 124) with the
candidate written but the work unfinished, against **zero** timeouts for the
identical tier under `files` delivery. Putting 33k tokens in every request moves
the bottleneck from context to wall-clock. That is a real cost of the mechanism
under a fixed session budget, but E2's failures are partly time-limited rather
than capability-limited. The budget was deliberately left identical across all
arms; raising it for E2 alone would trade one confound for another.

### E3 — feedback (critic)

| Arm | Pass | Turns→1st | Best ms | Speedup | Author input | Author output | **Critic tokens** | Critic hints |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 errors | **6/6** | 1 | **0.1160** | 3.29× | 7,434,221 | 60,662 | — | — |
| E3 | 4/6 | 1 | 0.1164 | 3.29× | 12,519,376 | 96,038 | **100,722** | 13 |

Critic tokens are reported in their own column and are never folded into the
author's counts. The critic is a clear negative: **two fewer solves, an
identical best speedup** (0.1164 vs 0.1160 ms — 0.3%, well inside run-to-run
noise), for 68% more author input plus 100,722 critic tokens of its own.

Its failure signature is diagnostic. E3 is the only arm whose failures are
overwhelmingly **import/API — 10 of 11**, against 0 of 8 for E2 and 0 of 2 for
E4 at the same tier. A read-only critic with no device access proposes API
spellings it cannot check, and the author follows them into methods that do not
exist. Advice that cannot be executed competes with the error text from a device
that can.

### E4 — profiling

The profiler writes the previous candidate's B300 kernel breakdown as
`PARENT_PROFILE.md`. **It can only act after a PASS** — a profile exists only
once a candidate has run correctly — so it is an optimization signal, not a
correctness one, and a null effect on pass rate is the expected result. The
metric that can move is speedup among passers.

| Arm | Pass | Best ms | Speedup | Turn-6 ms | Input | Output | Device evals |
| --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 errors | 6/6 | 0.1160 | 3.29× | 0.1160 | 7,434,221 | 60,662 | 14 |
| E4 | 5/6 | **0.1133** | **3.38×** | **0.1133** | 11,168,385 | 83,196 | 19 |

That is the predicted shape: pass rate unchanged within noise (5/6 vs 6/6),
speedup improved **+2.7%**. The trajectory is the evidence — E4 starts far worse
and optimizes past the control only at the end, once there is something to
profile:

| turn | 1 | 3 | 4 | 5 | 6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| E4 profiler | 0.2999 | 0.3050 | 0.1515 | 0.1269 | **0.1133** |
| E1 errors | 0.3711 | 0.1261 | 0.1179 | 0.1164 | 0.1160 |

E4 is also the cleanest arm after `errors`: 23 of 25 evaluations passed, its only
two failures compile errors. A 2.7% gain at 50% more input tokens is a real but
small effect that one replication cannot separate from noise.

### Cross-experiment table

| Arm | Knob | Solved | Turns→1st | Best ms | Baseline ms | Speedup | Input | Cached | Output | Reasoning | Tokens→1st | Device evals |
| --- | --- | :-: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| E1 | bare | 0/6 | — | — | 0.3815 | — | 10,323,585 | 9,936,512 | 61,096 | 198,552 | — | 8 |
| E1 | docs | 0/6 | — | — | 0.3822 | — | 14,519,559 | 13,687,552 | 92,626 | 470,913 | — | 15 |
| E1 | examples | 5/6 | 1 | 0.1181 | 0.3816 | 3.23× | 8,762,557 | 8,308,864 | 72,828 | 231,642 | 2,805,178 | 25 |
| E1 | errors | 6/6 | 1 | 0.1160 | 0.3818 | 3.29× | 7,434,221 | 6,977,792 | 60,662 | 179,156 | 1,604,327 | 14 |
| E2 | errors + delivery=prompt | 4/6 | 3 | **0.1080** | 0.3824 | **3.54×** | 12,986,542 | 12,368,768 | 102,068 | 410,755 | 5,648,113 | 21 |
| E3 | errors + critic=on | 4/6 | 1 | 0.1164 | 0.3829 | 3.29× | 12,519,376 | 11,677,568 | 96,038 | 374,641 | 1,780,536 | 20 |
| E4 | errors + profiler=on | 5/6 | 1 | 0.1133 | 0.3835 | 3.38× | 11,168,385 | 10,626,432 | 83,196 | 236,362 | 1,141,770 | 19 |

Seven arms accounted for; all token fields non-zero.

## Limitations

- **One replication.** This is now the sharpest limit. No tier or knob ordering
  is statistically separated, and the re-runs showed why: the same arm gave
  74.77x then 38.94x on identical settings. Pass rate reproduced exactly both
  times; speedup did not reproduce at all. Read the pass-rate column; treat every
  millisecond ordering as noise.
- **Five tasks, one family-heavy.** level2_12/14/63 are all GEMM-plus-epilogue;
  only level1_40 (reduction) and level1_02 (FP4 packing) sit outside that family.
  The `bare` = 0/6 result spans all five, but the `docs` behaviour splits 3-2 and
  may track how exotic the required API surface is rather than task difficulty.
- **Model coverage is thin and uneven.** DeepSeek ran all five tasks; Gemma and
  Qwen only level1_40, because their level2_63 grids were cancelled. Nothing here
  speaks to model dependence on the GEMM family. Gemma's endpoint is also
  50k-context against a 33,146-token bundle, so its top tiers are squeezed --
  its `errors` arm used the fewest tokens of its seven, and Qwen's collapsed to
  750,984.
- **`examples`/`errors` measure adaptation, not synthesis** -- those tiers carry
  a complete worked FP8 GEMM. Only `bare` and `docs` are from-scratch; `bare` is
  0/6 everywhere and `docs` solves three of five.
- **Evaluation queueing can manufacture a null.** ds-e1-l1-02-docs read 0/6 with
  5 of 6 turns killed at the author cap and 4/6 once uncontended. Any null must
  be reported with its cap-kill rate. E2 on level2_63 remains confounded this way
  (3 of 6 sessions killed).
- **The harness scores a no-op kernel as a pass.** Fourteen such passes were
  rejected across this grid, on three models and three tasks; one arm converged
  on the exploit, its author time falling 5x as its "success" rate rose. Any pass
  rate taken from `analyze.py` rather than the profile-checked criterion is an
  overcount.
- **A remote model service is not deterministic**, and two of the three
  evaluators used here are not interchangeable: an identical kernel measured
  0.039392 ms on the B300 and 0.049600 ms on the Hopper box, which also cannot
  run any level2_* task at all (tcgen05 vs sm_90a).

## Requirement map

| Requirement | Where it is demonstrated |
| --- | --- |
| Documentation / examples | **E1** — 4 cumulative tiers, 195 → 33,146 cl100k tokens, on 5 tasks |
| Retrieval | **E2** — same bytes by `files` vs `prompt` delivery |
| Feedback | **E3** — read-only LLM critic between turns, tokens reported separately |
| Profiling | **E4** — B300 kernel breakdown as `PARENT_PROFILE.md` |
| Compilation + tests | **all arms** — `cute-check` static policy gate every turn, then real B300 evaluation; 1,309 device evaluations across the grid (534 passed, 775 failed) |
| FP8 harness | **all arms** — E4M3FN inputs, FP32 accumulate, dual-anchored validation |
| Four metrics in every table | pass/not-pass with turns-to-first-pass, speedup vs the arm's own measured baseline, and the input/cached/output/reasoning token split |

## Evaluator stability

Ten independent submissions per task, five warmups and 50 CUDA-event timings
each, median reported. Predeclared acceptance CV ≤ 2%.

| Task | Mean | CV |
| --- | ---: | ---: |
| FP8 GEMM | 0.140624 ms | 0.181% |
| Packed FP4 scale | 0.132182 ms | 0.135% |
| FP8 LayerNorm | 8.216470 ms | 0.015% |

## Superseded protocol

An earlier design gave the model the **complete 321-line working kernel** and
asked for a minimal unified diff under a 1,024-token cap. Two matrices of 15
runs each were completed. They are recorded here because they justify the
change, not because they answer the question above.

| Tier | v1 passing | v2 passing |
| --- | ---: | ---: |
| Bare | 1/6 | 0/6 |
| API | 1/6 | 0/6 |
| Architecture | 0/6 | 0/6 |
| Examples | 5/6 | 2/6 |
| Error hints | 3/6 | 5/6 |

The winning mutation in a passing run was, in full:

```diff
-ACC_STAGES = 1
+ACC_STAGES = 2
```

All 17 invalid v1 candidates had instead raised `AB_STAGES` and exceeded shared
memory. So the measured "documentation effect" was really *does the bundle steer
the model toward the safe constant rather than the unsafe one* — a real finding,
but a much narrower one than framework mastery, and one where the model never
writes a TMA descriptor, an MMA atom, or a TMEM epilogue. No speedup was
established in either matrix; every result sat within 0.3% of the parent.

Those runs also motivated two fixes now in place: the diff parser's
lone-closing-fence bug, which cost three v1 runs and inverted the
Examples/Error-hints ordering between v1 and v2, and a strict non-overlapping
boundary between the task-only bare arm and every documentation arm.

Raw artifacts for both matrices are in commit `30fbf98`; they were removed from
the working tree when the evolve path was dropped.
