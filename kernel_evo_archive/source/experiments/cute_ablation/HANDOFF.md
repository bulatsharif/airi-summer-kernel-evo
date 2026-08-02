# Session prompt — run E1–E4 on DeepSeek V4 Flash

Copy everything below into a fresh session.

---

Run the E1–E4 CuTe FP8 ablation end to end in `kernel-evo` (work from its root). Use the existing
`experiments/cute_ablation/run_iter_matrix.py`. Do not modify library code, tier documentation,
task manifests, or evaluator logic — create config yamls and write analysis/report files only.

## Fixed protocol

| setting | value |
|---|---|
| model | `deepseek/deepseek-v4-flash` (critic too, in E3) |
| steps (turns per arm) | 6 |
| islands | 1 |
| replications | 1 |
| in-turn eval budget | `CUTE_AGENT_EVAL_BUDGET=6` |
| task | `level2_63_gemm_relu_divide_fp8` (single task) |

## Environment

```bash
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export DEEPSEEK_API_KEY='<key>'
export CUTE_HARNESS_API_KEY='<key>'
export CUTE_AGENT_EVAL_BUDGET=6
```

`opencode.json` already pins the settings that produced passing kernels — do not change them:
`reasoning_effort: "none"` (with thinking on, DeepSeek spends its whole output budget reasoning and
emits nothing), `output: 32768`, provider `timeout: 900000`. The B300 harness defaults to
`http://109.236.57.62:18081`, which is ~11× faster than the older `:18080`.

Verify before launching: `python3 tools/probe-toolcalls.py` must report DeepSeek `OK`. A `FAIL`
means the endpoint returns tool calls as unparsed text and every arm will silently write nothing.

## The grid — 7 arms, one task

All arms use `level2_63_gemm_relu_divide_fp8`.

- **E1, documentation ladder** — 4 tiers (`bare`, `docs`, `examples`, `errors`), knobs off. 4 arms,
  one root `results/e1-l2-63`, launched as a single matrix (`run_iter_matrix.py` iterates `tiers:`
  internally, so one process covers all four with `--concurrency 4`).
- **Best-tier rule** — most arms solved; tie → fewer turns-to-first-pass; tie → fewer
  tokens-to-first-pass. Zero solves everywhere → use `errors` and say so.
- **E2, retrieval** — best tier with `documentation: {delivery: prompt}`. 1 arm, `results/e2-l2-63`.
  `files` is already measured by E1, so `prompt` is the only new arm.
- **E3, critic** — best tier with `feedback: {critic: true}`. 1 arm, `results/e3-l2-63`. Report
  critic tokens in their own column, never merged into author tokens.
- **E4, profiler** — best tier with `profiling: {enabled: true}`. 1 arm, `results/e4-l2-63`. State
  upfront that this acts only after a PASS, so a pass-rate null is expected; the metric is speedup
  among passers.

One protocol per results root — the runner refuses to resume a root whose protocol changed. Each
config differs from `e1-l2-63.yaml` by exactly one knob.

**E2–E4 depend on E1's best tier, so they cannot start until E1 finishes.** If you want the whole
grid in one wave instead, pin the best tier to `errors` up front and launch all four roots together
— justified by DeepSeek already passing 12/12 at `errors` on this task — and say in the report that
the tier was pinned rather than selected.

## Parallelism — and one hazard

Launch with `nohup ... &` so runs survive the terminal. Run 4–6 arms concurrently; DeepSeek is a
remote API, so arms do not contend for inference, and B300 evaluation self-serializes through
`.kernelevo/b300.lock`. **Launch every arm from this one checkout** — the lock is per-checkout, so a
second clone would overlap timed runs and corrupt every measurement.

**Cross-run isolation is fixed.** Reading `**/candidate/*.py` and `**/agent-evals/**` is denied, so
an arm cannot open another arm's kernel — every arm's candidate sits at the same path shape under
`results/`, so one readable candidate was readable in every run. Writing is unaffected (`edit` still
allows `**/kernel_evo/run/iter_*/island_*/candidate/*.py`) and each island reads its own starting
point from `baseline/submission.py`, which `TASK.md` lists. Run arms in parallel freely; just keep
them all in this one checkout for the B300 lock.

### Measured cost, and the schedule

A completed 6-turn DeepSeek arm on this exact task took **87 min wall**, of which 72 min was author
time — per-turn 11.4, 10.9, 15.0, 6.9, 15.0, 12.9 min — and it spent 12 device evaluations. Turn
length is dominated by DeepSeek's ~25 s per request across 15–60 round trips, not by the GPU.

Two schedules:

- **Sequential best-tier selection** (faithful to the protocol): E1's four tiers run as one matrix at
  `--concurrency 4`, so ~**90 min**; then E2, E3, E4 launch together, ~**90–100 min**. Total
  **≈3 hours**. E3 is the long pole — its critic adds a second DeepSeek session between turns, so
  budget it ~30% longer than the others.
- **Pinned tier, one wave**: all four roots at once, 7 arms concurrent, **≈100–120 min**.

Contention is mild either way. The arms do not share inference (DeepSeek is remote), and B300
self-serializes: 7 arms × 6 turns × up to 6 evaluations is ~250 device runs at ~2–6 s each, i.e.
10–25 min of GPU spread over the whole grid. The real risk at 7-way concurrency is DeepSeek API rate
limiting — if turns start failing with provider errors rather than kernel errors, drop to 4
concurrent. `session_error` in each turn's `usage.json` distinguishes the two.

## Metrics — every table carries all three

1. **Pass / not pass** per arm, plus turns-to-first-pass.
2. **Speedup** against the measured baseline in `<run>/b300/baseline/result.json` (passing arms
   only; give best and the `kernel_time_ms` behind it).
3. **Tokens split**: `input_tokens`, `cached_input_tokens`, `output_tokens`, `reasoning_tokens`.
   `cached_input_tokens` is structurally 0 on SGLang — report as not-reported, not as zero.

Also derive tokens-to-first-pass, and `agent_evaluations` (in-turn device runs per arm — this is
the metric that shows debugging effort, and it is recorded per turn in `usage.json`).

## Analysis

Run `analyze.py` per root, build a cross-experiment table, and classify every failed evaluation's
stderr into {import/API, compile, layout/shape, numeric mismatch, timeout, other}. `out_abs=0.000000`
in a numeric failure means the kernel launched and wrote nothing — count that separately from a
scaling error. Sanity-check that 7 arms are accounted for and token fields are non-zero everywhere.

## What is already established — do not re-derive

- DeepSeek V4 Flash passes: **level2_63 12/12 evaluations, best 0.1188 ms vs 0.3830 ms baseline =
  3.23×**; **level1_40 9/13, best 0.4412 ms vs 8.4220 ms = 19.09×**. Both verified on B300.
- **Gemma 4 31B: 0 passes** in 17 evaluations on the same documentation. Qwen3.6-35B's endpoint
  returns tool calls as unparsed text and is unusable until its server is relaunched with a parser.
- The `examples` tier contains a complete verified FP8 GEMM building block. DeepSeek's passing
  level2_63 kernel is **76% byte-identical** to it, so results at `examples`/`errors` measure
  *adaptation*; `bare`/`docs` remain the from-scratch condition. Say which is which in the report.
- Tiers are cumulative, so `examples` and `errors` both contain that example. E1 cannot separate
  "worked example helped" from "error hints helped" — if you need that, they are adjacent rungs.
- A prior 6-turn run of this task at `errors` passed 12/12 evaluations, so a null at that tier
  would indicate something broke rather than a model limit — check `session_error` first.

## Report

Fill `experiments/cute_ablation/REPORT.md`, keeping its structure and frozen-protocol section:
a 5-sentence summary of what lets an LLM agent master CuTe DSL + FP8 without retraining; E1 tier
table with tokens and the taxonomy shift; E2 delivery comparison including input tokens per solve;
E3 critic effect with separate critic-token column; E4 speedup among passers; limitations
(1 replication, 1 task, single model — the single task is the sharpest limit and should lead); and a requirement map — docs/examples→E1, retrieval→E2,
feedback→E3, profiling→E4, compilation+tests→all, FP8 harness and the four metrics shown in every
table. State the FP16→FP8 position: the torch reference → FP8 kernel is the precision transition
under study. Commit configs, REPORT.md and analysis outputs at the final SHA with pytest green.
