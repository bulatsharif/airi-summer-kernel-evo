# Web vs local-handbook token benchmark

This experiment compares two runs of the same model on the same prepared CuTe
task:

- `web`: web search/fetch is enabled; the local CuTe skill and repository docs
  are denied.
- `local`: web access is denied; the project-local `cute-fp8-kernels` skill is
  enabled.

Both arms use the same task starter, model, timeout, local policy check, and
one external B300 evaluation. The B300 key is removed from the agent process,
so an agent cannot create an uncontrolled remote retry loop.

The launcher builds a standalone OpenCode config and disables normal project
config discovery for the agent process. This prevents the repository's root
`AGENTS.md` or `opencode.json` permissions from being silently appended to one
arm after isolation rules are applied.

## Cases

The initial suite uses two existing KernelBench Level 1 adaptations:

- `level1_01_square_matrix_multiplication_fp8`
- `level1_40_layer_norm_fp8`

The allowlist and recommended repetition count live in `cases.json`. Run at
least three trials per arm and alternate arm order between trials to reduce
warm-cache and server-load bias.

## Run one trial

Run from WSL with OpenCode, `jq`, and GNU `timeout` installed. The Qwen provider
environment must already be configured. The agent does not receive
`CUTE_HARNESS_API_KEY`.

```bash
./experiments/token_ab/run-trial.sh \
  web \
  level1_01_square_matrix_multiplication_fp8 \
  001 \
  30m
```

The command prepares a fresh ignored workspace under `work/`, starts the
existing detached OpenCode runner, and stores its state under `ab_runs/`.
Attach with the command printed by the runner.

After the agent reaches a terminal state, evaluate its immutable candidate
exactly once. Set the B300 key only in this evaluator shell:

```bash
export CUTE_HARNESS_API_KEY='<key>'
./experiments/token_ab/evaluate-trial.sh \
  ab_runs/level1_01_square_matrix_multiplication_fp8/web/001
```

Summarize all completed trials as CSV:

```bash
python experiments/token_ab/summarize.py ab_runs
```

Use `--json` for machine-readable JSON.

## Comparison rule

Correctness dominates token count. Compare arms in this order:

1. success rate under the one-shot evaluator;
2. median logical tokens among successful trials;
3. uncached input, cached input, output, and reasoning tokens separately;
4. local-check status, OpenCode status, and wall-clock/runtime evidence.

`logical_tokens` is the sum of uncached input, cached input, output, and
reasoning tokens. Cached input remains in the total because this benchmark asks
how much context the agent consumed, not provider billing cost. Keep the raw
components because cache pricing and prefill cost can be analyzed separately.

Do not compare a failed low-token run as better than a successful run. Report
failed trials as censored at their observed token budget.
