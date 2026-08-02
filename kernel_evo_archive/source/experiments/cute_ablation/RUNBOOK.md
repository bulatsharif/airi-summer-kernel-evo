# Runbook — running a CuTe ablation grid

Everything here was learned by getting it wrong first. The four sections marked
**MUST** are not style preferences; skipping any one of them silently produces
numbers that look fine and are not.

## 0. Environment

```bash
export DEEPSEEK_BASE_URL='https://api.deepseek.com'
export DEEPSEEK_API_KEY='...'
export GEMMA_BASE_URL='http://<host>:30001/v1'   GEMMA_API_KEY='...'
export QWEN_BASE_URL='http://<host>:18001/v1'    QWEN_API_KEY='...'
export CUTE_HARNESS_API_KEY='...'
export CUTE_AGENT_EVAL_BUDGET=6
```

Verify every model endpoint parses tool calls before launching anything. A
`FAIL` means the server returns tool calls as unparsed text and **every arm
silently writes nothing**:

```bash
python3 tools/probe-toolcalls.py                      # QWEN_/GEMMA_ from env
GEMMA_API_KEY="$DEEPSEEK_API_KEY" python3 tools/probe-toolcalls.py \
    'https://api.deepseek.com' 'deepseek-v4-flash'    # custom endpoint form
```

Check `opencode.json` provider `timeout` is the same for every provider. It was
once 180000 for Qwen/Gemma against 900000 for DeepSeek, which made the model
comparison measure a 5x handicap: 5 of 6 Qwen turns wrote no candidate at all.

## 1. MUST — isolate every arm in its own git worktree

Arms in a shared checkout read each other's kernels. This is not hypothetical:
a `bare` arm read the `examples` arm's promoted kernel from
`<run>/archive/iter-006-island-0/submission.py` and emitted it back — 329 lines,
identical SHA-256 — scoring a fake turn-1 solve at 3.30x.

**Permission globs cannot fix it.** All measured, not assumed:

| control | result |
| --- | --- |
| deny `**/candidate/*.py`, `**/agent-evals/**` | holds, but misses `archive/` |
| `external_directory: deny` | does **not** block absolute paths |
| read-deny globs generally | bind **relative to the session dir** only |
| `read` on a directory | returns a listing — `ls` is not the only way to enumerate |

Worktrees work because `results/` is untracked, so a fresh worktree has no
sibling data on disk at all. Also required:

```bash
chmod 0311 <worktree-parent>          # driver can create, agent cannot list
chmod 0000 results archive_runs diagnostics .kernelevo \
          src tasks tests examples build experiments/cute_ablation/results
```

`src/` matters: it holds `cute_harness/examples/hopper_wgmma_gemm/kernel.py`, a
complete verified FP8 GEMM. Readable through the primary checkout's absolute
path, it hands `bare` and `docs` the answer.

Move `b300.lock` out of `.kernelevo` before locking it:

```bash
mkdir -p .b300lock && touch .b300lock/b300.lock && chmod 0755 .b300lock
```

**Verify by probe, never by reasoning.** Run one throwaway session that attempts
the forbidden reads and reports SUCCEEDED/DENIED:

```bash
opencode run --format json --dir "$WT" --agent cute-fp8-author --model "$MODEL" \
"Do not write code. Report SUCCEEDED or DENIED for each:
P1: read $MAIN/src/kernel_evo/cute_harness/examples/hopper_wgmma_gemm/kernel.py
P2: read $MAIN/results/<some-run>/.../archive/iter-006-island-0/submission.py
P3: read the directory <worktree-parent>
P4: read $WT/AGENTS.md
Output P1=..., P2=..., P3=..., P4=... then stop."
```

Expect P1–P3 DENIED, P4 SUCCEEDED. Keep probes to ~4 items; longer ones run past
the timeout and flush nothing.

## 2. MUST — never trust a harness PASS on its own

The harness scores a kernel that **never launches** as PASS. `output` is
`torch.empty` — uninitialized, never poisoned — and correctness is judged only
by `(output - reference).abs().max()`. The allocator recycles a block that
already holds the right values, so a no-op inherits a correct buffer and the
timed region is empty.

Seen on 3 models and 2 tasks. Worst case observed: an arm scored 4 PASSes while
launching zero kernels, and its author time *fell* 5x across those turns — the
agent converged on doing nothing because doing nothing scored better than trying.

A pass is only real if the device profile contains a candidate kernel:

```python
kernels = result["profile_summary"]["top_kernels"]
real = any(k["name"].startswith("kernel_cutlass_") and "convert" not in k["name"]
           for k in kernels)
```

`cross_experiment.py` enforces this; `analyze.py` does **not** and will report
the inflated number. Sanity check: compute achieved FLOP/s. 137 GFLOP in 12.8us
is 10.7 PFLOP/s — above the device's dense FP8 peak, so it never happened.

The fix, when someone is free to make it (changes the harness, so do it between
grids and re-validate): `torch.empty(...).fill_(float("nan"))` in the task
starter, plus a `minimum_launched_kernels` policy check.

## 3. MUST — size concurrency against the device, not the API

Inference parallelism is nearly free; **arms are not**. Each arm is a GPU
reservation. Every arm serializes on `b300.lock`, and in-turn evaluations block
*inside* the 900s author budget, so over-subscription doesn't slow things down —
it silently truncates turns mid-work.

Measured on `level1_02`/`level1_40`: harness call 9.5–13s, `evaluation_seconds`
80–140s, i.e. ~85% queueing, ~67% device duty at 10 arms, and 57% of author
sessions killed at the cap. On `level2_63` a call is ~2s and the same
concurrency is fine. **Evaluation cost is task-dependent — measure it per task
before choosing concurrency.**

Rule of thumb: ~10 arms per evaluator server. Watch `timed_out` per arm; if it
exceeds ~30%, reduce concurrency or add a server.

## 4. MUST — one task per evaluator server

Two B300s measured a 26% gap on an identical kernel. Absolute `kernel_time_ms`
cannot be pooled across servers. Speedup ratios survive, since each arm times
its own baseline on its own server — but pin a whole task to one server so
within-task absolute times stay comparable. One lock file per server.

## 5. Launch

Write a plan file, then launch detached:

```json
{
  "parent": "/abs/path/arm-worktrees",
  "commit": "HEAD",
  "eval_budget": 6,
  "servers": {"A": ["http://<hostA>:18081", "b300.lock"],
              "B": ["http://<hostB>:18080", "b300-b.lock"]},
  "caps": {"A": 7, "B": 7},
  "arms": [
    {"tag": "ds-e1-l2-63-bare", "server": "A", "model": "deepseek/deepseek-v4-flash",
     "root": "results/ds-e1-l2-63", "config": "experiments/cute_ablation/e1-l2-63.yaml",
     "tier": "bare"}
  ]
}
```

One arm per E1 tier (`--tier`), one arm each for E2/E3/E4. Commit configs first
— worktrees are created from a commit, so uncommitted files are invisible, and
committing mid-run splits the grid across two different harnesses.

```bash
nohup python3 experiments/cute_ablation/run_grid.py \
      --plan plan.json --logs logs/grid > logs/grid/driver.log 2>&1 &
```

## 6. Watch it

Do not poll in a loop. Tail the driver log for completions and failures:

```bash
tail -f logs/grid/driver.log | grep -E "done |FAILED |GRID COMPLETE|Traceback"
```

Arm-level progress (worktree parent is unlistable, so enumerate via git):

```bash
git worktree list --porcelain | grep '^worktree .*arm-worktrees' | cut -d' ' -f2 |
while read -r w; do
  n=$(ls "$w"/results/*/iter/*/*/r01/samples/*.json 2>/dev/null | wc -l)
  echo "$(basename "$w") $n/6"
done
```

Run a **leak tripwire** for the whole run — grep each session's
`progress.jsonl` for sibling worktree names and quarantined paths
(`hopper_wgmma_gemm`, `kernel-evo/results`, `archive_runs`). It caught two
breaches that the permission changes had not closed.

Health signals worth checking per arm, all in `samples/iteration-*.json`:

| field | meaning |
| --- | --- |
| `wrote_candidate: false` | scaffold failure, not a kernel failure |
| `session_error` | provider problem (timeout, rate limit) |
| `timed_out: true` | killed at the author cap; >30% means over-subscribed |
| `agent_evaluations` | debugging effort actually spent |

## 7. Analyze

```bash
python3 experiments/cute_ablation/cross_experiment.py \
        E1:results/ds-e1-l2-63 E2:results/ds-e2-l2-63 --output out.json
python3 experiments/cute_ablation/classify_failures.py results/*-l2-63 --output tax.json
PYTHONPATH=src uv run python experiments/cute_ablation/analyze.py results/ds-e1-l2-63
```

`cross_experiment.py` is the authority — it applies the launched-kernel check
and reports spurious passes separately. `analyze.py` reports the raw harness
verdict and will disagree; that disagreement is expected and should be stated
wherever both appear.

Report `cached_input_tokens` as **not reported** when zero: it is structurally 0
on SGLang (Gemma/Qwen) and genuinely populated on the DeepSeek API.

## 8. Merge and clean up

Worktrees hold the only copy until merged. With **no agents running**, restore
permissions, copy each arm's results into the primary checkout, then prune:

```bash
chmod 0755 results src tests tasks experiments/cute_ablation/results ...
# copy <worktree>/results/<root>/iter/... into results/<root>/
git worktree list --porcelain | grep '^worktree .*arm-worktrees' | cut -d' ' -f2 |
  xargs -n1 git worktree remove --force
git worktree prune
```

Prefer re-run arms (`-rr` tag) over their originals, and record which was used.

## Known gotchas

- zsh errors on unmatched globs and aborts the script — `setopt NULL_GLOB` in
  any watcher loop.
- `pgrep -f "pattern"` matches the watcher's own command line. Use a bracket
  (`run_iter_matri[x].py`) or track PIDs explicitly.
- macOS `stat` needs `-L` to follow a symlink; without it you compare the
  symlink's inode, not the lock's.
- Confirm the shared lock actually serializes: open both paths and `flock` the
  second with `LOCK_NB`; it must block.
- Session budget: `session_timeout_seconds` is a guardrail, not a declared
  protocol variable. On expensive-evaluation tasks it is consumed by queueing,
  which prevents the declared 6 in-turn evaluations from ever being spent.
