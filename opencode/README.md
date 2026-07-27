# Headless OpenCode runner

Runs one OpenCode prompt detached from the terminal. It keeps progress, supports
timeouts, and reports final token usage. This folder also contains the
project-local `cute-fp8-kernels` skill and the instructions for the shared B300
runner.

> This directory documents the standalone/manual workflow where the agent owns
> a complete `submission.py`. Reproducible benchmark runs use
> `python -m experiment run` from the repository root. That workflow gives the
> agent a candidate-only file and keeps `main()` and validation in
> `cute_harness`; do not mix the two submission contracts in one run.

## CuTe FP8 setup

Run OpenCode with this `opencode` folder as its working directory. That lets
OpenCode discover both `AGENTS.md` and:

```text
.opencode/skills/cute-fp8-kernels/SKILL.md
```

Kernel files can live in sibling task directories. Name the target path in the
prompt. No separate local harness, specification directory, or candidate
directory is required: the prompt is the operation specification, and the task's
single `submission.py` contains the current kernel and its checks.

`AGENTS.md` is loaded automatically and tells OpenCode when to load the
`cute-fp8-kernels` skill. The task prompt does not need to repeat that
instruction. The skill body and its local references are loaded on demand rather
than added to every prompt.

`AGENTS.md` and the skill have different jobs:

- `AGENTS.md` is the small always-on policy: use CuTe DSL, keep one
  self-contained submission, validate remotely, and load the specialist skill
  for FP8 kernel work.
- `SKILL.md` is the workflow router: it selects the relevant local chapters and
  defines the correctness/performance sequence.
- `references/` is the detailed handbook. It is loaded progressively so a
  simple prompt does not pay the context cost of all chapters at once.

The local handbook contains:

| File | Purpose |
|---|---|
| `b300.md` | Target detection, SM100/SM103 compatibility, and remote runner behavior |
| `fp8.md` | E4M3/E5M2, dense versus MXFP8, scales, accumulation, and output semantics |
| `cute-dsl.md` | JIT boundaries, specialization, caching, Python subset, and DLPack |
| `layouts.md` | Shapes, strides, hierarchy, tiling, partitioning, and swizzles |
| `memory-pipelines.md` | GMEM/SMEM/TMEM, TMA, warp roles, barriers, and pipeline state |
| `examples.md` | Version-pinned dense FP8 and block-scaled implementation recipes |
| `correctness.md` | Operation contract, reference oracle, tolerances, test matrix, and gates |
| `debugging.md` | Failure diagnosis from imports and JIT through hangs and numerics |
| `performance.md` | Warmup, timing, native-path evidence, tuning order, and reporting |
| `submission.md` | One-file anatomy, framework boundary, remote submission, and exit behavior |

This makes normal work independent of web documentation. The agent should
inspect examples shipped with the installed CUTLASS package only when an exact
release-specific symbol or signature differs.

## Run

```bash
cd opencode
chmod +x opencode-headless.sh
./opencode-headless.sh "Write simple CUDA Kernel."
```

The command returns immediately and prints the run directory plus an exact
command for following the run:

```bash
./opencode-headless.sh --attach /path/to/run-directory
```

Attaching is read-only: it shows the latest output and exits when the run
finishes. Press `Ctrl+C` to stop watching; the detached task continues.

Cancel the detached task and its tool subprocesses with:

```bash
./opencode-headless.sh --cancel /path/to/run-directory
```

The run status becomes `cancelled`. Killing `--attach` alone only stops the
viewer.

With all options:

```bash
./opencode-headless.sh \
  --dir /path/to/repository \
  --timeout 30m \
  --progress ./run.jsonl \
  --run-dir ./run-state \
  --agents /path/to/AGENTS.md \
  -- "Run the tests, fix failures, and verify the result"
```

OpenCode must already be installed, configured, and connected to its model.

## CuTe harness API key

The runner inherits `CUTE_HARNESS_API_KEY` and passes it to detached OpenCode
and its shell tools without printing the value. Export it before launch:

```bash
read -r -s CUTE_HARNESS_API_KEY
export CUTE_HARNESS_API_KEY
./opencode-headless.sh --require-cute-key --timeout 30m -- "Your kernel task"
```

`--require-cute-key` fails before creating the task if the variable is absent.
Do not pass the key as a prompt or command-line option, and do not store it in
the repository.

## Example CuTe FP8 task

From this `opencode` directory:

```bash
./opencode-headless.sh \
  --dir . \
  --require-cute-key \
  --timeout 30m \
  -- "
Work on ../fp8-example/submission.py.

Implement a single-file CuTe DSL Python kernel for C = A @ B on the remote B300:
- M=N=K=1024
- A and B use E4M3FN FP8
- accumulate in FP32
- output FP16
- A is logically MxK and B is logically KxN
- use the native Blackwell FP8 MMA path

Use deterministic inputs. Compute the correctness reference from the actual
quantized inputs converted to FP32; do not compare kernel error against the
original unquantized inputs. PyTorch matmul is allowed only for the reference.
The implementation itself must use CuTe DSL, not torch.matmul, Triton, or CUDA
C++.

Keep the file self-contained with main(). Submit it through the remote GPU
service described in AGENTS.md. Establish remote correctness before measuring
performance. Warm up, report median kernel-only latency over repeated runs, and
exit nonzero on failure.

Do not stop after writing code: finish only after a successful remote run, or
report the exact blocker and remote error.
"
```

This example runs detached: the command returns immediately and prints a run
directory plus an exact `--attach` command. Add `--foreground` to keep the
current terminal attached and stream progress until OpenCode finishes. It
changes only how the shell process is run, not the prompt or agent behavior.

## Options

| Option | Description |
|---|---|
| `-d, --dir PATH` | Working directory; defaults to the current directory |
| `-t, --timeout TIME` | Whole-task limit such as `90s`, `30m`, or `2h` |
| `-o, --progress PATH` | Save raw JSONL progress; replaces the file |
| `-a, --agents PATH` | Add a specific Markdown instruction file |
| `-m, --model MODEL` | Select the OpenCode provider/model |
| `--kill-after TIME` | Grace period before force-killing; default `10s` |
| `--run-dir PATH` | Store detached state in a new or empty directory |
| `--foreground` | Run in the current terminal instead of detaching |
| `--attach RUN_DIR` | Follow a detached run |
| `--cancel RUN_DIR` | Stop a detached run and its tool subprocesses |
| `--require-cute-key` | Require inherited `CUTE_HARNESS_API_KEY` |
| `-p, --prompt TEXT` | Provide the prompt as an option |
| `-h, --help` | Show help |

## Notes

- Runs detach with `nohup`, so closing the terminal does not stop them.
- The run directory contains `output.log`, `pid`, `status`, `progress-path`,
  and—unless overridden—`progress.jsonl`. Without `--run-dir`, a unique
  temporary run directory is created and printed.
- The launch command reports whether spawning succeeded. `--attach` returns the
  finished task's exit status, including `124` for a timeout.
- Without `--agents`, OpenCode discovers `AGENTS.md` normally. An explicit file
  is added to any discovered project instructions.
- Without `--progress`, detached JSONL stays in the run directory. In
  `--foreground` mode, it is temporary and deleted afterward.
- `--timeout` covers model calls and tools. It sends `TERM`, then `KILL` after
  `--kill-after`. Exit status `124` means timed out.
- Token usage includes the main session and any subagents:

```text
Token input uncached: 13,665
Token input cached:   29,908
Token outputs:        739
Token reasoning:      0
```

Counts depend on usage metadata reported by the model server.

## Requirements

- `opencode`, Bash, `jq`, `tee`, `nohup`, `tail`, and `pgrep`
- GNU `timeout` only when using `--timeout`

On macOS:

```bash
brew install coreutils
```

Windows users: see [Windows](#windows).

## Windows

Use WSL2 with OpenCode installed and configured inside WSL:

```bash
sudo apt update
sudo apt install jq coreutils
chmod +x opencode-headless.sh
./opencode-headless.sh "Run the tests and fix failures"
```

Use WSL paths such as `/mnt/c/Users/name/project`. Ensure the model endpoint is
reachable from WSL; `127.0.0.1` may refer to WSL rather than the Windows host.
Windows `timeout.exe` is not compatible; GNU `timeout` is required.
