# Headless OpenCode runner

Runs one OpenCode prompt detached from the terminal. It keeps progress, supports
timeouts, and reports final token usage.

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

## Options

| Option | Description |
|---|---|
| `-d, --dir PATH` | Working directory; defaults to the current directory |
| `-t, --timeout TIME` | Whole-task limit such as `90s`, `30m`, or `2h` |
| `-o, --progress PATH` | Save raw JSONL progress; replaces the file |
| `-a, --agents PATH` | Add a specific Markdown instruction file |
| `--kill-after TIME` | Grace period before force-killing; default `10s` |
| `--run-dir PATH` | Store detached state in a new or empty directory |
| `--foreground` | Run in the current terminal instead of detaching |
| `--attach RUN_DIR` | Follow a detached run |
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

- `opencode`, Bash, `jq`, `tee`, `nohup`, and `tail`
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
