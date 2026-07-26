# Headless OpenCode runner

Runs one OpenCode prompt without its interactive terminal interface (TUI),
streams progress, reports token usage, and exits.

## Run

```bash
chmod +x opencode-headless.sh
./opencode-headless.sh "Write simple CUDA Kernel."
```

With all options:

```bash
./opencode-headless.sh \
  --dir /path/to/repository \
  --timeout 30m \
  --progress ./run.jsonl \
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
| `-p, --prompt TEXT` | Provide the prompt as an option |
| `-h, --help` | Show help |

## Notes

- Without `--agents`, OpenCode discovers `AGENTS.md` normally. An explicit file
  is added to any discovered project instructions.
- Without `--progress`, the temporary JSONL log is deleted after the run.
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

- `opencode`, Bash, `jq`, and `tee`
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
