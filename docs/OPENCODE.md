# OpenCode + Qwen setup

The project config is [`opencode.json`](../opencode.json). It uses the local
OpenAI-compatible endpoint through an SSH tunnel and reads both API keys only
from environment variables.

## Desktop: one-window launch

Fully quit an existing OpenCode Desktop process, then open one PowerShell:

```powershell
Set-Location "C:\Users\Timur\Documents\CuTe project"
.\scripts\start-opencode-ui.ps1
```

Alternatively, double-click `START_OPENCODE.cmd` in the project root.

The launcher:

- loads the local credentials from the Git-ignored `.opencode.local.ps1`;
- leaves the SSH tunnel running in the background;
- applies the Desktop `@opencode-ai/plugin@local` workaround;
- starts the installed OpenCode Desktop application.

Open `C:\Users\Timur\Documents\CuTe project` in the UI. The project-level
`opencode.json` is then loaded automatically.

## CLI alternative

### 1. Install OpenCode

The stable CLI package is:

```powershell
npm install -g opencode-ai
```

The helper can also run it through `npx` without a global installation.

### 2. Start the SSH tunnel

Open terminal 1:

```powershell
Set-Location "C:\Users\Timur\Documents\CuTe project"
.\scripts\start-qwen-tunnel.ps1
```

Enter the SSH password interactively and keep this terminal open.

### 3. Set environment variables

Open terminal 2:

```powershell
Set-Location "C:\Users\Timur\Documents\CuTe project"

$env:QWEN_BASE_URL = "http://127.0.0.1:18001/v1"
$env:QWEN_API_KEY = "<qwen-server-key>"
$env:CUTE_HARNESS_URL = "http://109.236.57.62:18080"
$env:CUTE_HARNESS_API_KEY = "<b300-harness-key>"
```

Do not put real keys in `.env.example`, `opencode.json`, prompts, or result
artifacts.

### 4. Health-check

```powershell
.\scripts\test-qwen-endpoint.ps1
python -m cute_harness doctor --require-key
```

Expected:

```text
qwen_endpoint=PASS model=qwen3.6-35b-a3b response=OPENCODE_QWEN_OK
```

### 5. Prepare an agent task

The current evaluator-owned workspace is prepared at
`work/opencode-square-v1`.
To create another clean attempt, use a new output directory:

```powershell
python -m cute_harness prepare `
  level1_01_square_matrix_multiplication_fp8 `
  --output work/opencode-square-2
```

OpenCode is instructed by `AGENTS.md` not to read known baselines or use the
web.

### 6. Start OpenCode

Interactive TUI:

```powershell
.\scripts\start-opencode.ps1
```

Or one-shot:

```powershell
.\scripts\start-opencode.ps1 run --auto @"
Read work/opencode-square-v1/TASK.md and work/opencode-square-v1/task.json.
Edit only work/opencode-square-v1/submission.py.
Use only repository-local documentation.
Run cute_harness check, then the B300 evaluator.
Stop after PASS or two remote attempts.
"@
```

The config:

- selects only `qwen-server/qwen3.6-35b-a3b`;
- supplies `baseURL` and `apiKey` via environment variables;
- loads `AGENTS.md` explicitly;
- denies web search/fetch and external directories;
- allows edits only to `work/*/submission.py`;
- allows shell execution only through `python -m cute_harness`;
- denies general file search and arbitrary shell/network commands.
