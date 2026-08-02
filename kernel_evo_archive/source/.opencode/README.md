# KernelEvo with OpenCode

OpenCode discovers the canonical KernelEvo skill from `.agents/skills/kernelevo` and the repository rules from
`AGENTS.md`. This directory adds three bounded OpenCode subagents:

- `kernelevo-island-author`
- `kernelevo-profile-reviewer`
- `kernelevo-repair-author`

Their definitions intentionally omit `model`. OpenCode therefore runs each subagent with the model selected by
the primary session, including a model from a custom or local provider. They use `mode: all`: the normal
KernelEvo coordinator invokes them as subagents through the Task tool, while scripts and smoke tests can select
one directly with `--agent` and pass an exact generated packet path.

Write permissions are also structurally bounded: author and repair agents may edit only generated
`iter_*/island_*/candidate/*.py` paths, while the reviewer may write only
`iter_*/island_*/context/PROFILE_REVIEW.json`.

Start OpenCode in the repository and ask it to use the `kernelevo` skill:

```bash
opencode
```

For a non-interactive run:

```bash
opencode run --model provider/model \
  "Use the kernelevo skill to optimize the program configured in examples/agent/evo.yaml."
```

For a local OpenAI-compatible model, copy and adjust `examples/agent/opencode-ollama.json`, or use it directly
with `OPENCODE_CONFIG`. The model must support tool calling and should have at least a 16K context window.

```bash
OPENCODE_CONFIG=examples/agent/opencode-ollama.json \
  opencode run --model ollama/gpt-oss:20b \
  "Use the kernelevo skill with my program, settings, and iteration count."
```

To process an already prepared packet directly:

```bash
OPENCODE_CONFIG=examples/agent/opencode-ollama.json \
  opencode run --model ollama/gpt-oss:20b \
  --agent kernelevo-island-author \
  "Process only /absolute/run/path/context/TASK.md."
```

KernelEvo remains responsible for scheduling, evaluation, profiling, archive updates, and reporting. OpenCode
only coordinates the barrier loop and invokes bounded author, reviewer, or repair subagents.
