# CuTe agent workspace rules

## Objective

Solve the selected CuTe DSL task as candidate code for the B300 harness.
Correctness comes first; performance is not scored unless the task explicitly
says otherwise.

## Information boundary

- Do not browse, search the web, fetch URLs, or request external documentation.
- Use only files already present in this repository and compiler/runtime
  feedback returned by `python -m cute_harness`.
- Never read or print `.env` files or API keys.
- Do not access paths outside this project.

## Benchmark integrity

When evaluating coding ability, do not inspect known answers:

- `cute_kernels/`
- `submissions/`
- `profiles/`
- `results/`
- `remote_agent_smoke/`

Use only the prepared workspace under `work/<task-name>/`, its `TASK.md`,
`task.json`, and `submission.py`.

The candidate does not contain numerical validation. Do not define `main()`,
generate reference outputs, or print a fake PASS marker. The harness owns those
parts and appends them only when constructing the uploaded file.

## Allowed workflow

1. Read the prepared `TASK.md`, public `task.json`, and `submission.py`.
2. Edit only that prepared `submission.py` unless the user explicitly expands
   the scope.
3. Run the local compatibility check:

   ```text
   python -m cute_harness check <task-id> work/<task-name>/submission.py
   ```

4. If it passes, run the B300 evaluator:

   ```text
   python -m cute_harness run <task-id> work/<task-name>/submission.py
   ```

5. Use returned compiler/runtime/numerical diagnostics for the next attempt.
6. Stop when the task passes or the user-provided attempt budget is exhausted.

## Submission restrictions

- Candidate output must be written by CuTe GPU kernels.
- PyTorch is allowed only for input generation and post-kernel reference checks.
- Keep candidate code self-contained; do not import project-local code.
- Use the exact FP8/FP4 storage, accumulator, output, scale, shape, and layout
  contract from the task.
- Do not define or call `main()`; the harness owns the evaluator entrypoint.
- Do not use `if __name__ == "__main__"` or dunder attribute access.
- Do not use filesystem, network, subprocess, package installation, or dynamic
  execution primitives in submissions.

## Safety

- Do not run destructive Git commands.
- Do not push, publish, deploy, or modify remote services.
- The SSH tunnel and environment variables are owned by the user outside the
  agent session.

Report the exact attempt count, final validation metrics, device time,
profile ID, and submission path.
