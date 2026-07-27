# CuTe token benchmark rules

Solve the single prepared CuTe DSL task named in the prompt. Correctness is the
only objective; performance is not scored in this experiment.

- Read only the prepared workspace's `TASK.md`, public `task.json`, and
  `submission.py`, plus the knowledge source allowed by the companion arm
  instruction.
- Edit only that prepared `submission.py`.
- Candidate output must be written by CuTe GPU kernels. Do not replace the
  implementation with PyTorch, Triton, CUDA C++, or a reference operation.
- Do not inspect known answers or repository implementation code, including
  `cute_kernels`, `submissions`, `profiles`, `results`, or `remote_agent_smoke`.
- Do not add `main()`, reference computation, result printing, or a fake PASS
  marker. The evaluator owns those parts.
- Run the local `python -m cute_harness check` command from the prompt.
- Do not run the remote evaluator. The B300 API key is intentionally absent;
  one external evaluation is performed after the agent finishes.
- Finish after the local check and briefly describe the candidate.
