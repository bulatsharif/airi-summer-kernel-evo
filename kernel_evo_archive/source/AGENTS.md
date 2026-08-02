# KernelEvo agent policy

When optimizing kernels with KernelEvo:

- Use the `kernel-evo run`, `iter`, and `island` commands; do not manually manage the evolution algorithm.
- KernelEvo owns scheduling, archive state, evaluation, profiling, promotion, and reporting.
- Use subagents only for bounded candidate authoring, compact profile review, or a specifically requested bounded repair.
- Profile reviewers may read only the generated compact profile packet and that island's candidate source;
  they write only the generated review JSON and must not inspect raw evaluator/profiler logs.
- Treat each island packet as isolated. Never inspect or edit another island from an authoring task.
- Never promote a candidate without KernelEvo correctness and benchmark results.
- Do not read raw evaluator/profiler logs unless KernelEvo reports that debug input is required.
