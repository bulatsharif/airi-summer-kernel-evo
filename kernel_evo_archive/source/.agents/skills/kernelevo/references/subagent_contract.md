# Bounded author contract

Give a kernel author one generated `TASK.md` path and no cross-island context. The packet identifies:

- one editable candidate;
- one read-only baseline;
- one idea card;
- at most five compressed feedback bullets;
- concise backend rules and a test summary.

The author must make one optimization, preserve the interface, avoid full benchmarks, and return:

```json
{
  "candidate_path": "/absolute/island/path/candidate/kernel.py",
  "idea_summary": "what changed",
  "expected_perf_mechanism": "why it should be faster",
  "risk": "main risk",
  "needs_evaluation": true
}
```

Do not let an author inspect state, archives, raw profiler logs, tests, or other islands. Do not start an
open-ended debug loop. For a repair, provide only the failed candidate, compact error, original idea, editable
path, and one-turn instruction to fix the localized failure without redesigning the branch.

# Bounded profile reviewer contract

Give a profile reviewer one generated `PROFILE_REVIEW.md`. The reviewer may read only that task and its listed
candidate source, may edit only `PROFILE_REVIEW.json`, and must not open raw traces, profiler/evaluator logs,
state, archives, tests, or another island. It returns causal findings plus a bounded list of new optimization
hypotheses. It never edits kernels, benchmarks candidates, promotes entries, or performs a repair.
