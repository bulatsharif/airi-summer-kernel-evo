# Which policy applies to you

`AGENTS.md` states the KernelEvo agent policy. It governs an **orchestrator** — the agent that drives
the harness, runs `kernel-evo run` / `iter` / `island`, delegates bounded work to subagents, and owns
promotion. Read it that way.

**If you are a bounded authoring agent, none of that is your job.** You do not run `kernel-evo`
commands, you do not delegate to subagents, you do not manage the archive or promotion, and you do
not orchestrate anything. Those commands are not available to you and searching for them spends your
turn for nothing.

Your contract is entirely in your own agent instruction and in the `TASK.md` you are given: implement
one kernel in one named file, using only the tools your instruction lists. Where `AGENTS.md` and your
own instruction appear to disagree, your own instruction wins.
