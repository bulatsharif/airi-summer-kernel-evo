"""Build tiny, file-oriented authoring packets for one island."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from kernel_evo.agent.models import AuthoringTask


def build_authoring_packet(
    *,
    run_id: str,
    backend: str,
    iteration: int,
    island: int,
    island_dir: Path,
    baseline_path: Path,
    candidate_path: Path,
    idea: dict[str, Any],
    feedback: Sequence[str],
    parent_profile_summary: str,
    rules: str,
    tests_summary: str,
    supplemental_context: str = "",
    supplemental_readable_files: Sequence[Path] = (),
    documentation_prompt: str = "",
    compile_check_command: str = "",
    require_graph_capturable: bool = True,
) -> AuthoringTask:
    context_dir = island_dir / "context"
    tests_dir = island_dir / "tests"
    context_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)

    idea_path = context_dir / "IDEA.md"
    feedback_path = context_dir / "FEEDBACK.md"
    rules_path = context_dir / "RULES.md"
    tests_path = tests_dir / "summary.md"
    task_path = context_dir / "TASK.md"
    supplemental_path = context_dir / "CUTE_HARNESS.md"
    parent_profile_path = context_dir / "PARENT_PROFILE.md"

    idea_path.write_text(
        f"# Seed hypothesis {idea.get('id', 'unspecified')}\n\n"
        f"{idea.get('summary', '').strip()}\n\n"
        "This is a starting hypothesis/capability contract, not a closed idea list. "
        "Use the compact parent profile to refine or replace its optimization mechanism.\n",
        encoding="utf-8",
    )
    feedback_lines = ["# Compressed feedback", ""]
    if feedback:
        feedback_lines.extend(f"- {item}" for item in feedback)
    else:
        feedback_lines.append("- No prior island results; make one conservative, measurable change.")
    feedback_path.write_text("\n".join(feedback_lines) + "\n", encoding="utf-8")
    rules_path.write_text(rules, encoding="utf-8")
    tests_path.write_text(tests_summary, encoding="utf-8")

    readable_list = [baseline_path]
    if parent_profile_summary:
        parent_profile_path.write_text(
            "# Parent profile and optimization guidance\n\n" + parent_profile_summary.strip() + "\n",
            encoding="utf-8",
        )
        readable_list.append(parent_profile_path)
    readable_list.extend([idea_path, feedback_path, rules_path, tests_path])
    if supplemental_context:
        supplemental_path.write_text(supplemental_context, encoding="utf-8")
        readable_list.append(supplemental_path)
    readable_list.extend(path for path in supplemental_readable_files if path not in readable_list)
    readable = tuple(readable_list)
    prompt_context_path: Path | None = None
    if documentation_prompt:
        # Delivered in the session prompt, so it is deliberately not readable_files:
        # the author must not spend a turn re-reading what it was already handed.
        prompt_context_path = context_dir / "DOCUMENTATION.md"
        prompt_context_path.write_text(documentation_prompt, encoding="utf-8")
    task_text = _task_markdown(
        run_id=run_id,
        backend=backend,
        iteration=iteration,
        island=island,
        candidate_path=candidate_path,
        readable=readable,
        codegen_contract=str(idea.get("codegen_contract", "")),
        requires_candidate_kernel=bool(idea.get("requires_candidate_kernel", False)),
        min_new_executors=int(idea.get("min_new_executors", 1) or 1),
        compile_check_command=compile_check_command,
        require_graph_capturable=require_graph_capturable,
        documentation_in_prompt=bool(documentation_prompt),
    )
    task_path.write_text(task_text, encoding="utf-8")

    task = AuthoringTask(
        run_id=run_id,
        backend=backend,
        iteration=iteration,
        island=island,
        task_file=task_path.resolve(),
        candidate_path=candidate_path.resolve(),
        editable_files=(candidate_path.resolve(),),
        readable_files=tuple(path.resolve() for path in readable),
        idea_id=str(idea.get("id", "unspecified")),
        idea_summary=str(idea.get("summary", "")),
        prompt_context_file=(
            prompt_context_path.resolve() if prompt_context_path else None
        ),
    )
    (context_dir / "packet.json").write_text(
        json.dumps(task.to_dict(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return task


def _task_markdown(
    *,
    run_id: str,
    backend: str,
    iteration: int,
    island: int,
    candidate_path: Path,
    readable: tuple[Path, ...],
    codegen_contract: str = "",
    requires_candidate_kernel: bool = False,
    min_new_executors: int = 1,
    compile_check_command: str = "",
    require_graph_capturable: bool = True,
    documentation_in_prompt: bool = False,
) -> str:
    readable_lines = "\n".join(f"- `{path.resolve()}`" for path in readable)
    if documentation_in_prompt:
        readable_lines += (
            "\n\nThe task statement and this run's documentation were given to you in "
            "full at the start of this session — work from what is already in front of "
            "you. Nothing else is listed here and nothing else is available to open."
        )
    navigation = next((path.resolve() for path in readable if path.name == "CUTE_HARNESS.md"), None)
    navigation_block = (
        f"""## Reading order

1. Read `{navigation}` first. It explains why each CuTe resource was selected and when the deep reference is justified.
2. Read the baseline and the one card or manifest that answers the assigned idea.
3. Open a full reference kernel only when the navigation file names the matching construction.
   Do not bulk-read every readable file.

"""
        if navigation
        else ""
    )
    preflight = (
        f"Run `kernel-evo cute lint {candidate_path.resolve()} --contract {codegen_contract}` before submission. "
        "This is a source preflight; KernelEvo still verifies the emitted artifact during evaluation."
        if backend == "cute" and codegen_contract
        else "Run the backend lint named in RULES.md before submission."
    )
    if compile_check_command:
        preflight += (
            f"\nThen run this bounded compile/execute check from the run directory:\n\n"
            f"```bash\n{compile_check_command}\n```\n"
            "Do not submit until it reports `compile_check_passed` and an executed executor. "
            "KernelEvo retains and inspects compiled artifacts during evaluation."
        )
    ownership = ""
    if backend == "cute" and requires_candidate_kernel:
        ownership = (
            "\nThis idea has a **candidate-owned kernel contract**. Add or materially modify at "
            "least one `@cute.kernel` body in the editable candidate; configuring or importing a "
            "reference kernel does not satisfy it. The owned kernel must contribute on the measured "
            f"path and evaluation must observe at least {min_new_executors} executed CuTe executor(s) "
            "in the candidate. Executor count may decrease relative to the parent when the change "
            "genuinely fuses work; do not add launches merely to increase the count. KernelEvo owns "
            "retained artifact/codegen evidence.\n"
        )
    graph_contract = (
        "Keep the measured `forward` CUDA-graph capturable: compile during "
        "initialization/preparation, keep module/function bindings static, and reuse "
        "preallocated work/output buffers. Do not compile, monkey-patch, or allocate "
        "tensors dynamically in the measured forward path."
        if require_graph_capturable
        else (
            "Optimize active device work without adding a CUDA-graph wrapper solely for "
            "the benchmark. An existing or internally used CUDA graph is allowed, but is "
            "neither required nor rewarded; fitness counts kernel and memcpy duration."
        )
    )
    return f"""# KernelEvo island authoring task

- role: `kernel-author`
- run: `{run_id}`
- backend: `{backend}`
- iteration: `{iteration}`
- island: `{island}`

## Editable file

- `{candidate_path.resolve()}`

{navigation_block}## Readable files

{readable_lines}

## Contract

Implement exactly one candidate optimization in the editable file. Preserve the interface.
Do not edit tests, baseline, state, context, or another island. Do not run the full benchmark.
{graph_contract}
Read `PARENT_PROFILE.md` before choosing the change. The seed hypothesis is not exhaustive:
you may implement a different bounded optimization when the compact per-operation breakdown
supports it, while preserving any explicit capability/codegen contract.
{preflight}
{ownership}
Return only the candidate path and a short structured rationale to the coordinator:

```json
{{
  "candidate_path": "{candidate_path.resolve()}",
  "idea_summary": "what changed",
  "expected_perf_mechanism": "why it should be faster",
  "risk": "main correctness/performance risk",
  "needs_evaluation": true
}}
```
"""
