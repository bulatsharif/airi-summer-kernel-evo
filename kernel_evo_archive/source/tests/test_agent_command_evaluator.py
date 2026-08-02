from __future__ import annotations

import sys
from pathlib import Path

from kernel_evo.agent import KernelEvoAgent


def test_external_command_evaluator_supports_candidate_placeholder(tmp_path: Path) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    harness = tmp_path / "harness.py"
    harness.write_text(
        "import json, pathlib, sys\n"
        "source = pathlib.Path(sys.argv[1]).read_text()\n"
        "assert 'ModelNew' in source\n"
        "print(json.dumps({'compiled': 1, 'correctness': 1, 'is_valid': 1, "
        "'runtime_us': 4.0, 'ref_runtime_us': 8.0, 'speedup': 2.0, 'fitness': 2.0}))\n",
        encoding="utf-8",
    )
    controller = KernelEvoAgent(tmp_path / "runs")
    controller.init_run(
        {
            "baseline": str(baseline),
            "backend": "triton",
            "islands": 1,
            "evaluation": {"command": [sys.executable, str(harness), "{candidate}"]},
        },
        run_id="command-eval",
    )
    controller.prepare_iteration("command-eval")
    report = controller.evaluate_iteration("command-eval")

    assert report["valid_candidates"] == 1
    assert report["islands"][0]["speedup"] == 2.0
