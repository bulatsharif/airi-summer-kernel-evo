from __future__ import annotations

import json
import sys
from pathlib import Path

from kernel_evo.cli import main


def test_nested_cli_drives_complete_single_island_barrier(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    baseline = tmp_path / "kernel.py"
    baseline.write_text("class ModelNew: ...\n", encoding="utf-8")
    harness = tmp_path / "harness.py"
    harness.write_text(
        "import json\n"
        "print(json.dumps({'compiled': 1, 'correctness': 1, 'is_valid': 1, "
        "'runtime_us': 5, 'ref_runtime_us': 10, 'speedup': 2, 'fitness': 2}))\n",
        encoding="utf-8",
    )
    runs_dir = tmp_path / "runs"
    evaluator_command = f"{sys.executable} {harness} {{candidate}}"

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kernel-evo",
            "run",
            "init",
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "cli-run",
            "--baseline",
            str(baseline),
            "--backend",
            "triton",
            "--steps",
            "1",
            "--islands",
            "1",
            "--evaluator-command",
            evaluator_command,
        ],
    )
    main()
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["phase"] == "ready"

    monkeypatch.setattr(
        sys,
        "argv",
        ["kernel-evo", "iter", "prepare", "--runs-dir", str(runs_dir), "--run-id", "cli-run"],
    )
    main()
    prepared = json.loads(capsys.readouterr().out)
    assert prepared["tasks"][0]["role"] == "kernel-author"

    monkeypatch.setattr(
        sys,
        "argv",
        ["kernel-evo", "iter", "evaluate", "--runs-dir", str(runs_dir), "--run-id", "cli-run"],
    )
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["valid_candidates"] == 1

    monkeypatch.setattr(
        sys,
        "argv",
        ["kernel-evo", "iter", "advance", "--runs-dir", str(runs_dir), "--run-id", "cli-run"],
    )
    main()
    completed = json.loads(capsys.readouterr().out)
    assert completed["phase"] == "complete"


def test_run_init_accepts_the_optional_profile_timeline_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    task = (
        Path(__file__).resolve().parents[1]
        / "tasks"
        / "cute"
        / "tasks"
        / "level1_01_square_matrix_multiplication_fp8"
    )
    runs_dir = tmp_path / "runs"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "kernel-evo",
            "run",
            "init",
            "--runs-dir",
            str(runs_dir),
            "--run-id",
            "timeline-cli",
            "--problem-path",
            str(task),
            "--backend",
            "cute",
            "--evaluator-kind",
            "cute_b300",
            "--profile",
            "--profile-timeline",
        ],
    )

    main()
    assert json.loads(capsys.readouterr().out)["phase"] == "ready"
    state = json.loads((runs_dir / "timeline-cli" / "state.json").read_text(encoding="utf-8"))
    assert state["config"]["profile_enabled"] is True
    assert state["config"]["profile_timeline"] is True
