import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cute_harness.assembly import (
    EVALUATOR_MARKER,
    assemble_submission,
    candidate_starter,
)
from cute_harness.client import build_multipart
from cute_harness.cli import _safe_print, main
from cute_harness.policy import check_submission
from cute_harness.tasks import discover_tasks


class TaskManifestTests(unittest.TestCase):
    def test_three_tasks_are_discoverable(self):
        tasks = discover_tasks()
        self.assertEqual(
            set(tasks),
            {
                "level1_01_square_matrix_multiplication_fp8",
                "level1_40_layer_norm_fp8",
                "level2_76_gemm_add_relu_fp8",
            },
        )

    def test_all_known_baselines_pass_policy(self):
        for task in discover_tasks().values():
            with self.subTest(task=task.id):
                report = check_submission(
                    task,
                    task.baseline_path,
                    candidate_mode=False,
                )
                self.assertTrue(report.passed, report.errors)

    def test_starters_are_incomplete_but_syntactically_valid(self):
        for task in discover_tasks().values():
            with self.subTest(task=task.id):
                with tempfile.TemporaryDirectory() as temp_dir:
                    candidate = Path(temp_dir) / "submission.py"
                    candidate.write_text(
                        candidate_starter(task),
                        encoding="utf-8",
                    )
                    report = check_submission(task, candidate)
                self.assertFalse(report.passed)
                self.assertTrue(
                    any(
                        "required CuTe call not found" in error
                        for error in report.errors
                    )
                )


class CliTests(unittest.TestCase):
    def test_prepare_excludes_baseline_path(self):
        task_id = "level1_01_square_matrix_multiplication_fp8"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "agent-work"
            code = main(["prepare", task_id, "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue((output / "TASK.md").is_file())
            self.assertTrue((output / "submission.py").is_file())
            public = json.loads((output / "task.json").read_text("utf-8"))
            self.assertNotIn("baseline", public)
            self.assertEqual(public["starter"], "submission.py")
            candidate = (output / "submission.py").read_text("utf-8")
            self.assertNotIn(EVALUATOR_MARKER, candidate)
            self.assertNotIn("def main(", candidate)
            self.assertNotIn(" PASS", candidate)

    def test_assembly_restores_owned_evaluator(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(candidate_starter(task), encoding="utf-8")
            assembled = assemble_submission(task, candidate)
        self.assertIn(EVALUATOR_MARKER, assembled)
        self.assertIn("def main(", assembled)
        self.assertIn(" PASS", assembled)

    def test_run_uses_one_immutable_candidate_snapshot(self):
        task_id = "level1_01_square_matrix_multiplication_fp8"
        task = discover_tasks()[task_id]
        original_source = candidate_starter(task) + "\n# original candidate\n"
        replacement_source = candidate_starter(task) + "\n# changed candidate\n"

        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(original_source, encoding="utf-8")

            def inspect_run(
                _task,
                candidate_snapshot,
                assembled_submission,
                *_args,
                **_kwargs,
            ):
                candidate.write_text(replacement_source, encoding="utf-8")
                self.assertNotEqual(candidate_snapshot.resolve(), candidate.resolve())
                self.assertEqual(
                    candidate_snapshot.read_text(encoding="utf-8"),
                    original_source,
                )
                assembled = assembled_submission.read_text(encoding="utf-8")
                self.assertIn("# original candidate", assembled)
                self.assertNotIn("# changed candidate", assembled)
                return True, {}

            with patch.dict(
                os.environ,
                {"CUTE_HARNESS_API_KEY": "test-only-placeholder"},
            ), patch("cute_harness.cli._run_one", side_effect=inspect_run):
                code = main(
                    [
                        "run",
                        task_id,
                        str(candidate),
                        "--output",
                        str(Path(temp_dir) / "artifacts"),
                    ]
                )

        self.assertEqual(code, 0)

    def test_safe_print_handles_cp1251_incompatible_diagnostics(self):
        raw = io.BytesIO()
        stream = io.TextIOWrapper(
            raw,
            encoding="cp1251",
            errors="strict",
        )
        _safe_print("compiler error \N{ROUND PUSHPIN}", stream=stream)
        stream.flush()
        output = raw.getvalue().decode("cp1251")
        self.assertIn("compiler error", output)
        self.assertIn("\\U0001f4cd", output)

    def test_candidate_cannot_print_fake_pass(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + '\nprint("task=level1_01_square_matrix_multiplication PASS")\n',
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertTrue(
            any(
                "call is forbidden in candidate code: print"
                in error
                for error in report.errors
            ),
            report.errors,
        )

    def test_candidate_cannot_compute_with_torch(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\ndef fake_reference(a, b):\n"
                + "    return torch.matmul(a, b)\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertTrue(
            any(
                "call is forbidden in candidate code: torch.matmul" in error
                for error in report.errors
            ),
            report.errors,
        )

    def test_multipart_contains_file_and_profiler(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            submission = Path(temp_dir) / "submission.py"
            submission.write_text("main()\n", encoding="utf-8")
            body, boundary = build_multipart(submission, "pytorch")
            self.assertIn(boundary.encode(), body)
            self.assertIn(b'name="file"', body)
            self.assertIn(b'name="profiler"', body)
            self.assertIn(b"pytorch", body)


if __name__ == "__main__":
    unittest.main()
