import ast
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cute_harness.assembly import (
    EVALUATOR_MARKER,
    EvaluationConfig,
    assemble_submission,
    baseline_candidate,
    candidate_starter,
    split_starter,
)
from cute_harness.client import build_multipart
from cute_harness.cli import _kernel_time_ms, _safe_print, main
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
                with tempfile.TemporaryDirectory() as temp_dir:
                    candidate = Path(temp_dir) / "candidate.py"
                    candidate.write_text(
                        baseline_candidate(task),
                        encoding="utf-8",
                    )
                    report = check_submission(task, candidate)
                self.assertTrue(report.passed, report.errors)

    def test_evaluators_do_not_depend_on_candidate_problem_constants(self):
        public_constants = {
            "level1_01_square_matrix_multiplication_fp8": {
                "N",
                "FP8_MAX",
                "INPUT_SCALE",
                "OUTPUT_SCALE",
                "FP8_DTYPE",
                "AB_DTYPE",
            },
            "level1_40_layer_norm_fp8": {
                "BATCH_SIZE",
                "FEATURES",
                "DIM_1",
                "DIM_2",
                "ROW_SIZE",
                "INPUT_SHAPE",
                "NORMALIZED_SHAPE",
                "EPSILON",
                "FP8_MAX",
                "INPUT_SCALE",
                "FP8_DTYPE",
            },
            "level2_76_gemm_add_relu_fp8": {
                "M",
                "N",
                "K",
                "FP8_MAX",
                "WEIGHT_BOUND",
                "SCALE_A",
                "SCALE_B",
                "FP8_DTYPE",
                "AB_DTYPE",
            },
        }
        for task in discover_tasks().values():
            with self.subTest(task=task.id):
                _, evaluator = split_starter(task)
                names = {
                    node.id
                    for node in ast.walk(ast.parse(evaluator))
                    if isinstance(node, ast.Name)
                }
                self.assertFalse(names & public_constants[task.id])

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
            assembled = assemble_submission(
                task,
                candidate,
                EvaluationConfig(seed=7, warmup=3, repeats=9),
            )
        self.assertIn(EVALUATOR_MARKER, assembled)
        self.assertIn("def main(", assembled)
        self.assertIn(" PASS", assembled)
        self.assertIn("_CUTE_HARNESS_SEED = 7", assembled)
        self.assertIn("_CUTE_HARNESS_WARMUP = 3", assembled)
        self.assertIn("_CUTE_HARNESS_REPEATS = 9", assembled)

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

    def test_kernel_time_is_parsed_from_evaluator_stdout(self):
        self.assertEqual(
            _kernel_time_ms(
                {
                    "stdout": (
                        "task=example kernel_time_ms=1.250000 PASS\n"
                    )
                }
            ),
            1.25,
        )

    def test_baseline_uses_shared_evaluator_and_records_kernel_time(self):
        task_id = "level1_01_square_matrix_multiplication_fp8"
        response = {
            "success": True,
            "exit_code": 0,
            "stdout": (
                "task=level1_01_square_matrix_multiplication "
                "kernel_time_ms=0.750000 PASS\n"
            ),
            "stderr": "",
            "device_time_ms": 10.0,
            "profile_id": "profile-test",
            "timed_out": False,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "baseline"
            with patch.dict(
                os.environ,
                {"CUTE_HARNESS_API_KEY": "test-only-placeholder"},
            ), patch(
                "cute_harness.cli.HarnessClient.run_file",
                return_value=response,
            ), patch(
                "cute_harness.cli.HarnessClient.download_profile",
                return_value=b"{}",
            ):
                code = main(
                    [
                        "run",
                        task_id,
                        "--baseline",
                        "--seed",
                        "0",
                        "--warmup",
                        "2",
                        "--repeats",
                        "5",
                        "--output",
                        str(output),
                    ]
                )
            record = json.loads(
                (output / "result.json").read_text(encoding="utf-8")
            )
            assembled = (output / "submission.py").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(record["candidate_kind"], "baseline")
        self.assertEqual(record["benchmark"]["seed"], 0)
        self.assertEqual(record["benchmark"]["kernel_time_ms"], 0.75)
        self.assertIn("_CUTE_HARNESS_SEED = 0", assembled)
        self.assertIn("kernel_time_ms=", assembled)


if __name__ == "__main__":
    unittest.main()
