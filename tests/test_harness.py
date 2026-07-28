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
from cute_harness.tasks import REPO_ROOT, discover_tasks


class TaskManifestTests(unittest.TestCase):
    def test_nine_tasks_are_discoverable(self):
        tasks = discover_tasks()
        self.assertEqual(
            set(tasks),
            {
                "level1_01_square_matrix_multiplication_fp8",
                "level1_40_layer_norm_fp8",
                "level1_72_conv_transpose3d_fp8",
                "level2_09_matmul_subtract_multiply_relu_fp8",
                "level2_12_gemm_multiply_leaky_relu_fp8",
                "level2_14_gemm_divide_sum_scaling_fp8",
                "level2_40_matmul_scaling_residual_add_fp8",
                "level2_63_gemm_relu_divide_fp8",
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

    def test_all_tasks_declare_public_context(self):
        for task in discover_tasks().values():
            with self.subTest(task=task.id):
                self.assertGreaterEqual(len(task.reference_paths), 1)
                for path in task.reference_paths:
                    self.assertTrue(path.is_file())
                    self.assertNotIn("cute_kernels", path.parts)
                    self.assertNotIn("runs", path.parts)
                self.assertEqual(
                    [path.name for path in task.agent_skill_paths],
                    ["cute-fp8-kernels"],
                )
                for path in task.agent_skill_paths:
                    self.assertTrue((path / "SKILL.md").is_file())
                    self.assertTrue((path / "references").is_dir())

    def test_dense_gemm_skill_pins_candidate_api_signatures(self):
        skill = (
            REPO_ROOT
            / "opencode"
            / ".opencode"
            / "skills"
            / "cute-fp8-kernels"
        )
        reference = (
            skill / "references" / "candidate-gemm-api.md"
        ).read_text(encoding="utf-8")
        patterns = (
            skill / "references" / "candidate-kernel-patterns.md"
        ).read_text(encoding="utf-8")
        skill_body = (skill / "SKILL.md").read_text(encoding="utf-8")

        for snippet in (
            "utils.LayoutEnum.from_tensor",
            "sm100_utils.make_trivial_tiled_mma(",
            "tcgen05.CtaGroup.ONE",
            "cute.nvgpu.make_tiled_tma_atom_A(",
            "pipeline.PipelineTmaUmma.create(",
            "tiled_mma.make_fragment_C(",
            "cute.gemm(",
            "cute.ceil_div(",
        ):
            self.assertIn(snippet, reference)
        self.assertIn(
            "[candidate-gemm-api.md](references/candidate-gemm-api.md)",
            skill_body,
        )
        self.assertIn(
            "[candidate-kernel-patterns.md]"
            "(references/candidate-kernel-patterns.md)",
            skill_body,
        )
        for snippet in (
            "empty_ab = ab_producer.acquire_and_advance()",
            "full_ab = ab_consumer.wait_and_advance()",
            "full_ab.release()",
            "empty_accumulator = acc_producer.acquire_and_advance()",
            "empty_accumulator.commit()",
            "full_accumulator = acc_consumer.wait_and_advance()",
            "full_accumulator.release()",
            "There is no",
            "task-specific",
        ):
            self.assertIn(snippet, patterns)
        for leaked_task_detail in (
            "4096",
            "1024",
            "8192",
            "OUTPUT_SCALE",
            "FP8_MAX",
            "relu(",
        ):
            self.assertNotIn(leaked_task_detail, patterns)

    def test_evaluators_do_not_depend_on_candidate_problem_constants(self):
        gemm_constants = {
            "M",
            "N",
            "K",
            "FP8_MAX",
            "WEIGHT_BOUND",
            "SCALE_A",
            "SCALE_B",
            "FP8_DTYPE",
            "AB_DTYPE",
        }
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
            "level1_72_conv_transpose3d_fp8": {
                "BATCH_SIZE",
                "IN_CHANNELS",
                "OUT_CHANNELS",
                "GROUPS",
                "IN_D",
                "IN_H",
                "IN_W",
                "OUT_D",
                "OUT_H",
                "OUT_W",
                "FP8_MAX",
                "INPUT_SCALE",
                "WEIGHT_SCALE",
                "FP8_DTYPE",
            },
            "level2_09_matmul_subtract_multiply_relu_fp8": gemm_constants,
            "level2_12_gemm_multiply_leaky_relu_fp8": gemm_constants,
            "level2_14_gemm_divide_sum_scaling_fp8": gemm_constants,
            "level2_40_matmul_scaling_residual_add_fp8": gemm_constants,
            "level2_63_gemm_relu_divide_fp8": gemm_constants,
            "level2_76_gemm_add_relu_fp8": gemm_constants,
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
        task = discover_tasks()[task_id]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "agent-work"
            code = main(["prepare", task_id, "--output", str(output)])
            self.assertEqual(code, 0)
            self.assertTrue((output / "TASK.md").is_file())
            self.assertTrue((output / "submission.py").is_file())
            public = json.loads((output / "task.json").read_text("utf-8"))
            self.assertNotIn("baseline", public)
            self.assertEqual(public["starter"], "submission.py")
            self.assertEqual(
                public["references"],
                ["references/TASK_REFERENCE.md"],
            )
            for reference in public["references"]:
                self.assertTrue((output / reference).is_file())
            self.assertEqual(
                public["agent_skills"],
                [
                    ".opencode/skills/cute-fp8-kernels/SKILL.md",
                ],
            )
            skill = output / public["agent_skills"][0]
            self.assertTrue(skill.is_file())
            source_chapters = {
                path.name
                for path in (
                    task.agent_skill_paths[0] / "references"
                ).glob("*.md")
            }
            installed_chapters = {
                path.name
                for path in (skill.parent / "references").glob("*.md")
            }
            self.assertEqual(installed_chapters, source_chapters)
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

    def test_compare_reuses_one_baseline_and_calculates_speedups(self):
        task_id = "level1_01_square_matrix_multiplication_fp8"
        task = discover_tasks()[task_id]
        candidate_times = iter((1.0, 0.5))

        def fake_run_one(*args):
            candidate_kind = args[10]
            kernel_time_ms = (
                2.0
                if candidate_kind == "baseline"
                else next(candidate_times)
            )
            return True, {
                "acceptance": {"passed": True},
                "benchmark": {"kernel_time_ms": kernel_time_ms},
                "response": {},
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            first = root / "first.py"
            second = root / "second.py"
            source = candidate_starter(task)
            first.write_text(source, encoding="utf-8")
            second.write_text(source, encoding="utf-8")
            output = root / "comparison"

            with patch.dict(
                os.environ,
                {"CUTE_HARNESS_API_KEY": "test-only-placeholder"},
            ), patch(
                "cute_harness.cli._run_one",
                side_effect=fake_run_one,
            ) as run_one:
                code = main(
                    [
                        "compare",
                        f"{task_id}={first}",
                        f"{task_id}={second}",
                        "--seed",
                        "0",
                        "--output",
                        str(output),
                    ]
                )

            report = json.loads(
                (output / "comparison.json").read_text(encoding="utf-8")
            )
            table = (output / "comparison.txt").read_text(encoding="utf-8")

        self.assertEqual(code, 0)
        self.assertEqual(run_one.call_count, 3)
        self.assertEqual(len(report["rows"]), 2)
        self.assertEqual(report["rows"][0]["speedup"], 2.0)
        self.assertEqual(report["rows"][1]["speedup"], 4.0)
        self.assertEqual(report["rows"][0]["baseline_ms"], 2.0)
        self.assertIn("2.000x", table)
        self.assertIn("4.000x", table)

    def test_compare_rejects_malformed_submission_spec(self):
        code = main(["compare", "missing-equals-sign"])
        self.assertEqual(code, 2)


if __name__ == "__main__":
    unittest.main()
