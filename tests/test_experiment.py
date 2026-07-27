import json
from pathlib import Path
import tempfile
import unittest

from experiment.agent import AgentMetrics
from experiment.evaluation import EvaluationResult
from experiment.runner import ExperimentConfig, run_experiment


def evaluation_record(kernel_time_ms, profile_id):
    return {
        "acceptance": {"passed": True},
        "benchmark": {"kernel_time_ms": kernel_time_ms},
        "response": {"profile_id": profile_id},
    }


class ExperimentRunnerTests(unittest.TestCase):
    def test_one_task_combines_agent_eval_and_baseline_metrics(self):
        workspaces = []

        def fake_agent(**kwargs):
            workspace = kwargs["workspace"]
            workspaces.append(workspace)
            self.assertTrue((workspace / "TASK.md").is_file())
            self.assertTrue((workspace / "AGENTS.md").is_file())
            public = json.loads(
                (workspace / "task.json").read_text(encoding="utf-8")
            )
            self.assertEqual(public["problem"]["seed"], 0)
            with (workspace / "submission.py").open(
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write("\n# fake agent edit\n")
            return AgentMetrics(
                requested_model="qwen-server/qwen3.6-35b-a3b",
                reported_model="qwen-server/qwen3.6-35b-a3b",
                provider="qwen-server",
                variant="default",
                session_id="ses_test",
                sessions=1,
                input_uncached=100,
                input_cached=200,
                cache_write=0,
                output=30,
                reasoning=0,
                wall_seconds=12.5,
                session_wall_seconds=12.0,
                exit_code=0,
                timed_out=False,
            )

        def fake_evaluation(**kwargs):
            if kwargs["baseline"]:
                return EvaluationResult(
                    0,
                    evaluation_record(10.0, "baseline-profile"),
                    None,
                )
            candidate = kwargs["candidate_path"].read_text(encoding="utf-8")
            self.assertIn("# fake agent edit", candidate)
            return EvaluationResult(
                0,
                evaluation_record(5.0, "candidate-profile"),
                None,
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ExperimentConfig(
                model="qwen-server/qwen3.6-35b-a3b",
                task_ids=(
                    "level1_01_square_matrix_multiplication_fp8",
                ),
                attempts=1,
                agent_timeout=600,
                gpu_timeout=600,
                seed=0,
                warmup=2,
                repeats=5,
                output_dir=root / "run",
                work_root=root / "work",
            )
            passed, rows = run_experiment(
                config,
                agent_runner=fake_agent,
                evaluation_runner=fake_evaluation,
            )

            self.assertTrue(passed)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["speedup"], 2.0)
            self.assertEqual(rows[0]["input_uncached"], 100)
            self.assertEqual(rows[0]["input_cached"], 200)
            self.assertEqual(rows[0]["output_tokens"], 30)
            self.assertEqual(
                rows[0]["model"],
                "qwen-server/qwen3.6-35b-a3b",
            )
            self.assertTrue((root / "run" / "results.csv").is_file())
            self.assertTrue((root / "run" / "results.json").is_file())
            self.assertTrue((root / "run" / "results.txt").is_file())

        self.assertEqual(len(workspaces), 1)


if __name__ == "__main__":
    unittest.main()
