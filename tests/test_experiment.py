from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from cute_harness.tasks import REPO_ROOT
from experiment.agent import (
    AgentMetrics,
    _event_session_metrics,
    _query_session_metrics,
    build_agent_prompt,
    build_workspace_inline_config,
)
from experiment.evaluation import EvaluationResult
from experiment.process import run_streaming
from experiment.runner import ExperimentConfig, run_experiment


def evaluation_record(kernel_time_ms, profile_id):
    return {
        "acceptance": {"passed": True},
        "benchmark": {"kernel_time_ms": kernel_time_ms},
        "response": {"profile_id": profile_id},
    }


class ExperimentRunnerTests(unittest.TestCase):
    def test_event_metrics_aggregate_finished_steps(self):
        events = (
            {
                "type": "step_finish",
                "timestamp": 1000,
                "sessionID": "session_test",
                "part": {
                    "tokens": {
                        "input": 10,
                        "output": 4,
                        "reasoning": 1,
                        "cache": {"read": 20, "write": 2},
                    }
                },
            },
            {
                "type": "step_finish",
                "timestamp": 2500,
                "sessionID": "session_test",
                "part": {
                    "tokens": {
                        "input": 3,
                        "output": 2,
                        "reasoning": 0,
                        "cache": {"read": 7, "write": 0},
                    }
                },
            },
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "events.jsonl"
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            metrics = _event_session_metrics(path)

        self.assertEqual(metrics["sessions"], 1)
        self.assertEqual(metrics["input_uncached"], 13)
        self.assertEqual(metrics["input_cached"], 27)
        self.assertEqual(metrics["cache_write"], 2)
        self.assertEqual(metrics["output"], 6)
        self.assertEqual(metrics["reasoning"], 1)
        self.assertEqual(metrics["root_wall_ms"], 1500)

    @patch("experiment.agent.shutil.which", return_value="C:/tools/opencode.cmd")
    @patch("experiment.agent.subprocess.run")
    def test_metrics_query_resolves_wrapper_and_has_timeout(
        self,
        run,
        _which,
    ):
        run.return_value = type(
            "Completed",
            (),
            {
                "returncode": 0,
                "stdout": (
                    '[{"sessions":1,"input_uncached":2,'
                    '"input_cached":3,"cache_write":0,'
                    '"output":4,"reasoning":0}]'
                ),
                "stderr": "",
            },
        )()

        row = _query_session_metrics("opencode", "session_test")

        self.assertEqual(row["output"], 4)
        self.assertEqual(run.call_args.args[0][0], "C:/tools/opencode.cmd")
        self.assertEqual(run.call_args.kwargs["timeout"], 15.0)

    def test_agent_config_bounds_requests_without_disabling_subagents(self):
        config = json.loads(
            (REPO_ROOT / "opencode.json").read_text(encoding="utf-8")
        )
        provider = config["provider"]["qwen-server"]
        model = provider["models"]["qwen3.6-35b-a3b"]

        self.assertNotEqual(config["permission"].get("task"), "deny")
        self.assertEqual(config["permission"]["doom_loop"], "deny")
        self.assertEqual(
            config["permission"]["read"]["runs/**"],
            "deny",
        )
        self.assertEqual(
            config["permission"]["read"]["work/**"],
            "deny",
        )
        self.assertEqual(
            config["permission"]["bash"]["python3 -m cute_harness *"],
            "allow",
        )
        self.assertEqual(
            config["permission"]["bash"][
                "cp .opencode/skills/cute-fp8-kernels/references/"
                "candidate-dense-gemm-template.py submission.py"
            ],
            "allow",
        )
        self.assertEqual(
            config["permission"]["bash"][
                "cp .opencode/skills/cute-fp8-kernels/references/"
                "candidate-elementwise-template.py submission.py"
            ],
            "allow",
        )
        self.assertEqual(
            config["permission"]["bash"][
                'cp ".opencode/skills/cute-fp8-kernels/references/'
                'candidate-dense-gemm-template.py" submission.py'
            ],
            "allow",
        )
        self.assertLessEqual(model["limit"]["output"], 8192)
        self.assertLessEqual(provider["options"]["timeout"], 180000)

    def test_agent_prompt_uses_permitted_python3_harness_command(self):
        prompt = build_agent_prompt(
            "task-id",
            Path("/tmp/work/submission.py"),
            seed=0,
            gpu_timeout=600.0,
            agent_timeout=600.0,
        )

        self.assertIn("python3 -m cute_harness check", prompt)
        self.assertIn("python3 -m cute_harness run", prompt)
        self.assertIn(
            'cp ".opencode/skills/cute-fp8-kernels/references/'
            'candidate-dense-gemm-template.py" submission.py',
            prompt,
        )
        self.assertIn("whole task has a 600-second wall-clock budget", prompt)
        self.assertIn("`cp` must be the first mutating tool call", prompt)
        self.assertNotIn("\npython -m cute_harness", prompt)
        self.assertIn("Do not add pipes, redirects,", prompt)
        self.assertIn("command chaining", prompt)
        self.assertIn("Do not use python3 -c", prompt)
        self.assertIn("remote harness compiler error is the API oracle", prompt)
        self.assertIn("compile-verified candidate template", prompt)
        self.assertIn("Never rerun an identical candidate", prompt)
        self.assertIn(
            "Resolve every path in task.json.references relative to the "
            "current workspace",
            prompt,
        )

    def test_inline_config_reopens_only_the_current_workspace(self):
        workspace = REPO_ROOT / "work" / "current-attempt"
        inline = json.loads(
            build_workspace_inline_config(
                REPO_ROOT,
                workspace,
                '{"instructions":["existing.md"]}',
            )
        )

        self.assertEqual(inline["instructions"], ["existing.md"])
        self.assertEqual(
            inline["permission"]["read"]["work/current-attempt/**"],
            "allow",
        )
        self.assertEqual(
            inline["permission"]["edit"][
                "work/current-attempt/submission.py"
            ],
            "allow",
        )

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
            self.assertEqual(
                public["references"],
                ["references/TASK_REFERENCE.md"],
            )
            for reference in public["references"]:
                self.assertTrue((workspace / reference).is_file())
            self.assertEqual(
                public["agent_skills"],
                [
                    ".opencode/skills/cute-fp8-kernels/SKILL.md",
                ],
            )
            skill = workspace / public["agent_skills"][0]
            self.assertTrue(skill.is_file())
            self.assertTrue((skill.parent / "references" / "fp8.md").is_file())
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

    def test_baseline_failure_skips_agent_attempts(self):
        def unexpected_agent(**_kwargs):
            self.fail("agent must not run when the baseline is invalid")

        def failed_baseline(**kwargs):
            self.assertTrue(kwargs["baseline"])
            return EvaluationResult(
                exit_code=1,
                record=None,
                error="baseline evaluator failed",
            )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config = ExperimentConfig(
                model="qwen-server/qwen3.6-35b-a3b",
                task_ids=(
                    "level1_01_square_matrix_multiplication_fp8",
                ),
                attempts=3,
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
                agent_runner=unexpected_agent,
                evaluation_runner=failed_baseline,
            )

            self.assertFalse(passed)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["status"], "BASELINE_FAIL")
            self.assertIsNone(rows[0]["attempt"])
            self.assertEqual(list((root / "work").iterdir()), [])


class StreamingProcessTests(unittest.TestCase):
    def test_output_is_teed_to_terminal_and_log(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_streaming(
                    [
                        sys.executable,
                        "-c",
                        "print('visible immediately', flush=True)",
                    ],
                    cwd=root,
                    environment=os.environ,
                    log_path=root / "process.log",
                    timeout=5.0,
                )

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.timed_out)
            self.assertEqual(output.getvalue(), "visible immediately\n")
            self.assertEqual(
                (root / "process.log").read_text(encoding="utf-8"),
                "visible immediately\n",
            )

    def test_timeout_stops_process_group(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            started = time.monotonic()
            with redirect_stdout(io.StringIO()):
                result = run_streaming(
                    [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(30)",
                    ],
                    cwd=root,
                    environment=os.environ,
                    log_path=root / "timeout.log",
                    timeout=0.1,
                )

            self.assertEqual(result.exit_code, 124)
            self.assertTrue(result.timed_out)
            self.assertLess(time.monotonic() - started, 10.0)
            self.assertIn(
                "process timed out after 0.1s",
                (root / "timeout.log").read_text(encoding="utf-8"),
            )

    def test_heartbeat_reports_a_silent_live_process(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = io.StringIO()
            with redirect_stdout(output):
                result = run_streaming(
                    [
                        sys.executable,
                        "-c",
                        "import time; time.sleep(0.15)",
                    ],
                    cwd=root,
                    environment=os.environ,
                    log_path=root / "heartbeat.log",
                    timeout=5.0,
                    heartbeat_interval=0.05,
                )

            self.assertEqual(result.exit_code, 0)
            self.assertFalse(result.timed_out)
            self.assertIn(
                "[experiment] still running",
                output.getvalue(),
            )
            self.assertIn(
                "no output for",
                (root / "heartbeat.log").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
