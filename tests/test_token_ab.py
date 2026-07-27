import json
from pathlib import Path
import tempfile
import unittest

from experiments.token_ab.summarize import (
    aggregate_trials,
    load_trial,
    parse_tokens,
)


class TokenAbSummaryTests(unittest.TestCase):
    def test_parse_tokens_includes_cached_context(self):
        parsed = parse_tokens(
            "Token input uncached: 1,200\n"
            "Token input cached:   3,400\n"
            "Token outputs:        500\n"
            "Token reasoning:      25\n"
        )
        self.assertEqual(
            parsed,
            {
                "input_uncached": 1200,
                "input_cached": 3400,
                "outputs": 500,
                "reasoning": 25,
            },
        )

    def test_load_and_aggregate_successful_trial(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            run_dir = Path(temp_dir) / "task" / "local" / "001"
            evaluation = run_dir / "evaluation"
            evaluation.mkdir(parents=True)
            (run_dir / "evaluation-attempt").mkdir()
            (run_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "task_id": "task",
                        "arm": "local",
                        "trial_id": "001",
                        "started_at": "2026-07-27T00:00:00Z",
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "status").write_text("completed\n", encoding="utf-8")
            (run_dir / "output.log").write_text(
                "check=PASS\n"
                "Token input uncached: 100\n"
                "Token input cached:   200\n"
                "Token outputs:        30\n"
                "Token reasoning:      4\n",
                encoding="utf-8",
            )
            (evaluation / "result.json").write_text(
                json.dumps(
                    {
                        "acceptance": {"passed": True},
                        "response": {
                            "device_time_ms": 1.25,
                            "profile_id": "profile-1",
                        },
                        "wall_seconds": 2.5,
                        "candidate_sha256": "abc",
                    }
                ),
                encoding="utf-8",
            )

            trial = load_trial(run_dir / "metadata.json")
            aggregates = aggregate_trials([trial])

        self.assertEqual(trial["logical_tokens"], 334)
        self.assertTrue(trial["local_check_passed"])
        self.assertTrue(trial["passed"])
        self.assertEqual(aggregates[0]["success_rate"], 1.0)
        self.assertEqual(aggregates[0]["median_logical_tokens_success"], 334)


if __name__ == "__main__":
    unittest.main()
