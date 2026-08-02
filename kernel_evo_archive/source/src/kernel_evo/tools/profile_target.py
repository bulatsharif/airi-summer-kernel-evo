from __future__ import annotations

import argparse
import json
from pathlib import Path

from kernel_evo.core.profile.torch_runner import run_torch_profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the candidate kernel repeatedly without torch.profiler.")
    parser.add_argument("--run-config", required=True)
    parser.add_argument("--candidate-file", required=True)
    parser.add_argument("--reference-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--full-profile",
        action="store_true",
        help="Collect Torch objectives instead of only checking repeated target execution.",
    )
    parser.add_argument(
        "--attempt-graph-with-allocations",
        action="store_true",
        help="Attempt CUDA graph capture even when Torch reports allocation events.",
    )
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--active-steps", type=int, default=None)
    parser.add_argument(
        "--target-steps",
        type=int,
        default=None,
        help="Run exactly this many target-only forwards instead of reusing Torch profiler steps.",
    )
    parser.add_argument(
        "--target-warmup-steps",
        type=int,
        default=None,
        help="Run this many uncaptured target forwards before target-only collection.",
    )
    parser.add_argument(
        "--cuda-profiler-range",
        action="store_true",
        help="Enable CUDA profiler collection only around the prepared target forwards.",
    )
    parser.add_argument("--evaluator-runtime-us", type=float, default=None)
    args = parser.parse_args()

    run_config = json.loads(Path(args.run_config).read_text(encoding="utf-8"))
    if args.attempt_graph_with_allocations:
        run_config["profile_torch_attempt_graph_with_allocations"] = True
    if args.warmup_steps is not None:
        run_config["profile_torch_warmup_steps"] = max(1, args.warmup_steps)
    if args.active_steps is not None:
        run_config["profile_torch_active_steps"] = max(1, args.active_steps)
    if args.target_steps is not None:
        run_config["profile_target_steps"] = max(1, args.target_steps)
    if args.target_warmup_steps is not None:
        run_config["profile_target_warmup_steps"] = max(0, args.target_warmup_steps)
    if args.cuda_profiler_range:
        run_config["profile_cuda_profiler_range"] = True
    if args.evaluator_runtime_us is not None:
        run_config["profile_evaluator_runtime_us"] = args.evaluator_runtime_us
    candidate_code = Path(args.candidate_file).read_text(encoding="utf-8")
    reference_code = Path(args.reference_file).read_text(encoding="utf-8")
    summary = run_torch_profile(
        run_config=run_config,
        ref_arch_src=reference_code,
        custom_model_src=candidate_code,
        out_dir=Path(args.out_dir).expanduser().resolve(),
        target_only=not args.full_profile,
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
