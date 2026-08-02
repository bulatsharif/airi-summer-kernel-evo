"""Configuration loading for visible, agent-authored evolution runs."""

from __future__ import annotations

import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from kernel_evo.agent.errors import ConfigurationError
from kernel_evo.core.precision import VALID_PRECISIONS, VALID_RUNTIME_PRECISIONS
from kernel_evo.cute_harness.ablation import DOCUMENTATION_DELIVERY, DOCUMENTATION_TIERS


@dataclass(frozen=True, slots=True)
class AgentRunConfig:
    """Serializable configuration shared by the controller and evaluator."""

    name: str = "kernel-evo-agent"
    problem_path: str = ""
    baseline: str = ""
    tests: str = ""
    custom_tests: str = ""
    backend: str = "triton"
    steps: int = 1
    islands: int = 4
    precision: str = "fp32"
    runtime_precision: str = ""
    measurement_mode: str = "wall-clock"
    timing_method: str = "cuda_event"
    num_correct_trials: int = 5
    num_perf_trials: int = 100
    output_rtol: float | None = 0.01
    output_atol: float | None = 0.01
    device: str = "cuda:0"
    arch_list: str = ""
    cute_harness_enabled: bool = True
    cute_arch: str = ""
    cute_context_cards: int = 7
    cute_context_max_chars: int = 10_000
    cute_context_deep_files: int = 1
    cute_context_lessons: int = 3
    cute_keep_ir: bool = False
    cute_optimization_warnings: bool = False
    cute_capability_gate: bool = True
    cute_compliance_gate: bool = True
    cute_codegen_gate: bool = True
    cute_record_experiments: bool = True
    cute_sanitizer_tools: tuple[str, ...] = ("memcheck", "synccheck")
    author_readable_files: tuple[str, ...] = ()
    documentation_enabled: bool = True
    documentation_tier: str = "errors"
    documentation_delivery: str = "files"
    b300_seed: str = "baseline"
    seed_preflight: bool = False
    max_repairs_per_island: int = 1
    execution_mode: str = "local_execution"
    remote_validator_url: str = "http://localhost:15000"
    remote_poll_interval: float = 1.0
    validator_debug: bool = False
    validator_debug_max_code_chars: int = 50_000
    dataset_src: str = "huggingface"
    dataset_name: str = "ScalingIntelligence/KernelBench"
    level: int | None = None
    problem_id: int | None = None
    candidate_name: str = ""
    tracker: str = ""
    evaluator_kind: str = "kernelbench"
    evaluator_command: tuple[str, ...] = ()
    evaluator_timeout: float = 900.0
    evaluation_seed: int = 0
    evaluation_warmup: int = 2
    evaluation_repeats: int = 5
    harness_url: str = ""
    profile_enabled: bool = False
    profile_timeline: bool = False
    profile_runners: tuple[str, ...] = ("torch",)
    profile_min_speedup: float = 1.0
    profile_regression_budget: int = 4
    profile_parent_before_use: bool = True
    profile_first_capability: bool = True
    profile_require_graph_capturable: bool = True
    profile_agent_ideas: bool = True
    profile_agent_idea_limit: int = 3
    profile_review_required: bool = True
    profile_gpu_idle_timeout: float = 120.0
    profile_gpu_idle_samples: int = 3
    profile_gpu_max_utilization: int = 5
    profile_torch_warmup_steps: int = 2
    profile_torch_active_steps: int = 3
    profile_subprocess_timeout: float = 600.0
    profile_ncu_path: str = "ncu"
    profile_ncu_tmpdir: str = ""
    profile_ncu_set: str = "full"
    profile_ncu_kernel_name: str = ""
    profile_ncu_extra_args: str = ""
    profile_ncu_target_steps: int = 1
    profile_ncu_warmup_steps: int = 1
    profile_ncu_launch_count: int = 128
    migration_interval: int = 3
    ideas: tuple[dict[str, Any], ...] = ()
    rules_file: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path) -> "AgentRunConfig":
        config_path = Path(path).expanduser().resolve()
        if not config_path.exists():
            raise ConfigurationError(f"Config file not found: {config_path}")
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if payload is None:
            payload = {}
        if not isinstance(payload, Mapping):
            raise ConfigurationError(f"Expected a YAML/JSON object in {config_path}")
        return cls.from_mapping(payload, base_dir=config_path.parent)

    @classmethod
    def from_mapping(
        cls,
        payload: Mapping[str, Any],
        *,
        base_dir: str | Path | None = None,
    ) -> "AgentRunConfig":
        base = Path(base_dir or Path.cwd()).expanduser().resolve()
        run = _mapping(payload.get("run"))
        problem = _mapping(payload.get("problem"))
        evaluation = _mapping(payload.get("evaluation"))
        profiling = _mapping(payload.get("profiling", payload.get("profile")))
        scheduler = _mapping(payload.get("scheduler"))
        cute = _mapping(payload.get("cute"))

        command = _command(
            _pick(payload, "evaluator_command", default=evaluation.get("command", ()))
        )
        ideas = _ideas(_pick(payload, "ideas", default=scheduler.get("ideas", ())))
        profile_runners = _string_tuple(
            _pick(payload, "profile_runners", default=profiling.get("runners", ("torch",)))
        )
        cute_sanitizer_tools = _string_tuple(
            _pick(
                payload,
                "cute_sanitizer_tools",
                default=cute.get("sanitizer_tools", ("memcheck", "synccheck")),
            )
        )
        author_readable_files = _resolved_paths(
            _pick(
                payload,
                "author_readable_files",
                default=problem.get("author_readable_files", ()),
            ),
            base,
        )

        known_top_level = set(cls.__dataclass_fields__) | {
            "run",
            "problem",
            "evaluation",
            "profiling",
            "profile",
            "scheduler",
            "cute",
        }
        extra = {str(key): value for key, value in _mapping(payload.get("extra")).items()}
        extra.update({str(key): value for key, value in payload.items() if key not in known_top_level})
        config = cls(
            name=str(_pick(payload, "name", "experiment_name", default=run.get("name", "kernel-evo-agent"))),
            problem_path=_resolved_path(
                _pick(payload, "problem_path", default=problem.get("path", "")), base
            ),
            baseline=_resolved_path(
                _pick(payload, "baseline", default=problem.get("baseline", "")), base
            ),
            tests=_resolved_path(_pick(payload, "tests", default=problem.get("tests", "")), base),
            custom_tests=_resolved_path(
                _pick(
                    payload,
                    "custom_tests",
                    default=evaluation.get("custom_tests", problem.get("custom_tests", "")),
                ),
                base,
            ),
            backend=str(_pick(payload, "backend", default=problem.get("backend", "triton"))),
            steps=int(_pick(payload, "steps", default=run.get("steps", 1))),
            islands=int(_pick(payload, "islands", default=run.get("islands", 4))),
            precision=str(_pick(payload, "precision", default=evaluation.get("precision", "fp32"))),
            runtime_precision=str(
                _pick(payload, "runtime_precision", default=evaluation.get("runtime_precision", ""))
            ),
            measurement_mode=str(
                _pick(
                    payload,
                    "measurement_mode",
                    default=evaluation.get("measurement_mode", "wall-clock"),
                )
            ),
            timing_method=str(
                _pick(payload, "timing_method", default=evaluation.get("timing_method", "cuda_event"))
            ),
            num_correct_trials=int(
                _pick(payload, "num_correct_trials", default=evaluation.get("num_correct_trials", 5))
            ),
            num_perf_trials=int(
                _pick(payload, "num_perf_trials", default=evaluation.get("num_perf_trials", 100))
            ),
            output_rtol=_optional_float(
                _pick(payload, "output_rtol", default=evaluation.get("output_rtol", 0.01))
            ),
            output_atol=_optional_float(
                _pick(payload, "output_atol", default=evaluation.get("output_atol", 0.01))
            ),
            device=str(_pick(payload, "device", default=evaluation.get("device", "cuda:0"))),
            arch_list=str(_pick(payload, "arch_list", default=evaluation.get("arch_list", "")) or ""),
            cute_harness_enabled=bool(
                _pick(payload, "cute_harness_enabled", default=cute.get("harness_enabled", True))
            ),
            cute_arch=str(
                _pick(payload, "cute_arch", default=cute.get("arch", evaluation.get("cute_arch", ""))) or ""
            ),
            cute_context_cards=int(
                _pick(payload, "cute_context_cards", default=cute.get("context_cards", 7))
            ),
            cute_context_max_chars=int(
                _pick(payload, "cute_context_max_chars", default=cute.get("context_max_chars", 10_000))
            ),
            cute_context_deep_files=int(
                _pick(payload, "cute_context_deep_files", default=cute.get("context_deep_files", 1))
            ),
            cute_context_lessons=int(
                _pick(payload, "cute_context_lessons", default=cute.get("context_lessons", 3))
            ),
            cute_keep_ir=bool(_pick(payload, "cute_keep_ir", default=cute.get("keep_ir", False))),
            cute_optimization_warnings=bool(
                _pick(
                    payload,
                    "cute_optimization_warnings",
                    default=cute.get("optimization_warnings", False),
                )
            ),
            cute_capability_gate=bool(
                _pick(payload, "cute_capability_gate", default=cute.get("capability_gate", True))
            ),
            cute_compliance_gate=bool(
                _pick(payload, "cute_compliance_gate", default=cute.get("compliance_gate", True))
            ),
            cute_codegen_gate=bool(
                _pick(payload, "cute_codegen_gate", default=cute.get("codegen_gate", True))
            ),
            cute_record_experiments=bool(
                _pick(payload, "cute_record_experiments", default=cute.get("record_experiments", True))
            ),
            cute_sanitizer_tools=cute_sanitizer_tools,
            author_readable_files=author_readable_files,
            documentation_enabled=bool(
                _pick(
                    payload,
                    "documentation_enabled",
                    default=run.get("documentation_enabled", True),
                )
            ),
            documentation_tier=str(
                _pick(
                    payload,
                    "documentation_tier",
                    default=run.get("documentation_tier", "errors"),
                )
            ),
            documentation_delivery=str(
                _pick(
                    payload,
                    "documentation_delivery",
                    default=run.get("documentation_delivery", "files"),
                )
            ),
            b300_seed=str(
                _pick(payload, "b300_seed", default=run.get("b300_seed", "baseline"))
            ),
            seed_preflight=bool(
                _pick(payload, "seed_preflight", default=run.get("seed_preflight", False))
            ),
            max_repairs_per_island=int(
                _pick(
                    payload,
                    "max_repairs_per_island",
                    default=run.get("max_repairs_per_island", 1),
                )
            ),
            execution_mode=str(
                _pick(payload, "execution_mode", default=evaluation.get("mode", "local_execution"))
            ),
            remote_validator_url=str(
                _pick(
                    payload,
                    "remote_validator_url",
                    default=evaluation.get("remote_validator_url", "http://localhost:15000"),
                )
            ),
            remote_poll_interval=float(
                _pick(payload, "remote_poll_interval", default=evaluation.get("remote_poll_interval", 1.0))
            ),
            validator_debug=bool(
                _pick(payload, "validator_debug", default=evaluation.get("debug", False))
            ),
            validator_debug_max_code_chars=int(
                _pick(
                    payload,
                    "validator_debug_max_code_chars",
                    default=evaluation.get("debug_max_code_chars", 50_000),
                )
            ),
            dataset_src=str(_pick(payload, "dataset_src", default=problem.get("dataset_src", "huggingface"))),
            dataset_name=str(
                _pick(
                    payload,
                    "dataset_name",
                    default=problem.get("dataset_name", "ScalingIntelligence/KernelBench"),
                )
            ),
            level=_optional_int(_pick(payload, "level", default=problem.get("level"))),
            problem_id=_optional_int(_pick(payload, "problem_id", default=problem.get("problem_id"))),
            candidate_name=str(_pick(payload, "candidate_name", default=run.get("candidate_name", "")) or ""),
            tracker=str(_pick(payload, "tracker", default=run.get("tracker", "")) or ""),
            evaluator_kind=str(
                _pick(payload, "evaluator_kind", default=evaluation.get("kind", "kernelbench"))
            ),
            evaluator_command=command,
            evaluator_timeout=float(
                _pick(payload, "evaluator_timeout", default=evaluation.get("timeout", 900.0))
            ),
            evaluation_seed=int(
                _pick(payload, "evaluation_seed", default=evaluation.get("seed", 0))
            ),
            evaluation_warmup=int(
                _pick(payload, "evaluation_warmup", default=evaluation.get("warmup", 2))
            ),
            evaluation_repeats=int(
                _pick(payload, "evaluation_repeats", default=evaluation.get("repeats", 5))
            ),
            harness_url=str(
                _pick(payload, "harness_url", default=evaluation.get("harness_url", "")) or ""
            ),
            profile_enabled=bool(
                _pick(payload, "profile_enabled", default=profiling.get("enabled", False))
            ),
            profile_timeline=bool(
                _pick(payload, "profile_timeline", default=profiling.get("timeline", False))
            ),
            profile_runners=profile_runners,
            profile_min_speedup=float(
                _pick(payload, "profile_min_speedup", default=profiling.get("min_speedup", 1.0))
            ),
            profile_regression_budget=int(
                _pick(
                    payload,
                    "profile_regression_budget",
                    default=profiling.get("regression_budget", 4),
                )
            ),
            profile_parent_before_use=bool(
                _pick(
                    payload,
                    "profile_parent_before_use",
                    default=profiling.get("parent_before_use", True),
                )
            ),
            profile_first_capability=bool(
                _pick(
                    payload,
                    "profile_first_capability",
                    default=profiling.get("first_capability", True),
                )
            ),
            profile_require_graph_capturable=bool(
                _pick(
                    payload,
                    "profile_require_graph_capturable",
                    default=profiling.get("require_graph_capturable", True),
                )
            ),
            profile_agent_ideas=bool(
                _pick(
                    payload,
                    "profile_agent_ideas",
                    default=profiling.get("agent_ideas", True),
                )
            ),
            profile_agent_idea_limit=int(
                _pick(
                    payload,
                    "profile_agent_idea_limit",
                    default=profiling.get("agent_idea_limit", 3),
                )
            ),
            profile_review_required=bool(
                _pick(
                    payload,
                    "profile_review_required",
                    default=profiling.get("review_required", True),
                )
            ),
            profile_gpu_idle_timeout=float(
                _pick(
                    payload,
                    "profile_gpu_idle_timeout",
                    default=profiling.get("gpu_idle_timeout", 120.0),
                )
            ),
            profile_gpu_idle_samples=int(
                _pick(
                    payload,
                    "profile_gpu_idle_samples",
                    default=profiling.get("gpu_idle_samples", 3),
                )
            ),
            profile_gpu_max_utilization=int(
                _pick(
                    payload,
                    "profile_gpu_max_utilization",
                    default=profiling.get("gpu_max_utilization", 5),
                )
            ),
            profile_torch_warmup_steps=int(
                _pick(payload, "profile_torch_warmup_steps", default=profiling.get("torch_warmup_steps", 2))
            ),
            profile_torch_active_steps=int(
                _pick(payload, "profile_torch_active_steps", default=profiling.get("torch_active_steps", 3))
            ),
            profile_subprocess_timeout=float(
                _pick(
                    payload,
                    "profile_subprocess_timeout",
                    default=profiling.get("subprocess_timeout", 600.0),
                )
            ),
            profile_ncu_path=str(
                _pick(payload, "profile_ncu_path", default=profiling.get("ncu_path", "ncu"))
            ),
            profile_ncu_tmpdir=str(
                _pick(payload, "profile_ncu_tmpdir", default=profiling.get("ncu_tmpdir", ""))
                or ""
            ),
            profile_ncu_set=str(
                _pick(payload, "profile_ncu_set", default=profiling.get("ncu_set", "full"))
                or "full"
            ),
            profile_ncu_kernel_name=str(
                _pick(
                    payload,
                    "profile_ncu_kernel_name",
                    default=profiling.get("ncu_kernel_name", ""),
                )
                or ""
            ),
            profile_ncu_extra_args=str(
                _pick(
                    payload,
                    "profile_ncu_extra_args",
                    default=profiling.get("ncu_extra_args", ""),
                )
                or ""
            ),
            profile_ncu_target_steps=max(
                1,
                int(
                    _pick(
                        payload,
                        "profile_ncu_target_steps",
                        default=profiling.get("ncu_target_steps", 1),
                    )
                ),
            ),
            profile_ncu_warmup_steps=max(
                0,
                int(
                    _pick(
                        payload,
                        "profile_ncu_warmup_steps",
                        default=profiling.get("ncu_warmup_steps", 1),
                    )
                ),
            ),
            profile_ncu_launch_count=max(
                0,
                int(
                    _pick(
                        payload,
                        "profile_ncu_launch_count",
                        default=profiling.get("ncu_launch_count", 128),
                    )
                ),
            ),
            migration_interval=int(
                _pick(payload, "migration_interval", default=scheduler.get("migration_interval", 3))
            ),
            ideas=ideas,
            rules_file=_resolved_path(
                _pick(payload, "rules_file", default=scheduler.get("rules_file", "")), base
            ),
            extra=extra,
        )
        config.validate()
        return config

    def with_overrides(self, overrides: Mapping[str, Any]) -> "AgentRunConfig":
        payload = self.to_dict()
        payload.update({key: value for key, value in overrides.items() if value is not None})
        return type(self).from_mapping(payload)

    def validate(self) -> None:
        if self.steps < 1:
            raise ConfigurationError("steps must be at least 1")
        if self.islands < 1:
            raise ConfigurationError("islands must be at least 1")
        if self.backend not in {"triton", "cuda_inline", "cute"}:
            raise ConfigurationError("backend must be one of: triton, cuda_inline, cute")
        if self.precision not in VALID_PRECISIONS:
            raise ConfigurationError(f"Unsupported precision: {self.precision}")
        if self.runtime_precision and self.runtime_precision not in VALID_RUNTIME_PRECISIONS:
            raise ConfigurationError(f"Unsupported runtime precision: {self.runtime_precision}")
        if self.measurement_mode not in {"wall-clock", "device-time"}:
            raise ConfigurationError(
                "measurement_mode must be one of: wall-clock, device-time"
            )
        if self.execution_mode not in {"local_execution", "remote_execution"}:
            raise ConfigurationError("execution_mode must be local_execution or remote_execution")
        if self.cute_context_cards < 1 or self.cute_context_cards > 20:
            raise ConfigurationError("cute.context_cards must be between 1 and 20")
        if self.cute_context_max_chars < 2_000:
            raise ConfigurationError("cute.context_max_chars must be at least 2000")
        if self.cute_context_deep_files < 0 or self.cute_context_deep_files > 3:
            raise ConfigurationError("cute.context_deep_files must be between 0 and 3")
        if self.cute_context_lessons < 0 or self.cute_context_lessons > 8:
            raise ConfigurationError("cute.context_lessons must be between 0 and 8")
        invalid_sanitizers = set(self.cute_sanitizer_tools).difference(
            {"memcheck", "racecheck", "initcheck", "synccheck"}
        )
        if invalid_sanitizers:
            raise ConfigurationError(
                "cute.sanitizer_tools contains unsupported tools: "
                + ", ".join(sorted(invalid_sanitizers))
            )
        if self.cute_arch:
            from kernel_evo.cute_harness.capabilities import normalize_cute_arch

            try:
                normalized_arch = normalize_cute_arch(self.cute_arch)
            except ValueError as exc:
                raise ConfigurationError(str(exc)) from exc
            if self.backend == "cute" and self.precision == "fp8" and normalized_arch == "sm_90":
                raise ConfigurationError("Hopper FP8 WGMMA requires cute.arch: sm_90a, not sm_90")
        if not self.problem_path and not self.baseline and self.level is None:
            raise ConfigurationError(
                "Provide baseline, problem_path, or a KernelBench level/problem_id pair."
            )
        if (self.level is None) != (self.problem_id is None):
            raise ConfigurationError("KernelBench level and problem_id must be provided together")
        if self.candidate_name and Path(self.candidate_name).name != self.candidate_name:
            raise ConfigurationError("candidate_name must be a filename, not a path")
        if self.evaluator_timeout <= 0:
            raise ConfigurationError("evaluator_timeout must be positive")
        if self.evaluator_kind not in {"kernelbench", "cute_b300"}:
            raise ConfigurationError("evaluation.kind must be kernelbench or cute_b300")
        if self.evaluation_seed < 0 or self.evaluation_warmup < 0 or self.evaluation_repeats < 1:
            raise ConfigurationError("evaluation seed/warmup/repeats are invalid")
        if self.evaluator_kind == "cute_b300" and self.backend != "cute":
            raise ConfigurationError("evaluation.kind=cute_b300 requires backend=cute")
        if self.documentation_tier not in DOCUMENTATION_TIERS:
            raise ConfigurationError(
                "documentation_tier must be one of: " + ", ".join(DOCUMENTATION_TIERS)
            )
        if self.documentation_delivery not in DOCUMENTATION_DELIVERY:
            raise ConfigurationError(
                "documentation_delivery must be one of: " + ", ".join(DOCUMENTATION_DELIVERY)
            )
        if self.documentation_delivery == "prompt" and self.evaluator_kind != "cute_b300":
            raise ConfigurationError(
                "documentation_delivery=prompt is implemented for the CuTe task bundle; "
                "it requires evaluation.kind=cute_b300"
            )
        if self.b300_seed not in {"baseline", "starter"}:
            raise ConfigurationError("b300_seed must be baseline or starter")
        if self.b300_seed == "starter" and self.seed_preflight:
            raise ConfigurationError(
                "b300_seed=starter is intentionally incomplete; disable seed_preflight"
            )
        if self.custom_tests and not Path(self.custom_tests).is_file():
            raise ConfigurationError(f"Custom test file not found: {self.custom_tests}")
        if self.migration_interval < 0:
            raise ConfigurationError("migration_interval cannot be negative")
        if self.profile_min_speedup < 0:
            raise ConfigurationError("profiling.min_speedup cannot be negative")
        if self.profile_timeline and not self.profile_enabled:
            raise ConfigurationError("profiling.timeline=true requires profiling.enabled=true")
        if self.profile_timeline and self.evaluator_kind != "cute_b300":
            raise ConfigurationError("profiling.timeline=true requires evaluation.kind=cute_b300")
        if self.profile_regression_budget < 0:
            raise ConfigurationError("profiling.regression_budget cannot be negative")
        if self.profile_agent_idea_limit < 1 or self.profile_agent_idea_limit > 8:
            raise ConfigurationError("profiling.agent_idea_limit must be between 1 and 8")
        if self.profile_gpu_idle_timeout < 0:
            raise ConfigurationError("profiling.gpu_idle_timeout cannot be negative")
        if self.profile_gpu_idle_samples < 1:
            raise ConfigurationError("profiling.gpu_idle_samples must be positive")
        if self.profile_gpu_max_utilization < 0 or self.profile_gpu_max_utilization > 100:
            raise ConfigurationError("profiling.gpu_max_utilization must be between 0 and 100")
        if self.max_repairs_per_island < 0 or self.max_repairs_per_island > 3:
            raise ConfigurationError("max_repairs_per_island must be between 0 and 3")
        missing_readables = [path for path in self.author_readable_files if not Path(path).is_file()]
        if missing_readables:
            raise ConfigurationError(
                "author_readable_files contains unavailable files: "
                + ", ".join(missing_readables)
            )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["evaluator_command"] = list(self.evaluator_command)
        payload["profile_runners"] = list(self.profile_runners)
        payload["cute_sanitizer_tools"] = list(self.cute_sanitizer_tools)
        payload["author_readable_files"] = list(self.author_readable_files)
        payload["ideas"] = [dict(idea) for idea in self.ideas]
        return payload


def load_agent_config(
    config: AgentRunConfig | Mapping[str, Any] | str | Path,
    *,
    overrides: Mapping[str, Any] | None = None,
) -> AgentRunConfig:
    if isinstance(config, AgentRunConfig):
        loaded = config
    elif isinstance(config, Mapping):
        loaded = AgentRunConfig.from_mapping(config)
    else:
        loaded = AgentRunConfig.from_file(config)
    return loaded.with_overrides(overrides or {}) if overrides else loaded


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _pick(payload: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in payload and payload[key] is not None:
            return payload[key]
    return default


def _resolved_path(value: Any, base: Path) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base / path
    return str(path.resolve())


def _resolved_paths(value: Any, base: Path) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        items = [item.strip() for item in value.split(",") if item.strip()]
    elif isinstance(value, Sequence):
        items = [str(item).strip() for item in value if str(item).strip()]
    else:
        raise ConfigurationError("author_readable_files must be a list or comma-separated string")
    return tuple(_resolved_path(item, base) for item in items)


def _command(value: Any) -> tuple[str, ...]:
    if not value:
        return ()
    if isinstance(value, str):
        try:
            return tuple(shlex.split(value))
        except ValueError as exc:
            raise ConfigurationError(f"Invalid evaluator command: {exc}") from exc
    if isinstance(value, Sequence):
        return tuple(str(item) for item in value)
    raise ConfigurationError("evaluation.command must be a string or a list of arguments")


def _string_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return tuple(item.strip() for item in value.split(",") if item.strip())
    if isinstance(value, Sequence):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()


def _ideas(value: Any) -> tuple[dict[str, Any], ...]:
    if not value:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ConfigurationError("ideas must be a list of strings or objects")
    normalized: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if isinstance(item, str):
            normalized.append({"id": f"user-{index + 1}", "summary": item})
        elif isinstance(item, Mapping):
            summary = str(item.get("summary", item.get("idea", ""))).strip()
            if not summary:
                raise ConfigurationError(f"Idea {index + 1} has no summary")
            normalized.append(
                {
                    "id": str(item.get("id", f"user-{index + 1}")),
                    "summary": summary,
                    "mechanism": str(item.get("mechanism", "")),
                    "codegen_contract": str(item.get("codegen_contract", "")),
                    "requires_capability": str(item.get("requires_capability", "")),
                    "produces_capability": str(item.get("produces_capability", "")),
                    "requires_candidate_kernel": bool(
                        item.get("requires_candidate_kernel", False)
                    ),
                    "min_new_executors": int(item.get("min_new_executors", 1) or 1),
                }
            )
        else:
            raise ConfigurationError(f"Idea {index + 1} must be a string or object")
    return tuple(normalized)


def _optional_int(value: Any) -> int | None:
    return None if value in (None, "") else int(value)


def _optional_float(value: Any) -> float | None:
    return None if value in (None, "") else float(value)
