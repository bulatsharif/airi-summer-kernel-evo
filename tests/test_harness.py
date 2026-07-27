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
    assemble_submission,
    candidate_starter,
    split_starter,
)
from cute_harness.client import build_multipart
from cute_harness.cli import _safe_print, main
from cute_harness.policy import check_submission
from cute_harness.tasks import discover_tasks


class TaskManifestTests(unittest.TestCase):
    def test_five_tasks_are_discoverable(self):
        tasks = discover_tasks()
        self.assertEqual(
            set(tasks),
            {
                "level1_01_square_matrix_multiplication_fp8",
                "level1_40_layer_norm_fp8",
                "level1_72_conv_transpose3d_fp8",
                "level2_12_gemm_multiply_leaky_relu_fp8",
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
    def test_probe_rejects_files_outside_probe_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            probe = Path(temp_dir) / "probe.py"
            probe.write_text("print('probe')\n", encoding="utf-8")
            with patch.dict(
                os.environ,
                {"CUTE_HARNESS_API_KEY": "test-only-placeholder"},
            ):
                code = main(["probe", str(probe)])
        self.assertEqual(code, 2)

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

    def test_prepare_can_snapshot_task_selected_api_context(self):
        task_id = "level2_76_gemm_add_relu_fp8"
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "agent-work"
            code = main(
                [
                    "prepare",
                    task_id,
                    "--output",
                    str(output),
                    "--with-api-context",
                ]
            )
            self.assertEqual(code, 0)
            context = output / "docs" / "INDEX.md"
            self.assertTrue(context.is_file())
            self.assertIn("Blackwell FP8 documentation pack", context.read_text("utf-8"))
            self.assertTrue((output / "docs" / "server-api-deltas.md").is_file())
            self.assertTrue((output / "docs" / "examples" / "README.md").is_file())
            public = json.loads((output / "task.json").read_text("utf-8"))
            self.assertEqual(public["api_context"], "docs/INDEX.md")
            self.assertNotIn("opencode/", json.dumps(public))

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

    def test_evaluator_runtime_prelude_is_candidate_independent(self):
        for task in discover_tasks().values():
            with self.subTest(task=task.id):
                candidate, evaluator = split_starter(task)
                self.assertNotIn("import torch", candidate)
                self.assertNotIn("from cutlass.cute.runtime", candidate)
                self.assertNotIn("create_cute_tensor_for_fp8", candidate)
                self.assertIn(
                    "import torch as _harness_torch",
                    evaluator,
                )

                module = ast.parse(evaluator)
                bound: set[str] = set()
                for statement in module.body:
                    if isinstance(statement, ast.Import):
                        for alias in statement.names:
                            bound.add(alias.asname or alias.name.split(".")[0])
                    elif isinstance(statement, ast.ImportFrom):
                        for alias in statement.names:
                            bound.add(alias.asname or alias.name)
                    elif isinstance(statement, ast.Assign):
                        for target in statement.targets:
                            if isinstance(target, ast.Name):
                                bound.add(target.id)
                    elif isinstance(statement, ast.FunctionDef):
                        bound.add(statement.name)

                private_loads = {
                    node.id
                    for node in ast.walk(module)
                    if isinstance(node, ast.Name)
                    and isinstance(node.ctx, ast.Load)
                    and (
                        node.id.startswith("_harness_")
                        or node.id.startswith("_HARNESS_")
                    )
                }
                self.assertEqual(set(), private_loads - bound)

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

    def test_candidate_rejects_release_invalid_cute_apis(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\ndef invalid_release_calls():\n"
                + "    cute.launch_config((1, 1, 1), (128, 1, 1))\n"
                + "    cute.launch_kernel()\n"
                + "    cute.arch.grid_dim_x()\n"
                + "    cute.cdiv(8, 4)\n"
                + "    cute.div(8, 4)\n"
                + "    cute.arch.block_idx(0)\n"
                + "    cute.arch.thread_idx(0)\n"
                + "    grid = cute.Shape[1, 1, 1]\n"
                + "    mode = cute.GemmMode.STREAM_K\n"
                + "    cute.gemm()\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        for name in (
            "cute.launch_config",
            "cute.launch_kernel",
            "cute.arch.grid_dim_x",
            "cute.cdiv",
            "cute.div",
            "cute.GemmMode",
        ):
            self.assertIn(name, joined)
        self.assertIn(
            "cute.gemm requires (atom, d, a, b, c)",
            joined,
        )
        self.assertIn("block_idx takes no positional arguments", joined)
        self.assertIn("thread_idx takes no positional arguments", joined)
        self.assertIn("cute.Shape is a typing union", joined)

    def test_candidate_rejects_release_incompatible_operand_imports(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\nfrom cutlass.cute.nvgpu.tcgen05 import OperandSource\n"
                + "from cutlass.cute.nvgpu import OperandMajorMode\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        self.assertIn("release-incompatible nvgpu import", joined)
        self.assertIn("OperandSource", joined)
        self.assertIn("OperandMajorMode", joined)

    def test_candidate_rejects_v10_hallucinated_mma_apis(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\ndef invalid_v10_calls(tiled_mma, thr_mma):\n"
                + "    cute.struct.MemRange(8)\n"
                + "    cute.thread_id(0)\n"
                + "    cute.block_dim(0)\n"
                + "    tiled_mma.get_slice()\n"
                + "    thr_mma.get_thread_slice(0)\n"
                + "    thr_mma.get_coord(0, 0)\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        self.assertIn("MemRange[...] is a @cute.struct field annotation", joined)
        self.assertIn("cute.thread_id", joined)
        self.assertIn("cute.block_dim", joined)
        self.assertIn("zero-argument get_slice()", joined)
        self.assertIn("get_thread_slice", joined)
        self.assertIn("get_coord", joined)

    def test_candidate_rejects_invalid_trivial_mma_tile_keywords(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\ndef invalid_mma():\n"
                + "    return sm100_utils.make_trivial_tiled_mma(\n"
                + "        m=128, n=128, k=64\n"
                + "    )\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertTrue(
            any(
                "does not accept tile-size keywords" in error
                for error in report.errors
            ),
            report.errors,
        )

    def test_candidate_rejects_bound_but_unlaunched_kernel(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\n@cute.kernel\n"
                + "def extra_kernel(x: cute.Tensor):\n"
                + "    return\n\n"
                + "@cute.jit\n"
                + "def invalid_launch(x: cute.Tensor):\n"
                + "    extra_kernel(x)\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertIn("bound but not launched", "\n".join(report.errors))

    def test_candidate_accepts_bound_kernel_launch_syntax(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\n@cute.kernel\n"
                + "def extra_kernel(x: cute.Tensor):\n"
                + "    return\n\n"
                + "@cute.jit\n"
                + "def valid_launch(x: cute.Tensor):\n"
                + "    extra_kernel(x).launch("
                + "grid=(1, 1, 1), block=(1, 1, 1), smem=0)\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertNotIn("bound but not launched", "\n".join(report.errors))

    def test_candidate_rejects_extra_cute_gemm_arguments(self):
        task = discover_tasks()[
            "level1_01_square_matrix_multiplication_fp8"
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(
                candidate_starter(task)
                + "\ndef invalid_gemm():\n"
                + "    cute.gemm(atom, d, a, b, c, m=128)\n",
                encoding="utf-8",
            )
            report = check_submission(task, candidate)
        self.assertIn(
            "accepts exactly (atom, d, a, b, c)",
            "\n".join(report.errors),
        )

    def test_candidate_rejects_launch_on_unbound_kernel_function(self):
        task = discover_tasks()["level1_40_layer_norm_fp8"]
        source = candidate_starter(task).replace(
            "    # TODO: launch layer_norm_kernel.\n    pass",
            "    layer_norm_kernel.launch(grid=(1, 1, 1), block=(32, 1, 1))",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertIn(
            "must be bound to its arguments before launch",
            "\n".join(report.errors),
        )

    def test_candidate_requires_named_jit_entrypoint(self):
        task = discover_tasks()["level1_40_layer_norm_fp8"]
        source = candidate_starter(task).replace(
            "@cute.jit\ndef layer_norm(",
            "def layer_norm(",
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertIn(
            "evaluator entry point layer_norm must be defined and decorated "
            "with @cute.jit",
            "\n".join(report.errors),
        )

    def test_candidate_rejects_observed_b300_release_mismatches(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

def invalid_release_paths(kernel):
    value = _
    cute._cute.make_smem_storage()
    cute.make_fake_compact_tensor()
    cute.make_shape(1, 2)
    cute.fill(0, layout)
    cute.float32(0)
    cute.full(4, 0.0)
    cute.arch.blockDim()
    cute.arch.make_pipeline_state(kind, stages)
    cute.make_smem_tensor()
    cute.partition_D(dst)
    cute.partition_S(src)
    cute.PipelineTmaUmma.create()
    cute.arch.PipelineTmaUmma.create()
    cute.pipeline.PipelineTmaUmma.create()
    pipeline.PipelineTmaUmma.create(
        num_stages=1,
        producer_group=producer_group,
        consumer_group=consumer_group,
        tx_count=1,
    )
    cute._range(0, 1)
    cute.Shape(1, 2)
    cute.Tile((1, 2))
    sm100_utils.make_smem_tensor_A()
    sm100_utils.make_trivial_pipeline()
    sm100_utils.stage_input_A()
    sm100_utils.stage_input_B()
    sm100_utils.wait_pipeline()
    sm100_utils.commit_pipeline()
    sm100_utils.epilog_tmem_copy_and_partition()
    sm100_utils.epilog_smem_copy_and_partition()
    sm100_utils.epilog_gmem_copy_and_partition()
    sm100_utils.make_tiled_tma_atom_A()
    tiled_mma.make_smem_A(layout)
    layout.cosize()
    output.numel()
    if cute.arch.elect_one():
        pass
    with cute.arch.elect_one():
        cute.copy(tma_atom, src, dst, tma_bar_ptr=barrier)
    fp8_gemm_kernel[1](a, b, c)
    fp8_gemm_kernel(a, b, c, compile_only=True).launch(
        grid=(1, 1, 1), block=(1, 1, 1)
    )
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        for expected in (
            "reading bare '_'",
            "cute._cute",
            "cute.make_fake_compact_tensor",
            "cute.make_shape",
            "cute.fill",
            "cute.float32",
            "cute.full requires (shape, fill_value, dtype)",
            "cute.arch.blockDim",
            "cute.arch.make_pipeline_state",
            "cute.make_smem_tensor",
            "cute.partition_D",
            "cute.partition_S",
            "cute.PipelineTmaUmma",
            "cute.arch.PipelineTmaUmma",
            "cute.pipeline",
            "requires an explicit non-None shared-memory barrier_storage",
            "cute._range",
            "cute.Shape is a typing union",
            "cute.Tile",
            "sm100_utils.make_smem_tensor_A",
            "sm100_utils.make_trivial_pipeline",
            "sm100_utils.stage_input_A",
            "sm100_utils.stage_input_B",
            "sm100_utils.wait_pipeline",
            "sm100_utils.commit_pipeline",
            "sm100_utils.epilog_tmem_copy_and_partition",
            "sm100_utils.epilog_smem_copy_and_partition",
            "sm100_utils.epilog_gmem_copy_and_partition",
            "sm100_utils.make_tiled_tma_atom_A",
            "TiledMma.make_smem_A",
            "use cute.cosize(layout)",
            "use cute.size(tensor)",
            "cute.arch.elect_one() is a context manager",
            "TMA cute.copy(..., tma_bar_ptr=...) already elects one issuing thread",
            "CUDA-style fp8_gemm_kernel[grid]",
            "does not accept compile_only",
        ):
            self.assertIn(expected, joined)
        self.assertIn(
            "use cute.nvgpu.make_tiled_tma_atom_A(...) instead",
            joined,
        )

    def test_candidate_accepts_verified_free_cosize_function(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + (
            "\ndef verified_calls():\n"
            "    cute.cosize(output.layout)\n"
            "    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertNotIn("layout.cosize() is unavailable", "\n".join(report.errors))

    def test_elect_one_guard_is_specific_to_tma_copy(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + (
            "\ndef ordinary_elected_copy():\n"
            "    with cute.arch.elect_one():\n"
            "        cute.copy(atom, matrix_a, output)\n"
            "    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertNotIn(
            "already elects one issuing thread",
            "\n".join(report.errors),
        )

    def test_tma_copy_without_partition_emits_launch_warning(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + (
            "\ndef unpartitioned_tma_copy():\n"
            "    cute.copy(\n"
            "        atom, matrix_a, output, tma_bar_ptr=barrier\n"
            "    )\n"
            "    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)\n"
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertIn(
            "without a tma_partition call",
            "\n".join(report.warnings),
        )

    def test_remote_feedback_guards_fail_before_submission(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

@cute.kernel
def invalid_kernel(matrix_a, matrix_b_nk, bias, output):
    if matrix_a.shape[0] > 0:
        return


@cute.jit
def invalid_remote_calls(matrix_a, matrix_b_nk, bias, output):
    cute.LayoutEnum.Major.K
    sm100_utils._get_major_mode(matrix_a)
    cute.pointer(output)
    cute.raw_pointer_as_ptr(output)
    cute.make_layout((1, 1), (1, 1))
    sm100_utils.make_trivial_tiled_mma(
        matrix_a.element_type,
        matrix_b_nk.element_type,
        cutlass.Float32,
        a_major=major,
    )
    invalid_kernel(matrix_a).launch(grid=(1, 1, 1), block=(32, 1, 1))
    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        for expected in (
            "early return is forbidden inside @cute.kernel",
            "cute.LayoutEnum",
            "sm100_utils._get_major_mode",
            "cute.pointer",
            "cute.raw_pointer_as_ptr",
            "cute.make_layout requires exactly one positional",
            "make_trivial_tiled_mma requires",
            "unsupported keywords: a_major",
            "invalid_kernel requires 4 bound arguments",
        ):
            self.assertIn(expected, joined)

    def test_candidate_rejects_layout_and_dtype_as_tensor_storage(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

def invalid_tensor_construction(pointer):
    cute.make_tensor(cute.make_layout((128, 128)), cutlass.Float32)
    cute.make_tensor(pointer, cutlass.Float32)
    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        self.assertIn("first argument must be a real Pointer", joined)
        self.assertIn("second argument must be a Layout", joined)

    def test_candidate_rejects_unpacking_tma_info(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

def invalid_tma_info():
    atom, tensor = cute.nvgpu.make_tiled_tma_atom_A(
        op, matrix_a, smem_layout, mma_tiler, tiled_mma
    )
    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        self.assertIn(
            "returns one TmaInfo object",
            "\n".join(report.errors),
        )

    def test_candidate_rejects_function_store_and_load_value(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

def invalid_fragment_io():
    fragment.load(fragment)
    thr_mma.partition_C.store(output, fragment)
    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        self.assertIn("tensor load() takes no value argument", joined)
        self.assertIn("do not expose a .store method", joined)

    def test_level2_requires_verified_blackwell_bridge_calls(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        self.assertEqual(
            set(task.policy["required_calls"]),
            {
                "cute.nvgpu.make_tiled_tma_atom_A",
                "cute.nvgpu.make_tiled_tma_atom_B",
                "cpasync.tma_partition",
                "pipeline.PipelineTmaUmma.create",
                "pipeline.PipelineUmmaAsync.create",
                "cute.gemm",
                "tcgen05.make_tmem_copy",
                "cute.autovec_copy",
            },
        )

    def test_candidate_rejects_v9_host_and_pipeline_mistakes(self):
        task = discover_tasks()["level2_76_gemm_add_relu_fp8"]
        source = candidate_starter(task) + """

@cute.kernel
def extra_kernel(x: cute.Tensor):
    pass

@cute.jit
def invalid_v9_patterns(x: cute.Tensor):
    class Storage:
        pass
    cute.arch.alloc_smem(cutlass.Int64, 4)
    cute.make_stride(1, 0)
    pipeline.CooperativeGroup(1, 128)
    extra_kernel(x).launch()
    cute.gemm(atom, output, matrix_a, matrix_b_nk, output)
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.py"
            candidate.write_text(source, encoding="utf-8")
            report = check_submission(task, candidate)
        joined = "\n".join(report.errors)
        for expected in (
            "cute.make_stride",
            "must be called inside @cute.kernel",
            "do not construct Python storage classes inside @cute.jit",
            "first argument must be a pipeline.Agent enum",
            "launch requires explicit grid= and block=",
        ):
            self.assertIn(expected, joined)

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
