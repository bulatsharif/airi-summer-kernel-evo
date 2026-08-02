# Compilation, debugging, and generated-code inspection (4.2.x)

Never guess an API name:

```bash
kernel-evo cute lookup cute.make_tiled_copy_tv
kernel-evo cute lookup warpgroup.MmaF8Op
kernel-evo cute lookup pipeline.PipelineTmaAsync
```

The 4.2.1 compiler accepts `cute.compile(jit_fn, *args, options="--opt-level 3")`. The returned `JitExecutor` exposes `ir_module`; it does not expose the newer `__ptx__`, `__cubin__`, or `__mlir__` attributes documented for later releases.

For 4.2.x, `CUTE_DSL_KEEP_IR=1` retains generated MLIR in the working directory. `kernel-evo cute compile` selects the appropriate artifact switches for the installed version and moves generated artifacts into a run directory.

Diagnose in this order:

1. Python import/annotation/preprocessor error.
2. CuTe layout or DSL lowering error at the first relevant source line.
3. NVVM/PTXAS architecture or resource error.
4. CUDA launch, alignment, or illegal-memory error.
5. Barrier hang or race.
6. Correct but scalarized/spilling code.

Code-generation questions:

```bash
kernel-evo cute inspect-codegen kernel.cubin --contract expected_codegen.yaml
```

The summary counts WGMMA, MMA, TMA, cp.async, barriers, vector GMEM operations, and local-memory loads/stores. Hopper SASS may spell WGMMA as `QGMMA`/`HGMMA`; the inspector groups those mnemonics into the WGMMA family. Local `LDL`/`STL` is a spill warning. A Hopper FP8 GEMM without WGMMA is not the intended hardware path.

`--contract` turns required/forbidden instruction families and resource expectations into a structured pass/fail gate. During a KernelEvo barrier, the same gate is applied when the assigned idea has a matching example contract and the candidate retained an artifact. Missing artifacts remain an explicit absence of evidence; they are not fabricated from source syntax.

Debug/retained-artifact builds can change compilation and caching behavior. Recompile and benchmark with debug switches off before accepting performance.
