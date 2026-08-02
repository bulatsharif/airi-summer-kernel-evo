# KernelEvo Python CuTe DSL laboratory

This directory is the executable, version-aware CuTe context for KernelEvo. It is intentionally limited to the Python CuTe DSL (`nvidia-cutlass-dsl`, imported as `cutlass.cute`). It does not mix in CuTe C++ or the legacy CUTLASS Python operation API.

The first architecture corpus targets Hopper `sm_90a`. BF16 storage with Float32 accumulation is the default path. Hopper FP8 covers E4M3/E5M2 WGMMA with TMA-fed shared memory, and keeps BF16 as the normal KernelEvo module ABI when `precision: fp8` uses `runtime_precision: bf16`.

Useful entry points:

```bash
kernel-evo cute path
kernel-evo cute doctor --arch sm_90a
kernel-evo cute lookup warpgroup.MmaF8Op
kernel-evo cute search --precision fp8 --operation gemm --concept wgmma
kernel-evo cute context --precision bf16 --operation gemm
kernel-evo cute probe-layout --shape 4,8 --stride 8,1 --coord 2,3
kernel-evo cute probe-layout --dsl --shape 4,8 --stride 8,1 --coord 2,3 --tile 2,4
kernel-evo cute lint my_kernel.py --precision fp8 --operation gemm --arch sm_90a
kernel-evo cute spec task.py --precision bf16 --arch sm_90a
kernel-evo cute check-hopper-config --tile 128,128,64 --cluster 1,1 --stages 3 --dtype bf16
kernel-evo cute correctness-plan --operation attention --precision fp8 --tile 128,128,64
kernel-evo cute inspect-codegen kernel.cubin --contract examples/hopper_wgmma_gemm/expected_codegen.yaml
```

The compile/check/benchmark tools execute an explicit command and return one structured JSON object. A command can print its own JSON metrics as its final line:

```bash
kernel-evo cute compile --arch sm_90a -- python my_kernel.py --compile-only
kernel-evo cute check --arch sm_90a --contract expected_codegen.yaml -- python my_kernel.py --check
kernel-evo cute benchmark --arch sm_90a -- python my_kernel.py --benchmark
kernel-evo cute sanitize --arch sm_90a --tool memcheck -- python my_kernel.py --check
kernel-evo cute profile --arch sm_90a --set basic -- python my_kernel.py --benchmark
```

The runner exports `CUTE_HARNESS_ARTIFACT_DIR`; harness-aware examples dump CUBIN/SASS/resource data there automatically.

## Navigation without prompt bloat

An island receives a compact `CUTE_HARNESS.md` route before any harness source. Each selected entry states:

- **use when** — the decision or symptom that justifies reading it;
- **why now** — the task, precision, architecture, and idea signals that selected it;
- **read first** — a short card, construction map, or manifest;
- **deep reference** — at most the configured number of full kernels, opened only for a matching construction;
- **verify APIs** — exact installed symbols to inspect before editing.

The full Hopper GEMM is therefore available but is not the default explanation. Its short `CONSTRUCTION.md` routes an author to the relevant class method and explains the invariant that must be preserved.

For an agent-authored KernelEvo run, `iter prepare` retrieves only the relevant cards and coherent example bundle into each isolated island packet. Configure it with:

```yaml
problem:
  backend: cute

evaluation:
  precision: bf16             # or fp8 with runtime_precision: bf16
  runtime_precision: bf16

cute:
  arch: sm_90a
  harness_enabled: true
  context_cards: 7
  context_max_chars: 10000
  context_deep_files: 1        # full reference kernels exposed per packet
  context_lessons: 3           # compact accepted/rejected local results
  capability_gate: true        # evaluator must prove the configured DSL/GPU contract
  compliance_gate: true        # reject dead/missing Python CuTe paths even if Torch is correct
  codegen_gate: true           # enforce a matching hypothesis contract when artifacts exist
  record_experiments: true     # barrier results feed later packet navigation
  sanitizer_tools: [memcheck, synccheck]
  keep_ir: false              # enable only while diagnosing compilation
  optimization_warnings: false
```

At evaluation, KernelEvo applies the Python CuTe compliance lint, records the evaluator-side capability fingerprint, and stores the result as experiment evidence. Promotion still depends only on KernelEvo correctness and benchmark results. Generated-code contracts become hard gates when a retained artifact and `--contract` are supplied.

Add `sanitizer` to `profiling.runners` for candidates that change copies, TMA, shared-memory ownership, or barriers. KernelEvo then runs memcheck first and stops before synccheck/racecheck/initcheck if memory safety fails; authors do not run this sequence themselves.

The package layout follows the laboratory model from `cute_dsl_py_harness_plan.md`: exact environment metadata, semantic cards, executable examples, repair cards, task contracts, narrow tools, and structured experiment memory.
