# CuTe DSL FP8 kernel work

This repository uses OpenCode to author NVIDIA CuTe DSL kernels in Python,
especially FP8 kernels for the shared B300 GPU.

## Required workflow

- Load the `cute-fp8-kernels` skill before writing, debugging, or optimizing an
  FP8 or block-scaled kernel.
- Implement the requested GPU operation with CuTe DSL Python. Do not substitute
  Triton, CUDA C++, a CUTLASS C++ wrapper, `torch.compile`, or a PyTorch operator.
- PyTorch may generate inputs and compute a correctness reference. It must not
  be the submitted GPU implementation.
- Treat the user's prompt as the operation specification. Identify the
  operation, layouts, FP8 format, scaling convention, accumulator/output types,
  required shapes, correctness rule, and performance goal before coding.
- Follow the task's submission mode. A standalone file owns `main()`, inputs,
  and validation. A prepared harness candidate must not define/call `main()`,
  generate references, or print `PASS`; `cute_harness` appends those parts.
- Establish correctness before optimizing. Build the reference from the actual
  quantized inputs and specified scales; report quantization error separately.
- Do not weaken tolerances, skip required shapes, catch an implementation error
  and report success, or use a reference operation as the kernel.
- Compile once, warm up before timing, and exclude compilation, allocation,
  input generation, and reference computation from kernel latency.
- Verify the intended FP8 MMA path before claiming native FP8 acceleration.
- Leave the requested submission in place and report remote correctness,
  latency, assumptions, and limitations.

## Remote GPU runner

Submit a self-contained Python file to the shared remote GPU service with
`curl` when CUDA-kernel correctness or performance needs to be checked.

- Base URL: `http://109.236.57.62:18080`
- Authentication: read `CUTE_HARNESS_API_KEY` from the environment. Never print,
  embed, or commit the key.

## Submission contents

For prepared tasks with a harness-owned evaluator, edit only the prepared
candidate and use the exact `python -m cute_harness check/run` commands from
the prompt. The rules below describe legacy standalone probes and submissions.

A task submission normally contains the CuTe kernel, its JIT launcher, and a
`main()` that creates deterministic inputs, computes a reference, compiles and
launches the kernel, checks every required case, warms up, and measures repeated
kernel-only executions. Keep this in the task's single `submission.py`; no
additional local harness files are required.

Submit and profile the file:

```bash
curl -sS 'http://109.236.57.62:18080/v1/runs/file' \
  -H "X-API-Key: ${CUTE_HARNESS_API_KEY}" \
  -F 'file=@path/to/submission.py' \
  -F 'profiler=pytorch'
```

The JSON response includes correctness/process information and performance fields
such as `success`, `exit_code`, `stdout`, `stderr`, `device_time_ms`,
`profile_id`, `profile_error`, and `timed_out`.

Download a profile when useful:

```bash
curl -fOJ 'http://109.236.57.62:18080/v1/profiles/<profile_id>' \
  -H "X-API-Key: ${CUTE_HARNESS_API_KEY}"
```

Establish correctness first, then optimize and remeasure promising versions.
Current `device_time_ms` comes from PyTorch Profiler without a controlled warmup,
so treat a single result as directional and repeat final candidates. Keep use of
the shared B300 modest and do not start an uncontrolled optimization loop.
