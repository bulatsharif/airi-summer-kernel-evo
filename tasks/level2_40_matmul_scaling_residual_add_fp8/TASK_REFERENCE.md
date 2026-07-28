# Task reference: KernelBench Level 2/40

Load the `cute-fp8-kernels` skill from `task.json.agent_skills`, then read
`references/candidate-dense-gemm-template.py`. Copy its compile-verified dense
FP8 GEMM core without changing its TMA/pipeline/TMEM flow; this task owns only
the operation-specific elementwise epilogue.
Also read `references/candidate-elementwise-template.py` and preserve its
two-dimensional indexing and launch; replace only its neutral expression.

The exact mathematical order is `linear = X @ W.T + bias; Y = linear * 0.5 + linear`. Do not algebraically move
the bias across scaling or activation, because the FP32 reference preserves the
declared order.

After a concrete remote failure, route the diagnostic through
`references/candidate-error-atlas.md`; do not browse other API families.
