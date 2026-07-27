# Scalar CuTe FP8 API context

This offline pack contains only neutral CuTe DSL mechanics for correctness-first
scalar kernels. It intentionally contains no convolution, transposed
convolution, grouping, padding, or task-shaped indexing recipe.

Read [`scalar-kernels.md`](scalar-kernels.md) before editing. `TASK.md` and
`task.json` remain the authority for the operation, shapes, precision contract,
and evaluator ABI.

The goal is to remove namespace guessing while leaving algorithm construction
to the candidate.
