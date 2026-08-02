# BF16 predicated add / residue path

This executable Python CuTe DSL 4.2.1 example demonstrates the smallest general KernelEvo pattern: dynamic 2D layouts, thread/value ownership, register fragments, identity-tensor predication, and one cached `cute.compile` executor.

The retained H200 SASS shows that its per-element predicate scalarizes BF16 global loads/stores. That makes it the residue/correctness half of a minimal pair, not the bandwidth fast path. Use `../bf16_vector_add_aligned` when an outer dispatch proves full 16-byte-aligned 1024-element tiles and 128-bit codegen is required.

It intentionally does not use WGMMA or TMA. Use it for elementwise fusion and copy/layout mutations, not as a GEMM template.

```bash
CUTE_DSL_ARCH=sm_90a python kernel.py
```

Ragged element counts are predicated, but a predicate is not an alignment proof for arbitrary sliced tensors. Do not describe this path as vectorized unless a changed implementation passes an explicit codegen contract.
