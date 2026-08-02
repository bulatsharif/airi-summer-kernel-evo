# Aligned BF16 vector-add fast path

This is the no-residue half of the BF16 copy minimal pair. It requires contiguous, 16-byte-aligned inputs whose flattened element count is divisible by the 1024-element TV tile. Because every participating lane owns a complete eight-BF16 vector, its generated code is required to contain 128-bit global loads and stores.

Use it when the task contract proves a dominant aligned shape or when building the fast branch of a fast-path-plus-residue design. Do not use it as the general fallback; `../bf16_vector_add` is the predicated residue example.

```bash
kernel-evo cute check --arch sm_90a \
  --contract expected_codegen.yaml -- python kernel.py
```

The important transition from the residue example is the removal of per-element predicates from the copy. The price is an explicit divisibility/alignment precondition that must be guarded outside this kernel.
