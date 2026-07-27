# Source provenance

The documentation pack is a condensed, task-routed summary. It is based on the
following NVIDIA CUTLASS `v4.6.1` primary sources:

- [tcgen05 MMA Programming Guide](https://github.com/NVIDIA/cutlass/blob/v4.6.1/media/docs/pythonDSL/mma_docs/tcgen05_programming.rst) — memory-space dataflow, partitions, fragments, and tcgen05 concepts.
- [Blackwell FP16 GEMM tutorial 0](https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_gemm/fp16_gemm_0.py) — connected TMA/UMMA/TMEM/epilogue structure.
- [Blackwell TMA tutorial](https://github.com/NVIDIA/cutlass/blob/v4.6.1/examples/python/CuTeDSL/cute/blackwell/tutorial/tutorial_tma/README.md) — `tma_partition`, `group_modes`, and mbarrier ordering.
- [CUTLASS Python utility guide](https://github.com/NVIDIA/cutlass/blob/v4.6.1/python/CuTeDSL/cutlass/utils/README.md) — `TmaInfo`, `TmemAllocator`, and buffer-pool contracts.
- [Pipeline helper source](https://github.com/NVIDIA/cutlass/blob/v4.6.1/python/CuTeDSL/cutlass/pipeline/helpers.py) — `Agent` and `CooperativeGroup` semantics.

Server-specific claims come from this project's neutral probes and are labeled
in `server-api-deltas.md`. Where an upstream signature conflicts with a
server-verified signature, the server delta is authoritative for evaluator
submissions.

No known task baseline or previous candidate is included in this pack.
