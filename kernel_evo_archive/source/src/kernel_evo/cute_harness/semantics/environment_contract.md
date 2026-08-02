# Exact environment contract (Python CuTe DSL 4.2.x)

Dialect: Python CuTe DSL only. Imports must come from `cutlass`, `cutlass.cute`, `cutlass.pipeline`, or version-matched Python helpers installed by `nvidia-cutlass-dsl`. CuTe C++ and the legacy CUTLASS Python operation API are outside this corpus.

Run `kernel-evo cute doctor` on the evaluator machine. Its report, not this checked-in validation snapshot, is authoritative for the installed version, GPU, CUDA runtime, tools, shared-memory capacity, and available symbols.

Agent runs persist `cute/author_capability.json`. The evaluator returns a compact identity and fingerprint beside every CuTe result. With `cute.capability_gate: true`, a different DSL version, target/native GPU architecture, or missing BF16/FP8 WGMMA feature invalidates the candidate before promotion. A coordinator snapshot is never treated as proof about a remote evaluator.

Hopper rules:

- Use `CUTE_DSL_ARCH=sm_90a`. `sm_90` omits architecture-accelerated features and cannot legally host the WGMMA atoms in this corpus.
- BF16 and FP8 WGMMA accumulate into `cutlass.Float32` unless a deliberately validated numerical contract says otherwise.
- TMA descriptors require compatible layouts, at least 16-byte alignment on the contiguous dimension, and a transaction-byte count matching the shared-memory tile.
- Cluster launch, multicast, and WGMMA are Hopper-specific. Do not silently reuse these patterns for SM80, SM100, or SM120.
- `cute.compile` is a separate compilation step. Cache the returned executor by every static property that changes generated code: dtype, layout/leading mode, tile, cluster shape, stage count, and any other `Constexpr`.
- Mark only intended runtime layout modes dynamic. For matrix tensors, preserve the leading mode: `mark_layout_dynamic(leading_dim=...)`.

The validated 4.2.1 environment supports `CUTE_DSL_KEEP_IR=1`. Newer CUTLASS documentation may mention `CUTE_DSL_KEEP`, `CUTE_DSL_DUMP_DIR`, or `compiled.__ptx__`; do not assume those newer interfaces exist. Use the version-aware harness runners.
