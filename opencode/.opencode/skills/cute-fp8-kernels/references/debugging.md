# Debugging

Classify the earliest failure before changing configuration:

1. environment/import
2. tracing/JIT types
3. MLIR/PTX/`ptxas`
4. launch/resources
5. hang/pipeline
6. memory safety
7. numerics
8. timing

## Diagnostics

Normal Python `print` inside JIT code runs while tracing and is useful for static
types/layouts. Use `cute.printf` sparingly for device values, restricted to a
few threads.

Useful release-dependent controls:

```bash
CUTE_DSL_DEBUG=1
CUTE_DSL_LINEINFO=1
CUTE_DSL_LOG_TO_CONSOLE=1
CUTE_DSL_LOG_LEVEL=DEBUG
CUTE_DSL_PRINT_IR=1
CUTE_DSL_KEEP=ir,ptx,cubin,sass
CUTE_DSL_DUMP_DIR=/tmp/cute-dsl-dump
```

Compile options may include optimization, assertions, line info, target
architecture, retained PTX/cubin, and `ptxas` options. Executors may expose
`__mlir__`, `__ptx__`, or `__cubin__`. Confirm installed support. Debug
settings alter code/cache/timing; disable them for measurement.

## Common causes

Tracing/type:

- runtime value used to mutate/index static Python structure
- incompatible types returned by branches
- missing `Constexpr` specialization
- runtime-dependent layout type
- dynamic `break`, `continue`, early `return`, `global`, or `nonlocal`

Layout/copy:

- mathematical B `[K,N]` confused with physical `[N,K,L]`
- tiled copy partitions source/destination incongruently
- vector/pointer/stride alignment mismatch
- tile coordinate omits a mode
- TMA source, SMEM layout, tile, bytes, mask, or barrier disagree

Launch:

- block does not match warp roles
- invalid cluster/grid
- SMEM/TMEM/resource overflow
- wrong address space or stream
- Torch storage dies before synchronization

## Hang reduction

A tiny-case timeout usually means protocol mismatch. Reduce to one CTA, one K
tile, minimal stages, one-CTA MMA, non-persistent scheduling, simple epilogue.
Compare producer acquire/commit, consumer wait/release, expected arrivals,
transaction bytes, stage/phase transitions, multicast participants, prologue,
and drain. All threads in a collective must take compatible control flow.

## Numerical isolation

Add complexity in this order: one dense tile; epilogue; multiple K tiles;
staging; multiple output tiles; scales; clusters/two-CTA; persistent/tails.

Use:

- identity/ramps for transpose and coordinates
- one-hot K for scale-block selection
- ones for reduction length
- alternating signs for missing tiles
- distinct scale blocks for scale layout

Error growing with K suggests accumulator initialization/stage reuse. The first
K MMA initializes; later MMAs accumulate. Edge-only failures suggest unsupported
tails. Never change tolerance before explaining the error.

## Timing anomalies

Check that compilation/allocation/reference are excluded, warmup ran, the same
executor/stream is used, synchronization is valid, debug output is off, the
grid is nonempty, and the kernel writes the whole output.

Remote `stderr` and the first compiler/runtime error are primary evidence.
`profile_error` may be independent. Keep minimal reproducers self-contained,
deterministic, one-tile where possible, and failing nonzero.
