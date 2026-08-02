# Python CuTe DSL harness policy

- This directory covers only the Python CuTe DSL distributed as `nvidia-cutlass-dsl` and imported through `cutlass.cute`.
- Do not add CuTe C++, legacy CUTLASS Python operation-API examples, or examples from a different installed DSL version to a retrieval bundle.
- Treat the installed package and its source as the API authority. Use `kernel-evo cute lookup SYMBOL` before introducing an unfamiliar symbol.
- In an island packet, read `CUTE_HARNESS.md` first. It is a routing map: open only the card or manifest justified by the current decision, and open a full reference kernel only for the named matching construction.
- Do not bulk-read the catalog. Deeper context is available for navigation, not mandatory prompt material.
- Hopper architecture-accelerated instructions require `CUTE_DSL_ARCH=sm_90a`; `sm_90` is not equivalent.
- A genuine FP8 optimization must use an FP8 compute/data-movement path. Merely converting a value to float8 and back is not FP8 acceleration.
- Keep executable examples paired with a manifest, stated invariants, correctness cases, and expected code-generation families.
- Run memory checking before racecheck, initcheck, or synccheck after changing memory movement or synchronization.
- Record correctness, timing, resource usage, and the accept/reject decision in structured experiment memory.
- KernelEvo owns evaluation, sanitizer sequencing, profiling, archive state, and promotion. Author turns may use API/layout/config probes but must not run the full benchmark.
