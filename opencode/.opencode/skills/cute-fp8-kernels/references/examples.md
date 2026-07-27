# Local Blackwell GEMM recipes

This is a compact implementation map, not a complete kernel. It captures the
architecture and constraints that should be known before inspecting exact APIs
in the installed package.

Knowledge baseline: CUTLASS 4.6.1, commit
`e05f953a5b3d38adc240df2ff928e0421c2abba3`.

## Contents

- [Where to inspect installed examples](#where-to-inspect-installed-examples)
- [Dense FP8 GEMM](#dense-fp8-gemm)
  - [Host/JIT setup](#hostjit-setup)
  - [Device kernel](#device-kernel)
  - [Baseline constraints](#baseline-constraints)
- [MXFP8 block-scaled GEMM](#mxfp8-block-scaled-gemm)
- [Tensor creation and execution](#tensor-creation-and-execution)
- [Starting configurations](#starting-configurations)
- [Choosing a starting point](#choosing-a-starting-point)
- [Adaptation rules](#adaptation-rules)
- [Recipe checklist](#recipe-checklist)

## Where to inspect installed examples

When CUTLASS source/examples are included with the installed package, look for
paths matching:

```text
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm.py
examples/python/CuTeDSL/cute/blackwell/kernel/dense_gemm/dense_gemm_persistent.py
examples/python/CuTeDSL/cute/blackwell/kernel/blockscaled_gemm/
```

The installed tree is used only to confirm exact imports, helper signatures,
shared-storage declarations, launch arguments, and release-specific constraints.
The design and acceptance contract live in these local references.

If copying a substantial source fragment, preserve its upstream license notice.

## Dense FP8 GEMM

Use dense GEMM when A and B are ordinary E4M3FN or E5M2 tensors and the
operation does not provide block scale-factor tensors.

### Host/JIT setup

1. Accept A as `M,K,L`, B physically as `N,K,L`, and C as `M,N,L`.
2. Derive A/B major modes and C layout from the tensors.
3. Require A and B to have the same FP8 type for the baseline kernel.
4. Construct the Blackwell tiled MMA with:
   - A/B type and major modes
   - accumulator type
   - one-CTA or two-CTA group
   - M/N MMA tile
5. Derive the instruction K size, CTA tile, cluster layout, and multicast counts.
6. Compute staged A/B shared-memory layouts and an optional C epilogue layout.
7. Create tiled TMA atoms and tensors for A and B.
8. Compute the total A+B TMA transaction bytes.
9. Compute the output grid from C shape and CTA tile.
10. Launch 128 threads per CTA with the selected cluster shape.

The relevant baseline API names are:

```python
sm100_utils.make_trivial_tiled_mma(...)
sm100_utils.make_smem_layout_a(...)
sm100_utils.make_smem_layout_b(...)
cute.nvgpu.make_tiled_tma_atom_A(...)
cute.nvgpu.make_tiled_tma_atom_B(...)
kernel(...).launch(grid=..., block=[128, 1, 1], cluster=..., stream=...)
cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream)
```

Confirm these names against the installed version.

### Device kernel

The device kernel normally assigns separate warp roles:

- TMA warp: move A/B tiles from GMEM to staged SMEM and signal barriers.
- MMA warp: wait for a full stage, issue `tcgen05.mma`, and release the stage.
- Epilogue warps: load accumulators from TMEM, convert/apply the epilogue, and
  store C.

The core sequence is:

```text
initialize cluster and pipeline barriers
allocate staged A/B shared memory
allocate tensor-memory accumulator columns
partition global and shared tensors
for each K tile:
    producer acquires an empty stage
    producer issues TMA A/B copies
    consumer waits for the full stage
    consumer issues tcgen05 MMA
    consumer releases the stage
wait for MMA completion
load accumulator TMEM -> registers
convert/apply epilogue
store registers -> GMEM, optionally through SMEM/TMA
release tensor memory
```

### Baseline constraints

- A and B use the same element type.
- Floating-point accumulation supports FP32; dense FP8 also supports FP16
  accumulation in the baseline.
- One-CTA MMA tile M is 64 or 128.
- Two-CTA MMA tile M is 128 or 256.
- MMA tile N is 32 through 256 in steps of 32.
- Cluster M/N are positive powers of two with total size at most 16.
- Two-CTA mode requires cluster M to be a multiple of two.
- A/B/C contiguous dimensions require at least 16-byte alignment. For FP8 this
  means a multiple of 16 elements.
- Without a TMA C store, out-of-bounds output tiles are not supported.

Start with a one-CTA, non-persistent configuration unless the task requires
otherwise. Add persistent scheduling only after the simpler kernel is correct.

## MXFP8 block-scaled GEMM

Use the block-scaled recipe only when the task supplies quantized A/B plus SFA
and SFB.

Baseline MXFP8 configuration:

- A/B: E4M3FN or E5M2
- SFA/SFB: E8M0
- scale vector size: 32 K-elements
- accumulation: FP32
- logical scale shapes:
  - SFA: `M,ceil_div(K,32),L`
  - SFB: `N,ceil_div(K,32),L`

The physical scale tensors use a block-scaled layout/swizzle produced by the
CUTLASS block-scaled layout utilities. Do not store them as plain row-major
2-D arrays merely because their logical shapes match.

Compared with dense GEMM, add:

1. SFA/SFB global and shared-memory layouts.
2. TMA loads and pipeline accounting for scale factors.
3. SMEM-to-TMEM scale-factor copies using `tcgen05.cp`.
4. Block-scaled MMA descriptors that consume A, B, SFA, and SFB.
5. A reference that applies the exact scale layout and direction.

Baseline block-scaled M/N tile choices are more restricted:

- MMA tile M: 128 or 256
- MMA tile N: 64, 128, 192, or 256
- scale-factor multicast limits cluster M and N to at most 4

Do not use the SM103 FP4 Ultra recipe as an FP8 recipe merely because both use
block scaling.

## Tensor creation and execution

Keep input creation and kernel execution distinct:

```python
# Host tensors may be Torch tensors.
# Convert or pass them through the CuTe runtime expected by the installed build.

gemm = DenseGemmKernel(
    acc_dtype=cutlass.Float32,
    use_2cta_instrs=False,
    mma_tiler_mn=(128, 128),
    cluster_shape_mn=(1, 1),
    use_tma_store=True,
)

compiled_gemm = cute.compile(gemm, a_tensor, b_tensor, c_tensor, stream)
compiled_gemm(a_tensor, b_tensor, c_tensor, stream)
```

This fragment describes object construction and JIT reuse. `DenseGemmKernel`
must still contain the full CuTe DSL host setup and device kernel; it is not a
built-in CUTLASS class.

Keep host setup in this order:

1. allocate or receive Torch storage
2. form CuTe tensor views with explicit physical layouts
3. construct the kernel object/configuration
4. run `can_implement`
5. convert the current CUDA stream
6. compile the JIT entry once
7. launch and synchronize
8. compute/check the reference
9. warm up and measure only after checks pass

## Starting configurations

These are conservative search seeds, not guaranteed optima:

| Operation | MMA tile M,N | Cluster M,N | Mode | Scheduling |
|---|---:|---:|---|---|
| Dense FP8 baseline | `128,128` | `1,1` | one CTA | non-persistent |
| Dense FP8 wider N | `128,256` | `1,1` | one CTA | non-persistent |
| Dense FP8 cooperative | release-valid | `2,1` | two CTA | non-persistent |
| MXFP8 baseline | `128,128` | `1,1` | release-valid | non-persistent |

Derive stage count from the installed helper/resource calculation. A hardcoded
stage count from another tile may overflow SMEM or break the pipeline storage
layout.

Start on shapes divisible by CTA and instruction requirements. Only add tails
when the requested contract requires them and the selected epilogue/copy path
supports them.

## Choosing a starting point

- Dense FP8, first implementation: one-CTA dense recipe.
- Dense FP8, known-correct but underutilized: try two-CTA or persistent variants.
- MXFP8 with SFA/SFB: block-scaled recipe.
- FP8 input with only tensorwise scales: dense recipe plus the explicitly
  specified pre/post scaling operation.
- Unknown installed API: inspect package examples or symbols locally, then make
  the smallest compatibility change needed.

## Adaptation rules

When adapting a recipe:

1. preserve operand physical layouts
2. preserve the helper that creates the tiled MMA
3. preserve all SMEM/TMEM allocation calculations
4. preserve producer/consumer role counts and pipeline APIs
5. preserve TMA transaction byte calculation
6. preserve cluster multicast construction
7. preserve first-K versus accumulate MMA behavior
8. preserve epilogue TMEM release ordering
9. change one configuration family at a time

If the task only changes the FP8 format, first check whether the same operation
class supports it. If the task changes dense to block-scaled semantics, treat it
as a different mainloop rather than a datatype substitution.

Persistent scheduling adds a work-tile scheduler and grid-shape decision.
Port it only after the non-persistent output and pipeline are correct.

Two-CTA MMA changes instruction shape, cluster requirements, TMEM ownership, and
potentially epilogue participation. Do not enable it as an isolated boolean
without rebuilding the dependent configuration.

## Recipe checklist

- [ ] Installed example/release matches the target closely enough.
- [ ] Mathematical and physical A/B/C layouts are recorded.
- [ ] Dense versus MXFP8 mainloop matches the operation.
- [ ] Tiled MMA supports the FP8 and accumulator types.
- [ ] CTA tile, cluster, instruction mode, and stages are jointly valid.
- [ ] TMA descriptor layouts and transaction bytes cover every loaded tensor.
- [ ] Warp roles and pipeline state transitions match.
- [ ] First K tile initializes; later K tiles accumulate.
- [ ] TMEM is not released before epilogue consumers finish.
- [ ] Tail constraints are enforced.
- [ ] Kernel class code is present; no fictitious built-in wrapper is assumed.
