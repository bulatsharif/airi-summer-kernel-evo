# FP4 candidate error atlas

Use the earliest concrete diagnostic. Change one cause at a time.

## Packed E2M1

| Symptom or diagnostic | Likely cause | Correction |
|---|---|---|
| Positive values are correct but negatives are wrong | Sign bit ignored or treated as integer sign extension | Decode magnitude from `nibble & 0x7`, then negate when `nibble & 0x8`. |
| Values `3`, `4`, or `6` are wrong | Nibble treated as linear fixed point | Use the exact E2M1 magnitude table. |
| Every pair is swapped | Wrong nibble order | Follow the task contract; for low-first use `byte & 0xF` then `byte >> 4`. |
| High values include bits from the low nibble | Shift/mask performed on the wrong value | Convert the byte to `Int32`; mask low and shift high from the original byte. |
| Alternating output elements are missing | One output written per packed byte | Write logical indices `2*i` and `2*i+1`. |
| Races or duplicated pairs | One thread assigned per logical element while both read/write a shared pair | Use one thread per byte, or give each thread a unique logical output. |
| Error is an exact constant factor | Scale omitted, inverted, or applied twice | Write the public equation and apply its scale once. |
| Last element is out of bounds | Odd logical length treated as two complete values | Predicate the final high-nibble store if the contract permits odd length. |
| `cutlass.cute has no attribute thread_layout` | Guessed helper from another API | Use `cute.arch.thread_idx`, `block_idx`, and `block_dim`. |
| Bitwise operation fails during tracing | Operation applied to a tensor object or FP value | Load one element and convert it to `cutlass.Int32` first. |
| Local check reports zero kernels | Launchable function is not exactly `@cute.kernel` | Restore the device kernel decorator; helpers may remain `@cute.jit`. |
| Local check reports a missing required `cute.gemm` | Scalar unpack code used for a native-MMA task | Implement the task's block-scaled MMA contract; unpacking cannot satisfy it. |

## Native block-scaled FP4

| Symptom or diagnostic | Likely cause | Correction |
|---|---|---|
| Missing `Float4...`, MMA, or scale-layout attribute | Symbol guessed from a different CUTLASS release | Restore the installed FP4 example's exact import and spelling. |
| Scale-layout congruence or shape failure | Logical row-major scales passed where an MMA-specific layout is required | Use the recipe's scale layout conversion and padding unchanged. |
| Correct with all-one scales, wrong otherwise | Scale tensors ignored, swapped, inverted, or indexed by the wrong K block | Test distinct adjacent SFA/SFB blocks and audit the public scale equation. |
| Error grows at each K-block boundary | Scale iterator is not advanced with the operand K stage | Couple scale stage/count transitions to the matching A/B stage. |
| Large constant-factor error | Inner block scale or outer tensor scale missing/doubled | Separate the two scale levels and compare with the exact task equation. |
| Timeout after using `range_constexpr` over K | Large loop fully unrolled at trace time | Use `cutlass.range` for runtime tile loops. |
| Illegal instruction or unspecified launch failure | Unsupported tile/type/CTA group or broken pipeline/TMEM lifetime | Restore the smallest release-valid FP4 recipe before changing one parameter. |
| Numerics pass but no FP4 speedup | Values were unpacked and computed through a wider path | Verify generated native FP4 tensor-core instructions before claiming success. |

## Retry discipline

1. Preserve any candidate that reached remote execution.
2. Fix the first diagnostic without rewriting unrelated working code.
3. Never retry an identical timeout, illegal access, or launch failure.
4. Reduce native MMA failures to one CTA, one output tile, one K block, and the
   release-valid stage count.
5. Stop after the first harness PASS unless optimization was requested.
