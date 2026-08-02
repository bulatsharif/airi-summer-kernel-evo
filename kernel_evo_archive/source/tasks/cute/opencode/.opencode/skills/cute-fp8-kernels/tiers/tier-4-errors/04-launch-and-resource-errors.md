# Launch and resource errors

## CUDA invalid value

This high-frequency error is broad. First separate:

1. launch geometry;
2. cluster geometry;
3. dynamic shared memory;
4. kernel attributes and occupancy hints;
5. descriptor or pointer arguments;
6. architecture/instruction availability.

Inspect the fully bound launch and generated resource requirements. Do not
change all dimensions simultaneously.

## Invalid grid or block

Verify:

- each dimension is an integer/DSL integer accepted by the launch;
- dimensions are positive where work is launched;
- tuple/list rank is at most three and canonicalizes correctly;
- block threads match warp-role assumptions;
- the grid covers the intended work space;
- cluster grouping agrees with grid dimensions.

An empty grid may appear fast while doing no work. Output validation must catch
it.

## Kernel argument binding versus launch configuration

A bound kernel object already holds declared kernel arguments. `.launch`
receives launch configuration, not the kernel tensors again.

Conversely, calling `.launch` on the undecorated function or an unbound kernel
omits required arguments.

## Shared-memory overflow

Compute:

```text
sum of every live SMEM allocation
+ staged operand bytes
+ barrier/pipeline storage
+ epilogue storage
+ alignment padding
```

If mutually exclusive branches do not opt into merged allocation, their
allocations can be counted additively.

Changing tile or stages requires recomputing all dependent layouts and launch
attributes.

## Register or occupancy pressure

Symptoms:

- compilation succeeds but launch/resources fail;
- a larger tile is slower;
- increasing stages reduces residency;
- `ptxas` reports high register use or spills.

Identify which live fragments and unrolling decisions increased state. Do not
assume shared memory is the only occupancy constraint.

## TMEM allocation/configuration failure

Check:

- requested columns follow from fragment layout;
- allocation owner and CTA group agree;
- all required participants wait for allocation;
- no previous allocation remains live;
- release occurs after final use;
- instruction and epilogue expect the same TMEM layout.

TMEM resources are distinct from shared memory and registers.

## Cluster launch failure

Verify:

- preferred and fallback cluster tuples have exactly three dimensions;
- grid can be grouped under the chosen cluster policy;
- architecture supports the requested cluster mode;
- multi-CTA instruction mode matches cluster cooperation;
- multicast masks stay within the cluster;
- cluster barriers have all required participants.

Test the same operation without cluster-specific behavior only to isolate the
cluster mechanism, not as a final substitute when the implementation depends
on it.

## Alignment errors

Alignment has two parts:

1. base pointer alignment;
2. stride/offset preservation for every selected tile.

An aligned base does not make `base + offset` aligned for a vector operation.
Derive the byte offset for each operand and stage.

`assumed_align` tells the compiler a fact; a false promise can turn a safe
scalar access into an illegal vector access.

## Unsupported architecture or instruction

An importable symbol may still lower to an instruction unavailable for the
actual target or dtype/layout combination.

Check:

- target compute capability selected by compilation;
- instruction family;
- operand types;
- CTA group;
- layout/shape constraints.

Do not install another package version inside a candidate to work around the
study environment.

## Resource-isolation sequence

When launch fails after construction:

1. record grid/block/cluster and dynamic SMEM;
2. record compiler resource output if available;
3. isolate optional cluster or epilogue attributes;
4. keep the same instruction/data contract;
5. change one resource dimension;
6. rerun a real launch.

Tracing alone cannot validate launch resources.
