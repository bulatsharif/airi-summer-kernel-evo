# Diagnostic workflow and observed failure classes

Tier IV adds diagnostic guidance derived from 200 pre-study AIRI and KernelEvo
error records. The records contained 65 normalized signatures. They are grouped
here by mechanism, without task paths, shapes, candidate code, or solution
choices.

## Observed high-frequency classes

| Normalized class | Records | Interpretation |
| --- | ---: | --- |
| CUDA invalid value | 46 | launch, resource, descriptor, or configuration rejected |
| required JIT entry point missing | 29 | public interface or decorator was changed |
| numerical validation families | 30+ | output ran but did not meet the fixed oracle |
| unspecified launch failure | 12 | asynchronous device failure surfaced later |
| operation creation failure | 7 | incompatible type/layout/instruction combination |
| IR verification failure | 4 | illegal compiler IR or unsupported lowering combination |
| repeated API hallucinations | many | plausible symbol used in wrong namespace or release |

Frequency helps prioritize diagnosis; it does not prove the cause of a new
failure.

## Always start with the earliest deterministic error

Compiler and runtime output often contains cascades. Use this order:

1. public candidate policy and entry point;
2. first Python/import/attribute error;
3. first argument-binding/type error;
4. first IR or operation-creation error;
5. launch configuration;
6. asynchronous timeout or memory failure;
7. numerical validation;
8. performance.

Do not optimize or restructure code while an earlier gate fails.

## Minimal diagnostic record

For each attempt, retain:

```text
candidate change:
earliest failing stage:
exact first diagnostic:
object type at failing call:
expected parameter roles:
actual parameter roles:
one hypothesis:
one next change:
result after that change:
```

This prevents repeated, unrelated edits.

## Failure localization

| Stage | Evidence |
| --- | --- |
| Parse/import | Python exception before tracing |
| Binding | signature/decorator complaint naming the entry point |
| Tracing | Python type, missing attribute, proxy/static error |
| Lowering | operation creation, IR verification, PTX assembly |
| Launch | invalid value or resource/configuration diagnostic |
| Execution | timeout, illegal access, unspecified launch failure |
| Validation | finite output differs from reference |
| Timing | correct output but unstable or slow measurement |

An error at a synchronization call may belong to the preceding asynchronous
launch. Insert a diagnostic synchronization only to localize; remove it from
the final measured path.

## Reduction strategy

Reduce the failing region without replacing the design:

- keep the public interface unchanged;
- retain one use of the suspect API;
- remove downstream consumers;
- use one legal work item or stage when the protocol permits;
- preserve address spaces and object kinds;
- compile or launch again.

Reduction is for identifying the first invalid boundary. It is not permission
to replace a task with a framework operation.

## Change discipline

After one concrete error:

1. classify it;
2. inspect the local API signature and actual object kinds;
3. change only the first mismatch;
4. rerun the same check;
5. keep the change only if the error advances or disappears.

If an identical candidate produces the same deterministic failure, resubmitting
it adds no information.

## Do not hide failures

Never:

- catch a compiler or launch failure and report success;
- weaken the harness comparison;
- skip required cases;
- leave output uninitialized;
- substitute a framework implementation;
- increase timeout without explaining a tiny-case hang;
- claim performance for an invalid output.
