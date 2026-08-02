# KernelEvo agent policy

For kernel optimization, use the `kernelevo` skill and the `kernel-evo` harness. KernelEvo owns the
algorithm, archive, evaluation, profiling, and reports. Delegate only one candidate patch per island,
keep islands isolated, wait at each evaluation barrier, and never promote an unevaluated candidate.
