# Hopper configuration feasibility

Use this before compiling a candidate that changes the WGMMA CTA tile, cluster, input dtype, or TMA stage count:

```bash
kernel-evo cute check-hopper-config \
  --tile 128,128,64 --cluster 1,1 --stages 3 --dtype bf16
```

The tool checks the envelope verified by the local 4.2.1 Hopper GEMM. It explains:

- the BF16 or FP8 WGMMA instruction-K divisibility requirement;
- validated CTA M/N choices;
- validated power-of-two cluster shapes of at most four CTAs;
- A/B bytes per TMA stage;
- a conservative epilogue shared-memory upper bound;
- remaining space below the evaluator's opt-in shared-memory limit.

Use it as a rejection filter, not an occupancy oracle. A feasible configuration can still lose because of registers, barriers, bank conflicts, poor wave quantization, or an oversized epilogue. After compilation, replace the estimate with exact CUBIN resource usage and, for a promising candidate, an NCU occupancy/profile summary.

KernelEvo owns which proposal is evaluated. This tool only prevents an author from spending a candidate on a configuration already outside the verified legal/resource envelope.
