> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 13 — Target-free prototype pipeline

Run the integrated prototype with:

```bash
python -m molscope.run_discovery_pipeline \
  --features frame_features.csv \
  --manifest frame_manifest.csv \
  --out-dir discovery_run
```

It writes feature-quality diagnostics, PCA representation outputs, SAM-MFP
Pareto candidates, candidate conformational states and transitions, and a
researcher report. It is label-independent: supervised ML remains a separate
target-specific layer. Supply `--target-column condition` only for a valid
trajectory-level target; its grouped predictive results and LOFO ablation are
reported separately from target-free SAM-MFP prioritization.
