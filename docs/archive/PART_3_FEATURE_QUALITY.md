> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 3: Label-Independent Feature Quality

The Part 3 prototype evaluates an existing feature matrix without a class label or predictive model. It is designed to answer whether a measurement is worth carrying forward, not whether it predicts a specific outcome.

## Run on the existing legacy output

From the repository root:

```bash
python -m molscope.audit_features \
  --features results/raw_features.csv \
  --legacy-run-metadata results/run_metadata.json \
  --out-dir results/feature_quality
```

The legacy adapter reconstructs frame identity from the old pipeline's recorded input ordering. It is only a bridge: future extraction must write a frame manifest directly.

## Run on a new workflow

Provide a manifest CSV with one row per feature-matrix row and these required columns:

```text
trajectory_id,replica_id,ensemble_id
```

Then run:

```bash
python -m molscope.audit_features \
  --features my_features.csv \
  --manifest my_frame_manifest.csv \
  --feature-costs measured_feature_costs.csv \
  --block-size-frames 20 \
  --pareto-bootstrap-samples 100 \
  --out-dir feature_quality_results
```

## Outputs

- `feature_quality_metrics.csv`: all diagnostics.
- `trajectory_level_features.csv`: values that are constant within each trajectory and therefore must not be treated as independent frames.
- `pareto_feature_candidates.csv`: frame-level features that are non-dominated across the prototype's four objectives.
- `pareto_quality_tradeoffs.png`: sampling efficiency versus replica consistency.
- `feature_quality_report.json`: scope and limitations of the run.

## Prototype criteria

All criteria are label-independent.

- **Valid fraction**: numerical coverage; missing values are not changed to zero.
- **Effective sample-size ratio**: a simple autocorrelation-based estimate of how many frames are informative.
- **Replica consistency**: agreement of feature distributions between independent replica trajectories within an ensemble, using normalized Wasserstein distance.
- **Occupancy entropy**: a descriptive measure of non-degenerate dynamic variation. It is not a claim of physical importance.
- **Uniqueness**: one minus the maximum absolute Spearman correlation with another frame-level feature.
- **Block standard error**: a contiguous-block, within-trajectory uncertainty diagnostic. It is not an independent-replica confidence interval.
- **Measured cost**: optional per-feature elapsed seconds, supplied in a CSV with `Feature` and `Elapsed_Seconds`. Missing cost is reported as `not_provided`, never estimated as zero.
- **Pareto membership frequency**: optional stability diagnostic from resampling complete trajectories within each ensemble. It measures sensitivity to the available replicas and does not turn correlated frames into independent replicates.

The Pareto candidate table does **not** rank features with a weighted total score. It returns features for which no other eligible feature is at least as good on all four criteria and better on at least one.

## Known limits, intentionally reported

- This first prototype does not yet measure per-analyzer computational cost.
- It does not yet bootstrap the Pareto front or report rank stability.
- It cannot repair legacy feature semantics where unavailable quantities were written as zero; that is addressed by the Part 2 feature-status contract and the later extractor migration.
- It is an audit and prioritization step, not an ML feature-importance analysis. ML LOFO and group-ablation accuracy drops remain a later part.
