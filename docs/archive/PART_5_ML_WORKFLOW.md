> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 5: Generic, Replica-Aware ML Workflow

This workflow is optional. It measures **predictive importance** for a user-supplied target; it does not replace label-independent feature quality or establish causation.

## Inputs

- Numeric feature CSV: one row per frame.
- Manifest CSV: one aligned row per frame, with `trajectory_id` and a user-selected target column.
- Optional feature-group CSV with columns `Feature` and `Feature_Group` for biological/group ablation.

Each trajectory must have one target value. Every class must have at least as many independent trajectories as requested outer folds.

## Run

```bash
python -m molscope.run_ml_workflow \
  --features selected_features.csv \
  --manifest frame_manifest.csv \
  --target-column condition \
  --feature-groups feature_groups.csv \
  --folds 3 \
  --out-dir ml_results
```

## Safeguards

- Outer folds use `StratifiedKFold` on one record per trajectory, then expand to frames. Each class is represented in every fold even when trajectory lengths differ.
- Logistic scaling occurs inside its model pipeline, fitted only on each training fold.
- Invalid feature values cause an error; the workflow never silently imputes unavailable features.
- LOFO/group ablation retrains on every outer fold.
- The primary LOFO endpoint is **trajectory-mean accuracy drop**, not raw frame accuracy alone.
- Importance output includes outer-fold bootstrap intervals, per-fold ranks, rank standard deviation, and positive-drop frequency.

## Outputs

- `fold_assignments.csv`: trajectory, target, and held-out fold for auditing/reproducing splits
- `outer_fold_metrics.csv`
- `model_summary.csv`
- `lofo_fold_drops.csv`
- `lofo_importance.csv`
- `predictive_feature_ranking.csv`: the same grouped LOFO result with explicit
  evidence and interpretation fields. It is a predictive-association ranking,
  not a causal ranking.
- `lofo_trajectory_accuracy_drop.png`
- `ml_report.json`

For publication-grade conclusions, combine these predictive results with Part 3 quality/reproducibility metrics and inspect individual trajectory behavior.

## Candidate-state explanatory ranking

The configured project runner also writes
`state_explanatory_feature_ranking.csv`. It ranks features by effect size among
candidate states inferred from the same frames. This is intentionally separate
from predictive ranking and SAM-MFP: it describes cluster separation only and
does not validate states, generalize to a supplied label, or establish a
mechanism.

Use `--seed` (default `42`) to control splitting and model randomness. Missing trajectory IDs and duplicate feature names are rejected before fitting.
