"""Evaluate expanded-panel sidechains as complete residue-level feature groups.

The standard MolScope report removes single columns. Here each selected
residue's chi1/chi2 sin/cos columns are removed together, using the same
trajectory-grouped folds and random-forest settings. Other features remain
single-column groups. These are predictive associations, not causal effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow


DEFAULT_RUN = Path(__file__).resolve().parents[1] / "runs/public_12_replica_expanded_first_100ns"
IDENTITY = ["trajectory_id", "condition", "frame_index", "frame_order", "time_ps"]


def group_for(feature: str) -> str:
    if not feature.startswith("SIDECHAIN_chi"):
        return feature
    parts = feature.split("__")
    if len(parts) != 4 or parts[-1] not in {"sin", "cos"}:
        raise ValueError(f"Unexpected sidechain feature name: {feature}")
    return f"SIDECHAIN_RESIDUE__{parts[1]}__{parts[2]}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    args = parser.parse_args()
    run = args.run
    frame = pd.read_csv(run / "gpr6_frame_features.csv")
    if len(frame) != 12_000 or not set(IDENTITY).issubset(frame):
        raise ValueError("Expected full 12,000-frame GPR6 feature table")
    features = frame.drop(columns=IDENTITY)
    if len(features.columns) != 63 or not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Expected 63 finite expanded-panel features")
    metadata = frame[["trajectory_id", "condition"]]
    mapping = {feature: group_for(feature) for feature in features}
    residue_groups = {group for group in mapping.values() if group.startswith("SIDECHAIN_RESIDUE__")}
    if len(residue_groups) != 9:
        raise ValueError("Expected nine selected sidechain residue groups")
    result = run_grouped_ml_workflow(features, metadata, "condition", MLWorkflowConfig(n_splits=3),
                                     feature_groups=mapping)
    old_folds = pd.read_csv(run / "supervised_fold_assignments.csv").sort_values("trajectory_id")
    new_folds = result["fold_assignments"].sort_values("trajectory_id")
    if not old_folds.reset_index(drop=True).rename(columns={"Target": "condition"}).equals(
        new_folds.reset_index(drop=True).rename(columns={"Target": "condition"})
    ):
        raise ValueError("Grouped ablation changed the held-out trajectory assignments")
    old_summary = pd.read_csv(run / "supervised_model_summary.csv")
    new_summary = result["model_summary"]
    for model in ("logistic_regression", "random_forest"):
        old = old_summary.loc[old_summary["Model"] == model, "Trajectory_Mean_Accuracy_Mean"].iloc[0]
        new = new_summary.loc[new_summary["Model"] == model, "Trajectory_Mean_Accuracy_Mean"].iloc[0]
        if not np.isclose(old, new, atol=1e-12):
            raise ValueError("Grouped ablation baseline differs from standard MolScope run")
    ranking = result["predictive_feature_ranking"]
    ranking.to_csv(run / "gpr6_grouped_ablation_summary.csv", index=False)
    result["lofo_fold_drops"].to_csv(run / "gpr6_grouped_ablation_folds.csv", index=False)
    plotted = ranking.assign(abs_change=lambda table: table["Trajectory_Mean_Accuracy_Drop_Mean"].abs())
    plotted = plotted.nlargest(15, "abs_change").iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, 6.5))
    changes = plotted["Trajectory_Mean_Accuracy_Drop_Mean"].to_numpy()
    colors = ["#21618c" if value >= 0 else "#d9822b" for value in changes]
    ax.barh(plotted["Feature_Group"], changes, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Grouped test accuracy drop after removing feature group")
    ax.set_title("GPR6 expanded panel: residue-level sidechain ablation")
    fig.subplots_adjust(left=0.47, right=0.97, top=0.91, bottom=0.12)
    fig.savefig(run / "gpr6_grouped_sidechain_ablation.png", dpi=180)
    plt.close(fig)
    print(f"Wrote {len(ranking)} feature groups; {len(residue_groups)} whole-residue sidechain groups")


if __name__ == "__main__":
    main()
