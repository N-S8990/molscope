"""Compare frozen GPR6 feature panels across two consecutive 100 ns windows.

Both windows come from the same 12 author trajectories. This is a temporal
stability check, not independent-replica or external validation. The 18
chemical features were frozen before evaluating the second window.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "runs"
OUT = ROOT / "panel_window_comparison"
PANELS = (
    ("39 Cα pilot", "public_12_replica_first_100ns", "public_12_replica_baseline_second_100ns",
     "supervised_model_summary.csv", "supervised_fold_assignments.csv"),
    ("63 motif", "public_12_replica_expanded_first_100ns", "public_12_replica_expanded_second_100ns",
     "supervised_model_summary.csv", "supervised_fold_assignments.csv"),
    ("18 chemical", "public_12_replica_chemical_first_100ns", "public_12_replica_chemical_second_100ns",
     "chemical_model_summary.csv", "chemical_fold_assignments.csv"),
    ("81 combined", "public_12_replica_augmented_first_100ns", "public_12_replica_augmented_second_100ns",
     "augmented_model_summary.csv", "augmented_fold_assignments.csv"),
)


def main() -> None:
    rows = []
    expected_folds = None
    for panel, first_dir, second_dir, summary_name, assignments_name in PANELS:
        for window, directory in (("0–99.9 ns", first_dir), ("100–199.9 ns", second_dir)):
            root = ROOT / directory
            fold = pd.read_csv(root / assignments_name).sort_values("trajectory_id").reset_index(drop=True)
            if expected_folds is None:
                expected_folds = fold
            elif not fold.equals(expected_folds):
                raise ValueError(f"Fold assignments differ for {panel}, {window}")
            summary = pd.read_csv(root / summary_name)
            if set(summary["Model"]) != {"logistic_regression", "random_forest"}:
                raise ValueError(f"Unexpected model set for {panel}, {window}")
            for record in summary.to_dict("records"):
                rows.append({
                    "Window": window, "Panel": panel, "Model": record["Model"],
                    "Trajectory_Mean_Accuracy_Mean": record["Trajectory_Mean_Accuracy_Mean"],
                    "Trajectory_Mean_Accuracy_Std_Across_3_Folds": record["Trajectory_Mean_Accuracy_Std"],
                    "ROC_AUC_Mean": record["ROC_AUC_Mean"],
                    "Replicas_Per_Class": 6, "Folds": 3,
                    "Interpretation": "same_trajectories_temporal_stability_not_external_validation",
                })
    table = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "gpr6_panel_window_comparison.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    panel_order = [panel for panel, *_ in PANELS]
    x = np.arange(len(panel_order))
    for ax, model, title in zip(
        axes,
        ("logistic_regression", "random_forest"),
        ("Logistic regression", "Random forest"),
        strict=True,
    ):
        selected = table[table["Model"] == model]
        first = [selected[(selected.Panel == panel) & (selected.Window == "0–99.9 ns")]
                 ["Trajectory_Mean_Accuracy_Mean"].iloc[0] for panel in panel_order]
        second = [selected[(selected.Panel == panel) & (selected.Window == "100–199.9 ns")]
                  ["Trajectory_Mean_Accuracy_Mean"].iloc[0] for panel in panel_order]
        bars_first = ax.bar(x - 0.18, first, width=0.36, label="0–99.9 ns", color="#3d79a9")
        bars_second = ax.bar(x + 0.18, second, width=0.36, label="100–199.9 ns", color="#dd8b32")
        for bar in [*bars_first, *bars_second]:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                    f"{bar.get_height():.3f}", ha="center", fontsize=8)
        ax.set_xticks(x, panel_order, rotation=15)
        ax.set_title(title)
        ax.set_ylim(0, 1.04)
        ax.axhline(0.5, linestyle="--", color="#888888", linewidth=0.8)
    axes[0].set_ylabel("Mean held-out trajectory accuracy")
    fig.legend([bars_first[0], bars_second[0]], ["0–99.9 ns", "100–199.9 ns"],
               loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=2)
    fig.suptitle("GPR6 feature panels across consecutive windows", y=0.995)
    fig.text(0.03, 0.01,
             "The same 12 trajectories and folds appear in both windows. "
             "Temporal consistency is not independent validation.",
             fontsize=9)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.80, bottom=0.24, wspace=0.12)
    fig.savefig(OUT / "gpr6_panel_window_accuracy.png", dpi=180)
    plt.close(fig)
    print(table[["Window", "Panel", "Model", "Trajectory_Mean_Accuracy_Mean"]].to_string(index=False))


if __name__ == "__main__":
    main()
