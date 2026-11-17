"""Join frozen 39/63-feature GPR6 panel results across two matched time windows.

The same replicas occur in both windows: these tables describe temporal
stability and are not independent validation or causal feature rankings.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
PANELS = {
    ("39", "0-99.9 ns"): RUNS / "public_12_replica_first_100ns",
    ("63", "0-99.9 ns"): RUNS / "public_12_replica_expanded_first_100ns",
    ("39", "100-199.9 ns"): RUNS / "public_12_replica_baseline_second_100ns",
    ("63", "100-199.9 ns"): RUNS / "public_12_replica_expanded_second_100ns",
}
OUT = RUNS / "gpr6_cross_window"
OUT.mkdir(parents=True, exist_ok=True)


def read(run: Path, filename: str) -> pd.DataFrame:
    return pd.read_csv(run / filename)


def checked_pair(filename: str, key: str, panel: str = "63") -> pd.DataFrame:
    first = read(PANELS[(panel, "0-99.9 ns")], filename)
    second = read(PANELS[(panel, "100-199.9 ns")], filename)
    assert len(first) == len(second) and first[key].is_unique and second[key].is_unique
    assert set(first[key]) == set(second[key])
    return first.merge(second, on=key, validate="one_to_one", suffixes=("_first", "_second"))


folds = [read(p, "supervised_fold_assignments.csv").sort_values("trajectory_id").reset_index(drop=True)
         for p in PANELS.values()]
assert all(f.equals(folds[0]) for f in folds[1:]), "Trajectory folds differ between panels/windows"

performance = []
for (panel, window), run in PANELS.items():
    metrics = read(run, "supervised_outer_fold_metrics.csv")
    for model, group in metrics.groupby("Model", sort=True):
        assert len(group) == 3
        row = {"Panel_Features": int(panel), "Window": window, "Model": model,
               "Mean_Trajectory_Accuracy": group["Trajectory_Mean_Accuracy"].mean(),
               "Min_Fold_Accuracy": group["Trajectory_Mean_Accuracy"].min(),
               "Max_Fold_Accuracy": group["Trajectory_Mean_Accuracy"].max(),
               "Fold_Accuracies": ";".join(f"{x:.6f}" for x in group.sort_values("Fold")["Trajectory_Mean_Accuracy"]),
               "Interpretation": "same_12_replicas_temporal_repeat_not_independent_validation"}
        performance.append(row)
pd.DataFrame(performance).to_csv(OUT / "gpr6_panel_performance.csv", index=False)

condition = checked_pair("gpr6_feature_condition_comparison.csv", "Feature")
lofo = checked_pair("supervised_lofo_importance.csv", "Feature_Group")
assert len(condition) == len(lofo) == 63
lofo = lofo.rename(columns={"Feature_Group": "Feature"})
condition = condition.merge(lofo[["Feature", "Trajectory_Mean_Accuracy_Drop_Mean_first",
                                  "Trajectory_Mean_Accuracy_Drop_Mean_second"]],
                            on="Feature", validate="one_to_one")
q_first = next(x for x in condition.columns if x.startswith("BH_Q_") and x.endswith("_first"))
q_second = next(x for x in condition.columns if x.startswith("BH_Q_") and x.endswith("_second"))
condition = condition.rename(columns={
    "Three_H_Minus_No_Added_Inverse_Agonist_first": "Condition_Effect_Angstrom_Or_Native_Units_First",
    "Three_H_Minus_No_Added_Inverse_Agonist_second": "Condition_Effect_Angstrom_Or_Native_Units_Second",
    q_first: "Condition_BH_Q_First", q_second: "Condition_BH_Q_Second",
    "Trajectory_Mean_Accuracy_Drop_Mean_first": "RF_Single_Feature_LOFO_Drop_First",
    "Trajectory_Mean_Accuracy_Drop_Mean_second": "RF_Single_Feature_LOFO_Drop_Second",
})
condition["Effect_Same_Direction"] = np.sign(condition["Condition_Effect_Angstrom_Or_Native_Units_First"]) == np.sign(condition["Condition_Effect_Angstrom_Or_Native_Units_Second"])
condition["BH_Q_Below_0p05_Both"] = (condition["Condition_BH_Q_First"] < .05) & (condition["Condition_BH_Q_Second"] < .05)
condition["RF_LOFO_Positive_Both"] = (condition["RF_Single_Feature_LOFO_Drop_First"] > 0) & (condition["RF_Single_Feature_LOFO_Drop_Second"] > 0)
condition["Interpretation"] = "exploratory_same_replicas;LOFO_is_redundancy_sensitive_not_causality"
condition[["Feature", "Condition_Effect_Angstrom_Or_Native_Units_First", "Condition_BH_Q_First",
           "RF_Single_Feature_LOFO_Drop_First", "Condition_Effect_Angstrom_Or_Native_Units_Second",
           "Condition_BH_Q_Second", "RF_Single_Feature_LOFO_Drop_Second",
           "Effect_Same_Direction", "BH_Q_Below_0p05_Both", "RF_LOFO_Positive_Both",
           "Interpretation"]].sort_values("Feature").to_csv(OUT / "gpr6_expanded_feature_stability.csv", index=False)

groups = checked_pair("gpr6_grouped_ablation_summary.csv", "Feature_Group")
assert len(groups) == 40
groups = groups.rename(columns={
    "Trajectory_Mean_Accuracy_Drop_Mean_first": "RF_Group_LOFO_Drop_First",
    "Trajectory_Mean_Accuracy_Drop_Mean_second": "RF_Group_LOFO_Drop_Second",
})
groups["Positive_Both"] = (groups["RF_Group_LOFO_Drop_First"] > 0) & (groups["RF_Group_LOFO_Drop_Second"] > 0)
groups["Interpretation"] = "exploratory_same_replicas;group_LOFO_is_redundancy_sensitive_not_causality"
groups[["Feature_Group", "Feature_Count_first", "RF_Group_LOFO_Drop_First",
        "RF_Group_LOFO_Drop_Second", "Positive_Both", "Interpretation"]].sort_values("Feature_Group").to_csv(OUT / "gpr6_expanded_group_stability.csv", index=False)

print(f"Same 12 trajectory folds; {len(condition)} features; {len(groups)} grouped ablations")
print(pd.DataFrame(performance)[["Panel_Features", "Window", "Model", "Mean_Trajectory_Accuracy"]].to_string(index=False))
print(f"Cross-window tables: {OUT}")
