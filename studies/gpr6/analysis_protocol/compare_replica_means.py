"""Exact condition-label permutation test of the 12 independent GPR6 replicas.

Every feature was declared in the project before this comparison. The unit of
permutation is a whole trajectory, not one of its correlated frames. These are
exploratory associations, not causal or mechanistic significance tests.
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molscope.core.project import load_project_spec
from molscope.features.geometry.generic import GenericFeatureConfig, GenericFeatureExtractor
from molscope.features.plugins import get_plugin


ROOT = Path(__file__).resolve().parents[1]
PROJECT = Path(__file__).resolve().parent / "protein_first_100ns_1000_frames/project.json"
RUN = ROOT / "runs/public_12_replica_first_100ns"


def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    ranked = p_values[order] * len(p_values) / np.arange(1, len(p_values) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=PROJECT)
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--panel-label", default="GPR6 first 100 ns")
    args = parser.parse_args()
    run = args.run
    project = load_project_spec(args.project)
    label = project.labels[0]
    assert label.label_id == "author_deposited_condition"
    assert len(project.trajectory_inputs()) == 12
    generic = GenericFeatureExtractor(GenericFeatureConfig(**project.analysis.generic_feature_parameters))
    plugin_ids = project.analysis.plugins
    plugins = [get_plugin(plugin_id, project.analysis.plugin_parameters.get(plugin_id)) for plugin_id in plugin_ids]
    records = []
    frame_tables = []
    for item in project.trajectory_inputs():
        generic_frame = generic.extract(item.topology, item.trajectory).frame_features
        if len(generic_frame) != 1000:
            raise ValueError(f"Expected exactly 1000 frames: {item.trajectory_id}")
        identity = generic_frame[["frame_index", "frame_order", "time_ps"]]
        values = generic_frame.drop(columns=identity.columns)
        for plugin in plugins:
            plugin_frame = plugin.extract(item.topology, item.trajectory).frame_features
            if not identity.equals(plugin_frame[identity.columns]):
                raise ValueError(f"Plugin frame identity differs: {item.trajectory_id}")
            values = pd.concat([values, plugin_frame.drop(columns=identity.columns)], axis=1)
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"Nonfinite feature values: {item.trajectory_id}")
        record = {"trajectory_id": item.trajectory_id, "condition": label.assignments[item.trajectory_id],
                  "frames": len(values), **values.mean().to_dict()}
        records.append(record)
        frame_tables.append(pd.concat([
            pd.DataFrame({"trajectory_id": item.trajectory_id,
                          "condition": label.assignments[item.trajectory_id]}, index=identity.index),
            identity, values,
        ], axis=1))
        print(f"Extracted replica means: {item.trajectory_id}", flush=True)
    replica = pd.DataFrame(records)
    feature_names = [column for column in replica if column not in {"trajectory_id", "condition", "frames"}]
    guide = pd.read_csv(run / "feature_guide.csv")
    if set(feature_names) != set(guide["Feature"]) or len(feature_names) != len(guide):
        raise ValueError("Replica comparison feature schema differs from MolScope run")
    replica = replica[["trajectory_id", "condition", "frames", *guide["Feature"]]]
    feature_names = list(guide["Feature"])
    conditions = replica["condition"].to_numpy()
    bound = conditions == "3h_bound"
    if int(bound.sum()) != 6 or int((~bound).sum()) != 6:
        raise ValueError("Expected six independent replicas per condition")
    matrix = replica[feature_names].to_numpy(dtype=float)
    masks = np.zeros((924, 12), dtype=float)
    for row, chosen in enumerate(combinations(range(12), 6)):
        masks[row, list(chosen)] = 1.0
    differences = (masks @ matrix - (1.0 - masks) @ matrix) / 6.0
    observed = (matrix[bound].mean(axis=0) - matrix[~bound].mean(axis=0))
    p_values = np.mean(np.abs(differences) >= np.abs(observed)[None, :] - 1e-12, axis=0)
    if not np.all((p_values > 0) & (p_values <= 1)):
        raise ValueError("Invalid exact permutation p values")
    q_values = bh_qvalues(p_values)
    q_column = f"BH_Q_Across_{len(feature_names)}_Features"
    comparison = pd.DataFrame({
        "Feature": feature_names,
        "No_Added_Inverse_Agonist_Replica_Mean": matrix[~bound].mean(axis=0),
        "Three_H_Bound_Replica_Mean": matrix[bound].mean(axis=0),
        "Three_H_Minus_No_Added_Inverse_Agonist": observed,
        "Exact_Two_Sided_Permutation_P": p_values,
        q_column: q_values,
        "Replicas_Per_Condition": 6,
        "Permutations": 924,
        "Inference_Unit": "independent_trajectory_mean",
        "Interpretation": "exploratory_condition_association_not_causality",
    }).sort_values([q_column, "Exact_Two_Sided_Permutation_P", "Feature"])
    run.mkdir(parents=True, exist_ok=True)
    frames = pd.concat(frame_tables, ignore_index=True)
    if len(frames) != 12000 or frames["trajectory_id"].nunique() != 12:
        raise ValueError("Full frame-feature matrix is incomplete")
    frames.to_csv(run / "gpr6_frame_features.csv", index=False)
    replica.to_csv(run / "gpr6_replica_feature_means.csv", index=False)
    comparison.to_csv(run / "gpr6_feature_condition_comparison.csv", index=False)
    # All plotted effects are distances/Rg/RMSD in angstroms. Exclude
    # dimensionless contact and asphericity measurements from this axis.
    distance_effects = comparison[
        comparison["Feature"].str.startswith("CA_DISTANCE_angstrom__")
        | comparison["Feature"].isin(["DYN_kabsch_rmsd_to_first_frame", "GEOM_global_radius_gyration"])
    ].head(10).iloc[::-1]
    labels = []
    for feature in distance_effects["Feature"]:
        if feature.startswith("CA_DISTANCE_angstrom__"):
            left, right = feature.removeprefix("CA_DISTANCE_angstrom__").split("__X__")
            labels.append(left.removeprefix("X__").replace("_", "") + "–" + right.replace("_", "") + " Cα")
        elif feature == "DYN_kabsch_rmsd_to_first_frame":
            labels.append("Receptor RMSD to first frame")
        else:
            labels.append("Receptor radius of gyration")
    q = distance_effects[q_column].to_numpy()
    changes = distance_effects["Three_H_Minus_No_Added_Inverse_Agonist"].to_numpy()
    fig, ax = plt.subplots(figsize=(11, 6.2))
    colors = ["#24628e" if value < 0.05 else "#9aa4ae" for value in q]
    bars = ax.barh(labels, changes, color=colors)
    for bar, value in zip(bars, q, strict=True):
        ax.text(bar.get_width() + 0.025, bar.get_y() + bar.get_height() / 2,
                f"q={value:.3f}", va="center", fontsize=9)
    ax.axvline(0, color="#333333", linewidth=0.8)
    ax.set_xlabel("3h-bound minus no-added-inverse-agonist replica mean (Å)")
    ax.set_title(f"{args.panel_label}: exploratory receptor differences")
    ax.text(0, -0.16, f"Exact two-sided trajectory-label permutations (924); BH q across all {len(feature_names)} tested features.\n"
            "Six independent replicas per condition. Residue IDs match the primary GPR6 paper; effects are not causal mechanisms.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8)
    ax.set_xlim(right=max(changes) + 0.25)
    fig.subplots_adjust(left=0.31, right=0.96, top=0.9, bottom=0.26)
    fig.savefig(run / "gpr6_condition_effects.png", dpi=180)
    plt.close(fig)
    print(comparison.head(12).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
