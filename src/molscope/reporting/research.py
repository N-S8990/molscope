"""Researcher-facing report assembly from MolScope result tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _largest_absolute_accuracy_changes(ranking: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    """Keep both harmful and helpful removals when selecting a compact plot."""
    column = "Trajectory_Mean_Accuracy_Drop_Mean"
    valid = ranking.copy()
    valid[column] = pd.to_numeric(valid[column], errors="coerce")
    valid = valid.dropna(subset=[column])
    return valid.loc[valid[column].abs().nlargest(limit).index].iloc[::-1]


def render_report_figures(
    sam_mfp: pd.DataFrame | None,
    state_populations: pd.DataFrame | None,
    output_dir: str | Path,
    *,
    quality: pd.DataFrame | None = None,
    representation_salience: pd.DataFrame | None = None,
    predictive_ranking: pd.DataFrame | None = None,
    state_explanatory_ranking: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    representation_coordinates: pd.DataFrame | None = None,
    supervised_model_summary: pd.DataFrame | None = None,
    state_assignments: pd.DataFrame | None = None,
    decision_rules: Any | None = None,
) -> list[str]:
    """Write compact, descriptive PNGs without converting them into claims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if sam_mfp is not None and "SAM_MFP_Pareto_Front" in sam_mfp.columns:
        salience_col = None
        for candidate in ("PCA_Salience", "TICA_Salience", "Kinetic_Salience"):
            if candidate in sam_mfp.columns:
                salience_col = candidate
                break

        if salience_col is not None:
            # Determine best y-axis objective that has valid finite values
            y_col = None
            y_label = None
            if (
                "Replica_Consistency" in sam_mfp.columns
                and pd.to_numeric(sam_mfp["Replica_Consistency"], errors="coerce").notna().sum() > 0
            ):
                y_col = "Replica_Consistency"
                y_label = "Replica consistency"
            elif (
                "ESS_Ratio" in sam_mfp.columns
                and pd.to_numeric(sam_mfp["ESS_Ratio"], errors="coerce").notna().sum() > 0
            ):
                y_col = "ESS_Ratio"
                y_label = "Effective sample size ratio (ESS / N)"
            elif (
                "Uniqueness" in sam_mfp.columns
                and pd.to_numeric(sam_mfp["Uniqueness"], errors="coerce").notna().sum() > 0
            ):
                y_col = "Uniqueness"
                y_label = "Information uniqueness (1 - redundancy)"
            elif (
                "Occupancy_Entropy" in sam_mfp.columns
                and pd.to_numeric(sam_mfp["Occupancy_Entropy"], errors="coerce").notna().sum() > 0
            ):
                y_col = "Occupancy_Entropy"
                y_label = "Occupancy entropy"

            if y_col is not None:
                valid_mask = (
                    pd.to_numeric(sam_mfp[salience_col], errors="coerce").notna()
                    & pd.to_numeric(sam_mfp[y_col], errors="coerce").notna()
                )
                if "SAM_MFP_Eligible" in sam_mfp.columns and sam_mfp.loc[valid_mask, "SAM_MFP_Eligible"].any():
                    plot_data = sam_mfp[valid_mask & sam_mfp["SAM_MFP_Eligible"]].copy()
                else:
                    plot_data = sam_mfp[valid_mask].copy()

                if not plot_data.empty:
                    fig, axis = plt.subplots(figsize=(7, 5))
                    front = plot_data["SAM_MFP_Pareto_Front"].astype(bool)
                    axis.scatter(
                        plot_data.loc[~front, salience_col],
                        plot_data.loc[~front, y_col],
                        alpha=0.35,
                        s=18,
                        color="#4393c3",
                        label="Other eligible features",
                    )
                    axis.scatter(
                        plot_data.loc[front, salience_col],
                        plot_data.loc[front, y_col],
                        color="#d95f02",
                        s=30,
                        label="SAM-MFP Pareto candidates",
                    )
                    salience_name = salience_col.replace("_", " ")
                    axis.set(
                        xlabel=f"Target-free {salience_name}",
                        ylabel=y_label,
                        title=f"Feature prioritization trade-off ({salience_name} vs {y_col.replace('_', ' ')})",
                    )
                    axis.legend(frameon=False)
                    fig.tight_layout()
                    path = out_dir / "sam_mfp_tradeoff.png"
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    written.append(path.name)
    if state_populations is not None and {"trajectory_id", "State", "Population"}.issubset(state_populations.columns):
        matrix = state_populations.pivot(index="trajectory_id", columns="State", values="Population").fillna(0.0)
        if not matrix.empty:
            fig, axis = plt.subplots(figsize=(max(6, matrix.shape[1] * 1.2), max(3, matrix.shape[0] * 0.45)))
            image = axis.imshow(matrix.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
            axis.set(xticks=range(matrix.shape[1]), xticklabels=[f"state {item}" for item in matrix.columns], yticks=range(matrix.shape[0]), yticklabels=matrix.index, xlabel="Candidate ML state", ylabel="Trajectory", title="Per-trajectory state populations")
            fig.colorbar(image, ax=axis, label="Population")
            fig.tight_layout()
            path = out_dir / "state_populations.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path.name)
    if quality is not None and {"Valid_Fraction", "Pareto_Front"}.issubset(quality.columns):
        y_q_col = None
        y_q_label = None
        if (
            "Replica_Consistency" in quality.columns
            and pd.to_numeric(quality["Replica_Consistency"], errors="coerce").notna().sum() > 0
        ):
            y_q_col = "Replica_Consistency"
            y_q_label = "Replica consistency"
        elif (
            "ESS_Ratio" in quality.columns
            and pd.to_numeric(quality["ESS_Ratio"], errors="coerce").notna().sum() > 0
        ):
            y_q_col = "ESS_Ratio"
            y_q_label = "Effective sample size ratio (ESS / N)"
        elif (
            "Uniqueness" in quality.columns
            and pd.to_numeric(quality["Uniqueness"], errors="coerce").notna().sum() > 0
        ):
            y_q_col = "Uniqueness"
            y_q_label = "Information uniqueness"

        if y_q_col is not None:
            valid = quality[["Valid_Fraction", y_q_col, "Pareto_Front"]].copy()
            valid[y_q_col] = pd.to_numeric(valid[y_q_col], errors="coerce")
            valid["Valid_Fraction"] = pd.to_numeric(valid["Valid_Fraction"], errors="coerce")
            valid = valid.dropna(subset=["Valid_Fraction", y_q_col])
            if not valid.empty:
                fig, axis = plt.subplots(figsize=(7, 5))
                front = valid["Pareto_Front"].astype(bool)
                axis.scatter(valid.loc[~front, "Valid_Fraction"], valid.loc[~front, y_q_col], alpha=0.35, s=18, label="Other features")
                axis.scatter(valid.loc[front, "Valid_Fraction"], valid.loc[front, y_q_col], color="#1b9e77", s=30, label="Quality Pareto features")
                axis.set(xlabel="Finite-value fraction", ylabel=y_q_label, title=f"Feature measurement quality (Valid fraction vs {y_q_col.replace('_', ' ')})")
                axis.legend(frameon=False)
                fig.tight_layout()
                path = out_dir / "feature_quality_tradeoff.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(path.name)
    if representation_salience is not None and {"Feature", "PCA_Salience"}.issubset(representation_salience.columns):
        top = representation_salience.loc[:, ["Feature", "PCA_Salience"]].dropna().nlargest(15, "PCA_Salience").iloc[::-1]
        if not top.empty:
            fig, axis = plt.subplots(figsize=(8, max(3, len(top) * 0.32)))
            axis.barh(top["Feature"], top["PCA_Salience"], color="#7570b3")
            axis.set(xlabel="Target-free PCA salience", title="Representation-driving feature measurements")
            fig.tight_layout()
            path = out_dir / "representation_salience.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path.name)
    if predictive_ranking is not None and {"Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean"}.issubset(predictive_ranking.columns):
        ranking = predictive_ranking.loc[:, ["Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean"]].copy()
        ranking = _largest_absolute_accuracy_changes(ranking)
        if not ranking.empty:
            fig, axis = plt.subplots(figsize=(14, max(5, len(ranking) * 0.55)))
            axis.barh(ranking["Feature_Group"], ranking["Trajectory_Mean_Accuracy_Drop_Mean"], color="#1b9e77")
            axis.axvline(0, color="#555555", linewidth=0.8)
            axis.set(xlabel="Grouped trajectory-mean accuracy drop (negative: removal helped)", title="Largest absolute predictive changes (not causal)")
            axis.tick_params(axis="y", labelsize=9)
            fig.subplots_adjust(left=0.59, right=0.96, top=0.9, bottom=0.12)
            path = out_dir / "predictive_feature_ranking.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path.name)
    if state_explanatory_ranking is not None and {"Feature", "State_Explained_Variance_Eta_Squared"}.issubset(state_explanatory_ranking.columns):
        ranking = state_explanatory_ranking.loc[:, ["Feature", "State_Explained_Variance_Eta_Squared"]].copy()
        ranking["State_Explained_Variance_Eta_Squared"] = pd.to_numeric(ranking["State_Explained_Variance_Eta_Squared"], errors="coerce")
        ranking = ranking.dropna().nlargest(15, "State_Explained_Variance_Eta_Squared").iloc[::-1]
        if not ranking.empty:
            fig, axis = plt.subplots(figsize=(8, max(3, len(ranking) * 0.32)))
            axis.barh(ranking["Feature"], ranking["State_Explained_Variance_Eta_Squared"], color="#7570b3")
            axis.set(xlabel="Eta-squared across derived candidate states", title="Candidate-state separation (descriptive, not validated)")
            fig.tight_layout()
            path = out_dir / "state_explanatory_feature_ranking.png"
            fig.savefig(path, dpi=200)
            plt.close(fig)
            written.append(path.name)
    if features is not None and representation_salience is not None and "Feature" in representation_salience:
        selected = [name for name in representation_salience["Feature"].head(30) if name in features]
        if len(selected) >= 2:
            matrix = features[selected].apply(pd.to_numeric, errors="coerce").corr()
            matrix.to_csv(out_dir / "feature_correlation_top30.csv")
            fig, axis = plt.subplots(figsize=(max(7, len(selected) * 0.36), max(6, len(selected) * 0.36)))
            image = axis.imshow(matrix.to_numpy(), vmin=-1, vmax=1, cmap="coolwarm")
            axis.set(xticks=range(len(selected)), yticks=range(len(selected)),
                     xticklabels=selected, yticklabels=selected,
                     title="Feature correlation (top 30 by PCA salience)")
            plt.setp(axis.get_xticklabels(), rotation=90, fontsize=7)
            plt.setp(axis.get_yticklabels(), fontsize=7)
            fig.colorbar(image, ax=axis, label="Pearson correlation")
            fig.tight_layout()
            path = out_dir / "feature_correlation_map.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            written.append(path.name)
    if representation_coordinates is not None:
        columns = [name for name in ("PC1", "PC2", "PC3") if name in representation_coordinates]
        if len(columns) >= 2:
            values = representation_coordinates[columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            values = values[np.isfinite(values).all(axis=1)]
            if len(values) >= 2:
                fig, axis = plt.subplots(figsize=(7, 5))
                image = axis.hist2d(values[:, 0], values[:, 1], bins=35, cmap="viridis", cmin=1)
                axis.set(xlabel="PC1", ylabel="PC2", title="2D PCA occupancy (frame counts)")
                fig.colorbar(image[3], ax=axis, label="Frames per bin")
                fig.tight_layout()
                path = out_dir / "pca_density_2d.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(path.name)
                if len(columns) == 3:
                    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
                    counts, edges = np.histogramdd(values, bins=8)
                    occupied = np.argwhere(counts > 0)
                    centers = [0.5 * (axis_edges[:-1] + axis_edges[1:]) for axis_edges in edges]
                    xyz = np.column_stack([centers[j][occupied[:, j]] for j in range(3)])
                    density = counts[tuple(occupied.T)]
                    fig = plt.figure(figsize=(8, 7))
                    axis = fig.add_subplot(111, projection="3d")
                    dots = axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=density,
                                        s=18 + 90 * density / density.max(), cmap="viridis", alpha=0.8)
                    axis.set(xlabel="PC1", ylabel="PC2", zlabel="PC3",
                             title="3D PCA occupancy (binned frame counts)")
                    fig.colorbar(dots, ax=axis, shrink=0.7, label="Frames per voxel")
                    fig.tight_layout()
                    path = out_dir / "pca_density_3d.png"
                    fig.savefig(path, dpi=200)
                    plt.close(fig)
                    written.append(path.name)

                # Publication-quality 2D Free Energy Landscape on leading PCA components
                try:
                    from molscope.reporting.free_energy import compute_pca_free_energy_landscape
                    fes_res = compute_pca_free_energy_landscape(
                        representation_coordinates,
                        pc_x=columns[0],
                        pc_y=columns[1],
                    )
                    fes_path = out_dir / "pca_free_energy_landscape.png"
                    fes_res.plot_static(
                        output_path=fes_path,
                        xlabel=columns[0],
                        ylabel=columns[1],
                        title=f"Free Energy Landscape ({columns[0]} vs {columns[1]})",
                        dpi=200,
                    )
                    written.append(fes_path.name)
                except Exception:
                    pass

    # Cluster membership timeline and state transition probability matrix
    if state_assignments is not None and not state_assignments.empty:
        try:
            from molscope.visualization.timeline import plot_cluster_timeline, plot_state_transitions
            tl_path = out_dir / "cluster_timeline.png"
            plot_cluster_timeline(state_assignments, tl_path, dpi=200)
            written.append(tl_path.name)

            st_path = out_dir / "state_transitions.png"
            plot_state_transitions(state_assignments, st_path, dpi=200)
            written.append(st_path.name)
        except Exception:
            pass

    # Decision tree rule visualization
    if decision_rules is not None:
        try:
            from molscope.visualization.rules import plot_decision_tree_rules
            dt_path = out_dir / "decision_tree_rules.png"
            plot_decision_tree_rules(decision_rules, dt_path, dpi=200)
            written.append(dt_path.name)
        except Exception:
            pass
    elif features is not None and state_assignments is not None and not state_assignments.empty:
        try:
            from molscope.analysis.discovery.decision_tree import DecisionTreeExplainer
            from molscope.visualization.rules import plot_decision_tree_rules
            s_cand = None
            for c in ("State_Label", "State", "Cluster"):
                if c in state_assignments.columns:
                    s_cand = c
                    break
            if s_cand:
                labels = state_assignments[s_cand].dropna()
                if labels.nunique() >= 2:
                    min_leaf = max(3, len(labels) // 200)
                    dt_expl = DecisionTreeExplainer(max_depth=3, min_samples_leaf=min_leaf)
                    dt_expl.fit(features.loc[labels.index], labels)
                    dt_path = out_dir / "decision_tree_rules.png"
                    plot_decision_tree_rules(dt_expl, dt_path, dpi=200)
                    written.append(dt_path.name)
        except Exception:
            pass
    if (predictive_ranking is not None and supervised_model_summary is not None
            and {"Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean"}.issubset(predictive_ranking)
            and {"Model", "Trajectory_Mean_Accuracy_Mean"}.issubset(supervised_model_summary)):
        baseline_rows = supervised_model_summary.loc[supervised_model_summary["Model"].eq("random_forest")]
        if not baseline_rows.empty:
            baseline = float(baseline_rows.iloc[0]["Trajectory_Mean_Accuracy_Mean"])
            ranking = predictive_ranking[["Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean"]].copy()
            ranking = _largest_absolute_accuracy_changes(ranking)
            if not ranking.empty and np.isfinite(baseline):
                fig, axis = plt.subplots(figsize=(14, max(5, len(ranking) * 0.55)))
                y = np.arange(len(ranking))
                axis.barh(y - 0.18, [baseline] * len(y), height=0.35, label="All features", color="#4c78a8")
                axis.barh(y + 0.18, baseline - ranking["Trajectory_Mean_Accuracy_Drop_Mean"].to_numpy(),
                          height=0.35, label="Without feature group", color="#f58518")
                axis.set(yticks=y, yticklabels=ranking["Feature_Group"], xlabel="Trajectory-mean test accuracy",
                         xlim=(0, 1))
                axis.tick_params(axis="y", labelsize=9)
                fig.suptitle("Random forest accuracy by feature group", y=0.98)
                handles, labels = axis.get_legend_handles_labels()
                fig.legend(handles, labels, frameon=False, loc="upper right",
                           bbox_to_anchor=(0.96, 0.93), ncol=2)
                fig.subplots_adjust(left=0.59, right=0.96, top=0.82, bottom=0.12)
                path = out_dir / "predictive_accuracy_gain_drop.png"
                fig.savefig(path, dpi=200)
                plt.close(fig)
                written.append(path.name)
    return written


@dataclass(frozen=True)
class ReportInputs:
    quality: pd.DataFrame | None = None
    sam_mfp: pd.DataFrame | None = None
    state_diagnostics: pd.DataFrame | None = None
    state_populations: pd.DataFrame | None = None
    supervised_summary: pd.DataFrame | None = None
    predictive_ranking: pd.DataFrame | None = None
    state_explanatory_ranking: pd.DataFrame | None = None
    system_profile: dict | None = None
    input_integrity: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _table_lines(table: pd.DataFrame, columns: list[str], limit: int = 10) -> list[str]:
    present = [column for column in columns if column in table.columns]
    if not present or table.empty:
        return ["No results available."]
    rendered = table.loc[:, present].head(limit).copy()
    for column in rendered.select_dtypes(include="number"):
        rendered[column] = rendered[column].map(lambda value: f"{value:.4g}" if pd.notna(value) else "not available")
    header = "| " + " | ".join(present) + " |"
    separator = "| " + " | ".join("---" for _ in present) + " |"
    rows = ["| " + " | ".join(str(value) for value in row) + " |" for row in rendered.itertuples(index=False, name=None)]
    return [header, separator, *rows]


def build_research_report(inputs: ReportInputs) -> tuple[str, dict]:
    """Build a concise, findings-first report without inferring mechanism."""
    lines = ["# MolScope analysis report", "", "## Executive summary", ""]
    summary: dict[str, object] = {"limitations": ["Feature rankings prioritize measurements; they do not establish causal mechanisms."]}
    findings: list[str] = []
    if inputs.input_integrity:
        status = str(inputs.input_integrity.get("status", "unknown"))
        findings.append(f"Input validation status: **{status}**.")
        lines.extend(["## Input validation", "", f"Status: **{status}**.", ""])
        summary["input_integrity"] = inputs.input_integrity.get("status", "unknown")
    if inputs.system_profile:
        components = inputs.system_profile.get("components", [])
        findings.append(f"Molecular system profile detected {len(components)} component(s).")
        lines.extend(["## System profile", "", f"Detected components: **{len(components)}**.", ""])
        summary["component_count"] = len(components)
    if inputs.quality is not None:
        quality = inputs.quality
        counts = quality.get("Resolution", pd.Series(dtype=str)).value_counts().to_dict()
        total = int(sum(counts.values()))
        available = int(counts.get("frame", 0) + counts.get("trajectory", 0))
        if total:
            findings.append(f"Feature quality assessed {total} measurements; {available} are frame- or trajectory-resolved and {total - available} have another resolution state.")
        lines.extend(["## Measurement quality", "", f"{total} feature measurements were assessed. **{available}/{total}** are frame- or trajectory-resolved.", "", "| Resolution state | Feature count |", "| --- | ---: |"])
        lines.extend(f"| {name} | {count} |" for name, count in sorted(counts.items()))
        lines.append("")
        summary["feature_resolution_counts"] = counts
    if inputs.sam_mfp is not None:
        candidates = inputs.sam_mfp[inputs.sam_mfp.get("SAM_MFP_Pareto_Front", False).astype(bool)] if "SAM_MFP_Pareto_Front" in inputs.sam_mfp else inputs.sam_mfp.iloc[0:0]
        if "PCA_Salience" in candidates:
            candidates = candidates.sort_values("PCA_Salience", ascending=False, kind="stable")
        if len(candidates):
            top = candidates.iloc[0]
            findings.append(f"SAM-MFP identified {len(candidates)} Pareto candidate(s); the leading candidate by PCA salience is **{top.get('Feature', 'unnamed')}**.")
        else:
            findings.append("SAM-MFP produced no Pareto candidates in the supplied results.")
        lines.extend(["## Target-free prioritized feature candidates", "", "SAM-MFP reports a Pareto set across measurement quality and representation evidence. The set is a trade-off, not a single validated importance ranking.", "", *_table_lines(candidates, ["Feature", "PCA_Salience", "Representation_Stability", "Replica_Consistency", "Uniqueness"]), ""])
        summary["sam_mfp_pareto_candidate_count"] = int(len(candidates))
        summary["sam_mfp_leading_candidate_by_pca_salience"] = str(candidates.iloc[0].get("Feature")) if len(candidates) else None
    if inputs.state_diagnostics is not None:
        diagnostics = inputs.state_diagnostics
        ari_column = next((name for name in ("Initialization_ARI_Median", "Restart_ARI_Median") if name in diagnostics), None)
        diagnostic_columns = ["State_Count", "Trajectory_Count", "Silhouette"] + ([ari_column] if ari_column else []) + ["Within_Ensemble_State_Population_JSD_Median", "State_Population_JSD_Median"]
        diagnostic_columns = list(dict.fromkeys(diagnostic_columns))
        if not diagnostics.empty:
            findings.append(f"State discovery evaluated {len(diagnostics)} candidate state-count configuration(s); these diagnostics do not validate physical metastability.")
        lines.extend(["## Target-free state discovery", "", "State labels are ML-derived partitions of the analyzed representation. Diagnostics describe fit and reproducibility; they do not establish physical metastability.", "", *_table_lines(diagnostics, diagnostic_columns), ""])
    if inputs.state_populations is not None:
        lines.extend(["### State populations", "", "Populations are reported per trajectory. Frames are not independent biological or simulation replicates.", "", *_table_lines(inputs.state_populations, ["trajectory_id", "State", "Frame_Count", "Population"]), ""])
    if inputs.state_explanatory_ranking is not None:
        lines.extend(["## Candidate-state explanatory ranking", "", "These effect sizes describe separation among states derived from these same frames. They are not predictive validation, independent replication, or evidence of physical mechanisms.", "", *_table_lines(inputs.state_explanatory_ranking, ["Rank", "Feature", "State_Explained_Variance_Eta_Squared", "Max_State_Mean_Difference"]), ""])
    if inputs.supervised_summary is not None:
        lines.extend(["## Predictive ML", "", "Predictive results describe association with the supplied target, not scientific causality.", "", *_table_lines(inputs.supervised_summary, ["Model", "Trajectory_Mean_Accuracy_Mean", "Balanced_Accuracy_Mean", "ROC_AUC_Mean"]), ""])
    if inputs.predictive_ranking is not None:
        lines.extend(["## Predictive feature ranking", "", "Ranking is based on trajectory-grouped leave-one-feature-group-out accuracy drops. It describes predictive association with the supplied target, not causal importance.", "", *_table_lines(inputs.predictive_ranking, ["Rank", "Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean", "Positive_Drop_Frequency"]), ""])
    executive_findings = [f"- {finding}" for finding in findings] or ["No analysis result tables were supplied. This report contains no scientific findings."]
    lines[4:4] = [*executive_findings, ""]
    lines.extend(["## Interpretation limits and next checks", "", "- Review prioritized features with their applicability state and method metadata.", "- Check findings across independent trajectories and sampling conditions.", "- Interpret clusters and transition counts as hypotheses that need physical validation.", "- Predictive scores apply only to the supplied target and trajectory-grouped validation design.", ""])
    summary["executive_findings"] = findings
    return "\n".join(lines), summary


def read_optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def read_optional_json(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def write_result_guide(output_dir: str | Path, outputs: dict, figures: list[str]) -> None:
    """Give readers one entry point and a complete, labeled feature table."""
    root = Path(output_dir)
    candidates = outputs.get("sam_mfp_candidates")
    ranking = outputs.get("supervised_predictive_feature_ranking")
    if isinstance(candidates, pd.DataFrame):
        columns = [name for name in (
            "Feature", "SAM_MFP_Eligible", "SAM_MFP_Pareto_Front", "PCA_Salience",
            "Representation_Stability", "Replica_Consistency", "Uniqueness",
            "Valid_Fraction", "Cost_Objective_Status", "Measured_Cost_Seconds",
        ) if name in candidates]
        guide = candidates[columns].copy()
        if isinstance(ranking, pd.DataFrame) and {"Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean"}.issubset(ranking):
            predictive_columns = [name for name in (
                "Feature_Group", "Baseline_Trajectory_Mean_Accuracy",
                "Without_Group_Trajectory_Mean_Accuracy", "Trajectory_Mean_Accuracy_Drop_Mean",
                "Trajectory_Mean_Accuracy_Gain_Without_Group", "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Lower",
                "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Upper", "Positive_Drop_Frequency",
                "Uncertainty_Qualification",
            ) if name in ranking]
            predictive = ranking[predictive_columns].drop_duplicates("Feature_Group")
            guide = guide.merge(predictive, left_on="Feature", right_on="Feature_Group", how="left", validate="one_to_one")
            guide = guide.drop(columns="Feature_Group")
        def explain_row(row: pd.Series) -> pd.Series:
            pareto = bool(row.get("SAM_MFP_Pareto_Front", False))
            drop = pd.to_numeric(row.get("Trajectory_Mean_Accuracy_Drop_Mean"), errors="coerce")
            if pd.notna(drop):
                predictive = ("removing this feature group reduced grouped test accuracy" if drop > 0
                              else "removing this feature group improved grouped test accuracy" if drop < 0
                              else "removing this feature group did not change grouped test accuracy")
            else:
                predictive = "no matched supervised accuracy result"
            priority = "Pareto candidate to inspect" if pareto else "assessed feature"
            next_check = ("Check definition, applicability, and stability across independent replicas; "
                          "test the finding on held-out systems before assigning mechanism.")
            return pd.Series({"Review_Priority": priority, "Evidence_Summary": predictive,
                              "Suggested_Next_Check": next_check})
        guide = pd.concat([guide, guide.apply(explain_row, axis=1)], axis=1)
        guide.to_csv(root / "feature_guide.csv", index=False)
    lines = [
        "# Start here: MolScope results", "",
        "Read [research_report.md](research_report.md) first for findings and limits, then "
        "[feature_guide.csv](feature_guide.csv) for the complete feature list.", "",
        "## What each result means", "",
        "- **Candidate measurements:** `feature_guide.csv` and `sam_mfp_candidates.csv` list all assessed features. "
        "A Pareto flag identifies trade-off candidates, not a single universal importance ranking.",
        "- **Structure of the data:** `representation_feature_salience.csv` gives PCA loading evidence. "
        "`state_assignments.csv` and `state_diagnostics.csv` describe candidate ML clusters; they are not validated physical states.",
        "- **Predictive evidence:** If an external trajectory-level target was supplied, "
        "`supervised_model_summary.csv`, `supervised_outer_fold_metrics.csv`, and "
        "`supervised_predictive_feature_ranking.csv` report grouped validation and accuracy changes "
        "when feature groups are removed. Positive drop means the model performed worse without the group; "
        "negative drop means it performed better. Fold-resample bounds are descriptive, not independent-replica confidence intervals. Neither establishes causality.",
        "- **Complete data:** Other CSV files retain the quality, population, transition, and fold-level tables. "
        "`feature_correlation_top30.csv` contains the displayed correlation coefficients. "
        "`research_report.json` carries the machine-readable summary.", "",
        "Predictive columns in `feature_guide.csv` are populated only when a feature name exactly matches "
        "the removed predictive group; use the full supervised ranking for grouped features.", "",
        "## Figures", "",
    ]
    descriptions = {
        "feature_correlation_map.png": "Pearson correlation among up to 30 features selected by PCA salience; correlated features can be redundant.",
        "pca_density_2d.png": "Frame occupancy in PC1–PC2 bins. These axes are learned feature coordinates, not physical x/y positions.",
        "pca_density_3d.png": "Binned frame occupancy in PC1–PC3 shown as a colored 3D scatter. These axes are learned coordinates, not physical x/y/z positions.",
        "predictive_accuracy_gain_drop.png": "Grouped test accuracy with all features and with each of the 15 groups having the largest absolute changes removed.",
        "predictive_feature_ranking.png": "The 15 largest absolute signed changes in grouped test accuracy when a feature group is removed.",
        "sam_mfp_tradeoff.png": "Representation salience plotted against measurement quality (replica consistency or effective sample size), with Pareto candidates highlighted.",
        "state_populations.png": "Candidate state populations by trajectory.",
        "feature_quality_tradeoff.png": "Measurement validity plotted against quality metrics (replica consistency or effective sample size).",
        "representation_salience.png": "Features contributing most to the PCA representation.",
        "state_explanatory_feature_ranking.png": "Descriptive separation of states inferred from the same data.",
    }
    lines.extend(f"- [{name}]({name}): {descriptions.get(name, 'Descriptive analysis figure.')}" for name in figures)
    lines.extend(["", "## Measurements to inspect first", ""])
    if isinstance(candidates, pd.DataFrame) and {"Feature", "SAM_MFP_Pareto_Front"}.issubset(candidates):
        front = candidates.loc[candidates["SAM_MFP_Pareto_Front"].astype(bool)].head(10)
        if not front.empty:
            lines.append("These are Pareto candidates in the supplied run, shown in report order. "
                         "Check their definitions, quality, and replica consistency before interpretation.")
            lines.append("")
            lines.extend(f"- `{name}` — inspect its measurement definition and system applicability; "
                         "compare its feature quality and, when available, grouped predictive accuracy drop."
                         for name in front["Feature"])
        else:
            lines.append("No eligible Pareto candidates were found; inspect quality and applicability tables before selecting features.")
    else:
        lines.append("No prioritization table was supplied.")
    lines.extend(["", "## Interpretation", "",
                  "Feature importance here means descriptive or predictive evidence within this dataset. "
                  "It does not establish a molecular mechanism. Review the feature definition, applicability, "
                  "replica consistency, and independent validation before deciding what to study experimentally.", ""])
    (root / "START_HERE.md").write_text("\n".join(lines), encoding="utf-8")
