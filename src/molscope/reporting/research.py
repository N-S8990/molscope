"""Researcher-facing report assembly from MolScope result tables."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd


def render_report_figures(
    sam_mfp: pd.DataFrame | None,
    state_populations: pd.DataFrame | None,
    output_dir: str | Path,
    *,
    quality: pd.DataFrame | None = None,
    representation_salience: pd.DataFrame | None = None,
    predictive_ranking: pd.DataFrame | None = None,
    state_explanatory_ranking: pd.DataFrame | None = None,
) -> list[str]:
    """Write compact, descriptive PNGs without converting them into claims."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    if sam_mfp is not None and {"PCA_Salience", "Replica_Consistency", "SAM_MFP_Pareto_Front"}.issubset(sam_mfp.columns):
        fig, axis = plt.subplots(figsize=(7, 5))
        front = sam_mfp["SAM_MFP_Pareto_Front"].astype(bool)
        axis.scatter(sam_mfp.loc[~front, "PCA_Salience"], sam_mfp.loc[~front, "Replica_Consistency"], alpha=0.35, s=18, label="Other eligible features")
        axis.scatter(sam_mfp.loc[front, "PCA_Salience"], sam_mfp.loc[front, "Replica_Consistency"], color="#d95f02", s=30, label="SAM-MFP Pareto candidates")
        axis.set(xlabel="Target-free PCA salience", ylabel="Replica consistency", title="Feature prioritization trade-off")
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
    if quality is not None and {"Valid_Fraction", "Replica_Consistency", "Pareto_Front"}.issubset(quality.columns):
        valid = quality[["Valid_Fraction", "Replica_Consistency", "Pareto_Front"]].copy()
        valid["Replica_Consistency"] = pd.to_numeric(valid["Replica_Consistency"], errors="coerce")
        valid = valid.dropna(subset=["Replica_Consistency"])
        if not valid.empty:
            fig, axis = plt.subplots(figsize=(7, 5))
            front = valid["Pareto_Front"].astype(bool)
            axis.scatter(valid.loc[~front, "Valid_Fraction"], valid.loc[~front, "Replica_Consistency"], alpha=0.35, s=18, label="Other features")
            axis.scatter(valid.loc[front, "Valid_Fraction"], valid.loc[front, "Replica_Consistency"], color="#1b9e77", s=30, label="Quality Pareto features")
            axis.set(xlabel="Finite-value fraction", ylabel="Replica consistency", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), title="Feature measurement quality")
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
        ranking["Trajectory_Mean_Accuracy_Drop_Mean"] = pd.to_numeric(ranking["Trajectory_Mean_Accuracy_Drop_Mean"], errors="coerce")
        ranking = ranking.dropna().nlargest(15, "Trajectory_Mean_Accuracy_Drop_Mean").iloc[::-1]
        if not ranking.empty:
            fig, axis = plt.subplots(figsize=(8, max(3, len(ranking) * 0.32)))
            axis.barh(ranking["Feature_Group"], ranking["Trajectory_Mean_Accuracy_Drop_Mean"], color="#1b9e77")
            axis.axvline(0, color="#555555", linewidth=0.8)
            axis.set(xlabel="Grouped trajectory-mean accuracy drop", title="Predictive feature association ranking (not causal)")
            fig.tight_layout()
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
    """Build cautious Markdown and JSON summaries; do not infer mechanism."""
    lines = ["# MolScope research report", "", "## Scope and interpretation", "", "This report distinguishes measurement quality, target-free ML representation evidence, and predictive ML evidence. None alone establishes causality.", ""]
    summary: dict[str, object] = {"limitations": ["Feature rankings are prioritization aids, not causal claims."]}
    if inputs.input_integrity:
        lines.extend(["## Input integrity", "", f"Validation status: `{inputs.input_integrity.get('status', 'unknown')}`.", ""])
        summary["input_integrity"] = inputs.input_integrity.get("status", "unknown")
    if inputs.system_profile:
        components = inputs.system_profile.get("components", [])
        lines.extend(["## Molecular system", "", f"Detected components: {len(components)}.", ""])
        summary["component_count"] = len(components)
    if inputs.quality is not None:
        quality = inputs.quality
        counts = quality.get("Resolution", pd.Series(dtype=str)).value_counts().to_dict()
        lines.extend(["## Feature quality", "", f"Feature resolutions: {counts}.", ""])
        summary["feature_resolution_counts"] = counts
    if inputs.sam_mfp is not None:
        candidates = inputs.sam_mfp[inputs.sam_mfp.get("SAM_MFP_Pareto_Front", False).astype(bool)] if "SAM_MFP_Pareto_Front" in inputs.sam_mfp else inputs.sam_mfp.iloc[0:0]
        lines.extend(["## Target-free prioritized feature candidates", "", "These are Pareto trade-offs across quality and representation evidence; they are not a single importance ranking.", "", *_table_lines(candidates, ["Feature", "PCA_Salience", "Representation_Stability", "Replica_Consistency", "Uniqueness"]), ""])
        summary["sam_mfp_pareto_candidate_count"] = int(len(candidates))
    if inputs.state_diagnostics is not None:
        lines.extend(["## ML conformational-state discovery", "", "Candidate state diagnostics (not proof of physical metastable states):", "", *_table_lines(inputs.state_diagnostics, ["State_Count", "Trajectory_Count", "Silhouette", "Initialization_ARI_Median", "Within_Ensemble_State_Population_JSD_Median"]), ""])
    if inputs.state_populations is not None:
        lines.extend(["State populations are reported per trajectory; frame counts are not treated as independent replicates.", ""])
    if inputs.state_explanatory_ranking is not None:
        lines.extend(["## Candidate-state explanatory ranking", "", "These effect sizes describe separation among states derived from these same frames. They are not predictive validation, independent replication, or evidence of physical mechanisms.", "", *_table_lines(inputs.state_explanatory_ranking, ["Rank", "Feature", "State_Explained_Variance_Eta_Squared", "Max_State_Mean_Difference"]), ""])
    if inputs.supervised_summary is not None:
        lines.extend(["## Predictive ML", "", "Predictive results describe association with the supplied target, not scientific causality.", "", *_table_lines(inputs.supervised_summary, ["Model", "Trajectory_Mean_Accuracy_Mean", "Balanced_Accuracy_Mean", "ROC_AUC_Mean"]), ""])
    if inputs.predictive_ranking is not None:
        lines.extend(["## Predictive feature ranking", "", "Ranking is based on trajectory-grouped leave-one-feature-group-out accuracy drops. It describes predictive association with the supplied target, not causal importance.", "", *_table_lines(inputs.predictive_ranking, ["Rank", "Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean", "Positive_Drop_Frequency"]), ""])
    lines.extend(["## Recommended next investigations", "", "- Review Pareto candidates together with their applicability and scientific-method metadata.", "- Check candidate findings across independent replicas before interpretation.", "- Treat clusters and transition counts as hypotheses requiring sampling and physical validation.", "- Use supervised results only when a valid trajectory-level target is available.", ""])
    return "\n".join(lines), summary


def read_optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def read_optional_json(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None
