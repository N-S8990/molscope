"""One coherent target-free MolScope discovery pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from molscope.analysis.discovery.representation import RepresentationDiscoveryConfig, discover_representation_features
from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow
from molscope.analysis.quality.assessment import PrioritizationConfig, assess_feature_quality, validate_manifest
from molscope.reporting.research import ReportInputs, build_research_report
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.discovery.states import StateDiscoveryConfig, discover_conformational_states, rank_state_explanatory_features


@dataclass(frozen=True)
class DiscoveryPipelineConfig:
    pca_components: int = 5
    state_count: int = 2
    supervised_splits: int = 3
    supervised_bootstrap_samples: int = 1_000

    def to_dict(self) -> dict:
        return asdict(self)


def run_target_free_discovery_pipeline(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: DiscoveryPipelineConfig = DiscoveryPipelineConfig(),
    *,
    feature_costs: pd.DataFrame | None = None,
    target_column: str | None = None,
) -> dict[str, pd.DataFrame | str | dict]:
    """Run the reproducible, label-independent core analysis path."""
    validate_manifest(manifest, len(features))
    if not ({"frame_order", "frame_index"} & set(manifest.columns)):
        raise ValueError("Manifest requires frame_order or frame_index for ML state-transition analysis.")
    quality = assess_feature_quality(features, manifest, PrioritizationConfig(), feature_costs=feature_costs)
    representation = discover_representation_features(
        features, manifest, RepresentationDiscoveryConfig(n_components=config.pca_components),
    )
    candidates, notes = prioritize_sam_mfp(quality, representation["representation_feature_salience"], SAMMFPConfig())
    states = discover_conformational_states(
        representation["representation_coordinates"], manifest, StateDiscoveryConfig(n_states=config.state_count),
    )
    state_explanatory_ranking = rank_state_explanatory_features(features, states["state_assignments"])

    # Density Peak Advanced (DPA) Clustering
    dpa_outputs = {}
    try:
        from molscope.analysis.discovery.density_peaks import cluster_dpa
        dpa_res = cluster_dpa(representation["representation_coordinates"], manifest)
        dpa_outputs["dpa_cluster_assignments"] = dpa_res.assignments
        dpa_outputs["dpa_cluster_populations"] = dpa_res.populations
        dpa_outputs["dpa_decision_graph"] = dpa_res.decision_graph
    except Exception:
        pass

    # Decision Tree Rule Explainer
    dt_outputs = {}
    try:
        from molscope.analysis.discovery.decision_tree import DecisionTreeExplainer
        explainer = DecisionTreeExplainer(max_depth=3)
        target_labels = states["state_assignments"]["State_Label"]
        explainer.fit(features, target_labels)
        dt_outputs["decision_tree_rules"] = explainer.rules_dataframe()
        dt_outputs["decision_tree_feature_importances"] = explainer.get_feature_importances()
    except Exception:
        pass

    # ML Workflow is a MUST: if no external target_column is declared, train on discovered conformational states
    ml_manifest = manifest.copy()
    effective_target = target_column
    if effective_target is None:
        effective_target = "conformational_state"
        ml_manifest[effective_target] = states["state_assignments"]["State_Label"].to_numpy()

    # Global Feature Importance across ALL features based on ALL data using regularized Random Forest
    global_rf_outputs = {}
    try:
        from sklearn.ensemble import RandomForestClassifier
        global_rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=10,
            max_features="sqrt",
            random_state=config.random_state,
            n_jobs=-1,
        )
        target_labels = ml_manifest[effective_target].to_numpy()
        global_rf.fit(features, target_labels)
        global_imp_df = pd.DataFrame({
            "Feature": features.columns,
            "Importance_MDI": global_rf.feature_importances_,
        }).sort_values("Importance_MDI", ascending=False).reset_index(drop=True)
        global_imp_df["Rank"] = np.arange(1, len(global_imp_df) + 1)
        global_rf_outputs["global_feature_importance_all_features"] = global_imp_df
    except Exception:
        pass

    # Prioritize candidate features for predictive ML workflow (cap to top 30 to keep LOFO ablation fast & non-redundant)
    if "Feature" in candidates.columns and len(candidates) >= 2:
        cand_names = [f for f in candidates["Feature"] if f in features.columns]
    elif "Feature" in state_explanatory_ranking.columns and len(state_explanatory_ranking) >= 2:
        cand_names = [f for f in state_explanatory_ranking["Feature"].head(30) if f in features.columns]
    else:
        cand_names = list(features.columns[:30])

    if len(cand_names) > 30:
        cand_names = cand_names[:30]
    ml_features = features[cand_names] if len(cand_names) >= 2 else features

    supervised = run_grouped_ml_workflow(
        ml_features, ml_manifest, effective_target,
        MLWorkflowConfig(n_splits=config.supervised_splits, bootstrap_samples=config.supervised_bootstrap_samples),
    )

    markdown, summary = build_research_report(ReportInputs(
        quality=quality, sam_mfp=candidates, state_diagnostics=states["state_diagnostics"], state_populations=states["state_populations"],
        supervised_summary=supervised.get("model_summary"),
        predictive_ranking=supervised.get("predictive_feature_ranking"),
        state_explanatory_ranking=state_explanatory_ranking,
    ))
    return {
        "feature_quality_metrics": quality,
        **representation,
        "sam_mfp_candidates": candidates,
        **states,
        **dpa_outputs,
        **dt_outputs,
        **global_rf_outputs,
        "state_explanatory_feature_ranking": state_explanatory_ranking,
        **{f"supervised_{name}": table for name, table in supervised.items()},
        "research_report_markdown": markdown,
        "research_report_summary": {"pipeline_configuration": config.to_dict(), "target_column": effective_target, "sam_mfp_notes": notes, **summary},
    }
