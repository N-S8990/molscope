"""One coherent target-free MolScope discovery pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass

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
    supervised = {}
    if target_column is not None:
        supervised = run_grouped_ml_workflow(
            features, manifest, target_column,
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
        "state_explanatory_feature_ranking": state_explanatory_ranking,
        **{f"supervised_{name}": table for name, table in supervised.items()},
        "research_report_markdown": markdown,
        "research_report_summary": {"pipeline_configuration": config.to_dict(), "target_column": target_column, "sam_mfp_notes": notes, **summary},
    }
