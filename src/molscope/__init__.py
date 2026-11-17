"""MolScope's system-aware molecular-feature discovery platform.

For a new study, the public front door is :class:`MDContactMLPipeline` together
with a JSON project declaration and ``python -m molscope.run_project``.
Small command modules expose individual stages for diagnostics and development.
Legacy projects are maintained separately and are not part of this package.
"""

from molscope.profiling.analyzers import AnalyzerCatalog, AnalyzerSpec, ProposalCatalog
from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.analysis.discovery.representation import RepresentationDiscoveryConfig, discover_representation_features
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.features.geometry.generic import GenericFeatureConfig, GenericFeatureExtractor
from molscope.io.validation import InputIntegrityReport, InputIntegrityStatus, inspect_md_inputs, topology_identity
from molscope.features.plugins import ProteinBackboneTorsionPlugin
from molscope.features.registry import DEFAULT_FEATURE_REGISTRY, METHODS, FeatureAvailability, FeatureDefinition, FeatureStatus, ScientificMethod
from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow
from molscope.core.identity import TrajectoryInput, build_frame_manifest
from molscope.profiling.system import ComponentProfile, SystemProfile, profile_system
from molscope.analysis.discovery.states import StateDiscoveryConfig, discover_conformational_states, evaluate_state_count_sensitivity
from molscope.analysis.discovery.tica import TICAConfig, TICAResult, fit_tica
from molscope.analysis.discovery.contact_kpca import ContactKPCAConfig, ContactKPCAResult, fit_contact_kpca
from molscope.analysis.discovery.decision_tree import DecisionRule, DecisionTreeExplainer
from molscope.analysis.discovery.density_peaks import DPAConfig, DPAResult, cluster_dpa
from molscope.execution.discovery import DiscoveryPipelineConfig, run_target_free_discovery_pipeline
from molscope.execution.pipeline import MDContactMLPipeline
from molscope.core.project import OpenMMSimulationSpec
from molscope.api import Study
from molscope.io.transformations import ImagingPolicy, load_universe
from molscope.analysis.quality.statistical_rigor import (
    check_simulation_convergence,
    compute_clustering_validity_indices,
    evaluate_bootstrap_cluster_stability,
    evaluate_feature_ranking_stability,
    evaluate_group_bootstrap_uncertainty,
    evaluate_permutation_null_control,
    evaluate_seed_stability,
)

__version__ = "0.2.0"

__all__ = [
    "__version__",
    "AnalyzerCatalog",
    "ProposalCatalog",
    "AnalyzerSpec",
    "ApplicabilityStatus",
    "ComponentProfile",
    "FeatureAssessment",
    "FeatureProvenance",
    "RepresentationDiscoveryConfig",
    "GenericFeatureConfig",
    "GenericFeatureExtractor",
    "DEFAULT_FEATURE_REGISTRY",
    "FeatureAvailability",
    "FeatureDefinition",
    "FeatureStatus",
    "InputIntegrityReport",
    "InputIntegrityStatus",
    "ProteinBackboneTorsionPlugin",
    "MLWorkflowConfig",
    "METHODS",
    "run_grouped_ml_workflow",
    "SystemProfile",
    "ScientificMethod",
    "TrajectoryInput",
    "build_frame_manifest",
    "profile_system",
    "discover_representation_features",
    "SAMMFPConfig",
    "prioritize_sam_mfp",
    "inspect_md_inputs",
    "topology_identity",
    "StateDiscoveryConfig",
    "discover_conformational_states",
    "evaluate_state_count_sensitivity",
    "TICAConfig",
    "TICAResult",
    "fit_tica",
    "ContactKPCAConfig",
    "ContactKPCAResult",
    "fit_contact_kpca",
    "DecisionRule",
    "DecisionTreeExplainer",
    "DPAConfig",
    "DPAResult",
    "cluster_dpa",
    "DiscoveryPipelineConfig",
    "run_target_free_discovery_pipeline",
    "MDContactMLPipeline",
    "OpenMMSimulationSpec",
    "Study",
    "ImagingPolicy",
    "load_universe",
    "evaluate_seed_stability",
    "evaluate_bootstrap_cluster_stability",
    "compute_clustering_validity_indices",
    "check_simulation_convergence",
    "evaluate_feature_ranking_stability",
    "evaluate_group_bootstrap_uncertainty",
    "evaluate_permutation_null_control",
]
