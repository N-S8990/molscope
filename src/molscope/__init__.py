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
from molscope.execution.discovery import DiscoveryPipelineConfig, run_target_free_discovery_pipeline
from molscope.execution.pipeline import MDContactMLPipeline
from molscope.core.project import OpenMMSimulationSpec

__all__ = [
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
    "DiscoveryPipelineConfig",
    "run_target_free_discovery_pipeline",
    "MDContactMLPipeline",
    "OpenMMSimulationSpec",
]
