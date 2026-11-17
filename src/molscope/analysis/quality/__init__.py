"""Analysis quality and statistical ML rigor diagnostics."""

from molscope.analysis.quality.assessment import (
    PrioritizationConfig,
    assess_feature_quality,
    validate_manifest,
)
from molscope.analysis.quality.statistical_rigor import (
    check_simulation_convergence,
    compute_clustering_validity_indices,
    evaluate_bootstrap_cluster_stability,
    evaluate_feature_ranking_stability,
    evaluate_group_bootstrap_uncertainty,
    evaluate_permutation_null_control,
    evaluate_seed_stability,
)

__all__ = [
    "PrioritizationConfig",
    "assess_feature_quality",
    "validate_manifest",
    "evaluate_seed_stability",
    "evaluate_bootstrap_cluster_stability",
    "compute_clustering_validity_indices",
    "check_simulation_convergence",
    "evaluate_feature_ranking_stability",
    "evaluate_group_bootstrap_uncertainty",
    "evaluate_permutation_null_control",
]
