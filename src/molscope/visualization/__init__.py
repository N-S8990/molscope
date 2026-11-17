"""MolScope visualization and interpretability tools."""

from .pdb_mapping import (
    extract_residues_from_feature_name,
    extract_residue_scores,
    map_feature_importance_to_pdb,
    generate_pymol_script,
    generate_chimerax_script,
    generate_3d_visualization,
)
from .timeline import (
    compute_transition_matrix,
    plot_cluster_timeline,
    plot_state_transitions,
    generate_cluster_timeline_plots,
)
from .rules import (
    plot_feature_importance,
    plot_decision_tree_rules,
    render_decision_tree_html,
)

__all__ = [
    # PDB mapping & 3D viewers
    "extract_residues_from_feature_name",
    "extract_residue_scores",
    "map_feature_importance_to_pdb",
    "generate_pymol_script",
    "generate_chimerax_script",
    "generate_3d_visualization",
    # Timelines & transitions
    "compute_transition_matrix",
    "plot_cluster_timeline",
    "plot_state_transitions",
    "generate_cluster_timeline_plots",
    # Rules & feature importance
    "plot_feature_importance",
    "plot_decision_tree_rules",
    "render_decision_tree_html",
]
