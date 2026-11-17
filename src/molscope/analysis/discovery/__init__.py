"""analysis discovery components."""

from molscope.analysis.discovery.contact_kpca import (
    ContactKPCAConfig,
    ContactKPCAResult,
    fit_contact_kpca,
)
from molscope.analysis.discovery.decision_tree import (
    DecisionRule,
    DecisionTreeExplainer,
)
from molscope.analysis.discovery.density_peaks import (
    DPAConfig,
    DPAResult,
    cluster_dpa,
)
from molscope.analysis.discovery.representation import (
    RepresentationDiscoveryConfig,
    discover_representation_features,
)
from molscope.analysis.discovery.states import (
    StateDiscoveryConfig,
    discover_conformational_states,
    rank_state_explanatory_features,
)
from molscope.analysis.discovery.tica import (
    TICAConfig,
    TICAResult,
    fit_tica,
)

__all__ = [
    "ContactKPCAConfig",
    "ContactKPCAResult",
    "fit_contact_kpca",
    "DecisionRule",
    "DecisionTreeExplainer",
    "DPAConfig",
    "DPAResult",
    "cluster_dpa",
    "RepresentationDiscoveryConfig",
    "discover_representation_features",
    "StateDiscoveryConfig",
    "discover_conformational_states",
    "rank_state_explanatory_features",
    "TICAConfig",
    "TICAResult",
    "fit_tica",
]
