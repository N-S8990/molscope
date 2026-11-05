"""Proposal capability matches must never imply an implemented measurement."""
from dataclasses import replace

from molscope.analyzers import AnalyzerCatalog, ProposalCatalog
from molscope.registries import DEFAULT_FEATURE_REGISTRY, FeatureRegistry, FeatureStatus
from molscope.system_profile import SystemProfile


def profile():
    return SystemProfile(topology_path="synthetic.pdb", atom_count=3, residue_count=1,
                         components=(), capabilities=frozenset({"coordinates", "lipid", "periodic_cell"}),
                         has_periodic_cell=True, has_bonds=False)


def test_legacy_catalog_is_proposal_catalog_and_preserves_family_assessment():
    from molscope import ProposalCatalog as public_catalog
    from molscope.profiling.analyzers import ProposalCatalog as canonical_catalog
    assert public_catalog is canonical_catalog is ProposalCatalog
    assert AnalyzerCatalog is ProposalCatalog
    catalog = AnalyzerCatalog()
    proposal = next(item for item in catalog.assess(profile()) if item.feature_id == "membrane.order")
    assert proposal.status.value == "applicable"
    assert "membrane.order" not in {item.feature_id for item in catalog.implemented_features}


def test_implemented_features_and_assessments_derive_from_supplied_registry():
    definition = DEFAULT_FEATURE_REGISTRY.get("geometry.global_radius_of_gyration")
    unavailable = replace(definition, feature_id="custom.unimplemented", implementation=None)
    registry = FeatureRegistry((definition, unavailable), DEFAULT_FEATURE_REGISTRY.methods)
    catalog = ProposalCatalog(feature_registry=registry)
    assert catalog.implemented_features == (definition,)
    assessments = catalog.assess_features(profile(), available_inputs={"topology"})
    assert {item.feature_id: item.status for item in assessments} == {
        definition.feature_id: FeatureStatus.MISSING_INPUT,
        unavailable.feature_id: FeatureStatus.UNSUPPORTED,
    }


def test_proposal_assessment_does_not_hide_required_trajectory_input():
    catalog = ProposalCatalog()
    result = next(item for item in catalog.assess_features(profile(), available_inputs={"topology"})
                  if item.feature_id == "geometry.global_radius_of_gyration")
    assert result.status == FeatureStatus.MISSING_INPUT
    assert result.missing_inputs == ("trajectory",)
