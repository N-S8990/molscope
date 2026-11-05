"""Analyzer applicability registry.

This compatibility catalogue proposes families of analyses. Capability matches
here do not establish implementation availability. The scientific registry in
``molscope.features.registry`` is the authoritative catalogue of implemented features.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment
from molscope.features.registry import DEFAULT_FEATURE_REGISTRY, FeatureAvailability, FeatureDefinition, FeatureRegistry
from molscope.profiling.system import SystemProfile


@dataclass(frozen=True)
class AnalyzerSpec:
    analyzer_id: str
    family: str
    description: str
    granularity: str
    units: str | None
    required_capabilities: frozenset[str]
    required_input_capabilities: frozenset[str] = frozenset()

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing_inputs = tuple(sorted(self.required_input_capabilities - profile.capabilities))
        if missing_inputs:
            return FeatureAssessment(
                feature_id=self.analyzer_id,
                family=self.family,
                status=ApplicabilityStatus.MISSING_INPUT,
                reason="Required input data are absent from the supplied topology or trajectory.",
                granularity=self.granularity,
                units=self.units,
                missing_inputs=missing_inputs,
            )
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(
                feature_id=self.analyzer_id,
                family=self.family,
                status=ApplicabilityStatus.NOT_APPLICABLE,
                reason="Required system capabilities are absent.",
                granularity=self.granularity,
                units=self.units,
                missing_capabilities=missing,
            )
        return FeatureAssessment(
            feature_id=self.analyzer_id,
            family=self.family,
            status=ApplicabilityStatus.APPLICABLE,
            reason="All declared system capabilities are available.",
            granularity=self.granularity,
            units=self.units,
        )


DEFAULT_ANALYZERS = (
    AnalyzerSpec("geometry.global_shape", "geometry", "Global size and shape descriptors.", "frame", "angstrom", frozenset({"coordinates"})),
    AnalyzerSpec("geometry.component_distances", "geometry", "Distances between molecular components.", "frame", "angstrom", frozenset({"coordinates", "multiple_molecular_components"})),
    AnalyzerSpec("dynamics.displacement", "dynamics", "Reference-relative conformational displacement.", "frame", "angstrom", frozenset({"coordinates"})),
    AnalyzerSpec("dynamics.rmsf", "dynamics", "Per-atom or per-residue fluctuation; trajectory-level by definition.", "trajectory", "angstrom", frozenset({"coordinates"})),
    AnalyzerSpec("protein.backbone_torsions", "protein_chemistry", "Protein backbone torsions using bond topology.", "frame", "degrees", frozenset({"protein_backbone", "bonds"})),
    AnalyzerSpec("protein.sidechain_interactions", "protein_chemistry", "Protein-specific sidechain contact chemistry.", "frame", None, frozenset({"protein_sidechains"})),
    AnalyzerSpec("nucleic.base_pairing", "nucleic_acid", "Nucleic base-pair and stacking descriptors.", "frame", None, frozenset({"nucleobases"})),
    AnalyzerSpec("membrane.order", "membrane", "Membrane lipid order and thickness descriptors.", "frame", None, frozenset({"lipid", "periodic_cell"})),
    AnalyzerSpec("solvation.local_environment", "solvation", "Solvent coordination and hydration descriptors.", "frame", None, frozenset({"solvent"})),
    AnalyzerSpec(
        "interactions.hydrogen_bonds", "interactions",
        "Inter-component D-H-A hydrogen bonds with explicit geometric criteria.", "frame", "count",
        frozenset({"coordinates", "bonds", "multiple_molecular_components"}),
        frozenset({"partial_charges"}),
    ),
    AnalyzerSpec("network.contact_graph", "network", "Graph topology of component contacts.", "frame", None, frozenset({"coordinates", "multiple_molecular_components"})),
)


class ProposalCatalog:
    """Capability-based proposals, with implementation evidence from the registry.

    ``assess`` retains the historical family-proposal behavior. Its applicable
    state means that a family might be meaningful, not that its implementation
    exists. Use ``assess_features`` for authoritative feature availability.
    There is deliberately no fuzzy mapping from proposal names to features.
    """

    def __init__(self, analyzers: Iterable[AnalyzerSpec] = DEFAULT_ANALYZERS, *,
                 feature_registry: FeatureRegistry = DEFAULT_FEATURE_REGISTRY):
        self._analyzers = tuple(analyzers)
        self._feature_registry = feature_registry

    def assess(self, profile: SystemProfile) -> tuple[FeatureAssessment, ...]:
        return tuple(analyzer.assess(profile) for analyzer in self._analyzers)

    @property
    def analyzers(self) -> tuple[AnalyzerSpec, ...]:
        return self._analyzers

    @property
    def implemented_features(self) -> tuple[FeatureDefinition, ...]:
        """Registered implementations, without asserting current applicability."""
        return tuple(item for item in self._feature_registry.values() if item.implementation is not None)

    def assess_features(self, profile: SystemProfile, *,
                        available_inputs: Iterable[str]) -> tuple[FeatureAvailability, ...]:
        """Delegate actual feature availability to the authoritative registry."""
        present = frozenset(available_inputs)
        return tuple(self._feature_registry.assess(item.feature_id, profile, available_inputs=present)
                     for item in self._feature_registry.values())


# Preserve existing imports and family-assessment semantics.
AnalyzerCatalog = ProposalCatalog
