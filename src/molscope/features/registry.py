"""Machine-readable scientific-method and feature registries.

Registry records describe a calculation before it is run.  They therefore do
not treat a feature value, software support, and scientific applicability as
the same claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Iterable

from molscope.profiling.system import SystemProfile


class FeatureStatus(str, Enum):
    AVAILABLE = "available"
    CALCULATED = "calculated"
    NOT_APPLICABLE = "not_applicable"
    UNSUPPORTED = "unsupported"
    MISSING_INPUT = "missing_input"
    FAILED = "failed"
    SKIPPED_BY_USER = "skipped_by_user"
    SKIPPED_BY_COST_POLICY = "skipped_by_cost_policy"


@dataclass(frozen=True)
class ScientificMethod:
    method_id: str
    question: str
    formulation: str
    physical_interpretation: str
    assumptions: tuple[str, ...]
    applicability_conditions: tuple[str, ...]
    limitations: tuple[str, ...]
    validation_requirements: tuple[str, ...]
    citations: tuple[str, ...]
    version: str = "1"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureDefinition:
    feature_id: str
    name: str
    category: str
    description: str
    method_id: str
    units: str | None
    granularity: str
    molecular_levels: tuple[str, ...]
    applicable_systems: tuple[str, ...]
    required_capabilities: frozenset[str]
    required_inputs: frozenset[str]
    external_data: frozenset[str] = frozenset()
    parameters: dict[str, object] = field(default_factory=dict)
    cost_estimate: str = "unmeasured"
    known_limitations: tuple[str, ...] = ()
    validation_status: str = "implementation_tested_not_reference_validated"
    implementation: str | None = None
    citations: tuple[str, ...] = ()
    version: str = "1"

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["required_capabilities"] = sorted(self.required_capabilities)
        payload["required_inputs"] = sorted(self.required_inputs)
        payload["external_data"] = sorted(self.external_data)
        return payload


@dataclass(frozen=True)
class FeatureAvailability:
    feature_id: str
    status: FeatureStatus
    reason: str
    missing_capabilities: tuple[str, ...] = ()
    missing_inputs: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["status"] = self.status.value
        return payload


class MethodRegistry:
    def __init__(self, methods: Iterable[ScientificMethod] = ()):
        items = tuple(methods)
        self._methods = {method.method_id: method for method in items}
        if len(self._methods) != len(items):
            raise ValueError("Scientific method IDs must be unique.")

    def get(self, method_id: str) -> ScientificMethod:
        return self._methods[method_id]

    def values(self) -> tuple[ScientificMethod, ...]:
        return tuple(self._methods.values())


class FeatureRegistry:
    def __init__(self, definitions: Iterable[FeatureDefinition], methods: MethodRegistry):
        items = tuple(definitions)
        self._definitions = {item.feature_id: item for item in items}
        if len(self._definitions) != len(items):
            raise ValueError("Feature IDs must be unique.")
        missing_methods = sorted({item.method_id for item in items} - {method.method_id for method in methods.values()})
        if missing_methods:
            raise ValueError(f"Features reference unknown scientific methods: {missing_methods}")
        self.methods = methods

    def get(self, feature_id: str) -> FeatureDefinition:
        return self._definitions[feature_id]

    def values(self) -> tuple[FeatureDefinition, ...]:
        return tuple(self._definitions.values())

    def assess(
        self,
        feature_id: str,
        profile: SystemProfile,
        *,
        available_inputs: Iterable[str],
        skipped_by_user: bool = False,
        skipped_by_cost_policy: bool = False,
    ) -> FeatureAvailability:
        """Return an explicit pre-execution state for one registry feature."""
        definition = self.get(feature_id)
        if skipped_by_user:
            return FeatureAvailability(feature_id, FeatureStatus.SKIPPED_BY_USER, "Excluded by the researcher.")
        if skipped_by_cost_policy:
            return FeatureAvailability(feature_id, FeatureStatus.SKIPPED_BY_COST_POLICY, "Excluded by the configured cost policy.")
        if definition.implementation is None:
            return FeatureAvailability(feature_id, FeatureStatus.UNSUPPORTED, "No implementation is registered.")
        missing_capabilities = tuple(sorted(definition.required_capabilities - profile.capabilities))
        if missing_capabilities:
            return FeatureAvailability(
                feature_id, FeatureStatus.NOT_APPLICABLE,
                "Required molecular capabilities are absent.", missing_capabilities=missing_capabilities,
            )
        present = frozenset(available_inputs)
        missing_inputs = tuple(sorted((definition.required_inputs | definition.external_data) - present))
        if missing_inputs:
            return FeatureAvailability(
                feature_id, FeatureStatus.MISSING_INPUT,
                "Required input data are unavailable.", missing_inputs=missing_inputs,
            )
        return FeatureAvailability(feature_id, FeatureStatus.AVAILABLE, "Requirements and implementation are available.")


METHODS = MethodRegistry((
    ScientificMethod(
        method_id="geometry.radius_of_gyration",
        question="How spatially extended is the selected molecular scope?",
        formulation="Root mean square distance of atoms from their geometric center.",
        physical_interpretation="A geometric compactness descriptor, not a direct free-energy or stability measure.",
        assumptions=("The selected atom set is scientifically meaningful.",),
        applicability_conditions=("Coordinates are available for the selected scope.",),
        limitations=("Mass weighting and alignment choices change the observable.",),
        validation_requirements=("Compare selected-scope values with a trusted reference implementation.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="geometry.component_proximity",
        question="How do selected molecular components approach one another?",
        formulation="Center-of-geometry separation and a declared all-atom contact-pair fraction.",
        physical_interpretation="A coarse geometric description of component proximity.",
        assumptions=("Component boundaries and contact cutoff are declared.",),
        applicability_conditions=("At least two non-environment components have coordinates.",),
        limitations=("A distance cutoff is an operational definition, not a universal interaction criterion.",),
        validation_requirements=("Test cutoff sensitivity and periodic-boundary preprocessing.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="dynamics.aligned_displacement",
        question="How does a selected structure deviate from a declared reference?",
        formulation="Least-squares rigid-body alignment followed by root mean square displacement.",
        physical_interpretation="Reference-relative structural displacement.",
        assumptions=("The atom mapping and reference are valid.",),
        applicability_conditions=("Mapped coordinates are available.",),
        limitations=("Results depend on the alignment and atom-selection policy.",),
        validation_requirements=("Verify atom mapping and compare alignment results with a reference implementation.",),
        citations=("doi:10.1107/S0567739476001873",),
    ),
    ScientificMethod(
        method_id="dynamics.rmsf",
        question="What is the fluctuation magnitude over a trajectory?",
        formulation="Root mean square displacement around a trajectory mean after declared alignment.",
        physical_interpretation="A trajectory-level fluctuation estimate.",
        assumptions=("The trajectory section represents the ensemble of interest.",),
        applicability_conditions=("Multiple mapped coordinate frames are available.",),
        limitations=("Not a frame-level independent observation; convergence requires sampling assessment.",),
        validation_requirements=("Assess block uncertainty and independent-replica consistency.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="interactions.protein_ligand_geometric_contacts",
        question="Which protein residues are geometrically proximal to a declared ligand component?",
        formulation="Counts and fractions of protein-ligand atom pairs within a declared distance cutoff.",
        physical_interpretation="Operational geometric contact evidence, not a chemical interaction classification or binding affinity.",
        assumptions=("The ligand component is correctly identified.", "The distance cutoff is declared and scientifically appropriate."),
        applicability_conditions=("Protein coordinates and exactly one ligand-like component are available.",),
        limitations=("No PBC correction in the initial plugin.", "Distance contacts do not establish hydrogen bonding, binding energy, or causality."),
        validation_requirements=("Check ligand identity and cutoff sensitivity.", "Compare contacts against a trusted structural-analysis workflow for a benchmark system."),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="interactions.protein_salt_bridge_geometry",
        question="Are declared acidic and basic protein side-chain atoms geometrically proximal?",
        formulation="Count ASP/GLU oxygen to LYS/ARG nitrogen pairs within a declared cutoff.",
        physical_interpretation="A geometric salt-bridge proxy, not an electrostatic energy calculation.",
        assumptions=("Standard residue and atom naming is present.", "Protonation and cutoff policies are declared."),
        applicability_conditions=("Protein coordinates include the declared acidic and basic atom sets.",),
        limitations=("Histidine is excluded due to protonation ambiguity.", "No PBC correction in the initial plugin."),
        validation_requirements=("Test cutoff and protonation sensitivity.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="interactions.protein_hydrophobic_contact_geometry",
        question="Which nonlocal declared hydrophobic side-chain carbons are geometrically packed?",
        formulation="Count retained carbon-carbon pairs within a cutoff after sequence-separation filtering.",
        physical_interpretation="A packing proxy, not a hydrophobic free-energy measurement.",
        assumptions=("Hydrophobic residue set and sequence-separation policy are appropriate."),
        applicability_conditions=("Protein coordinates include at least two declared hydrophobic carbon atoms.",),
        limitations=("No PBC correction in the initial plugin.", "Results depend on residue and cutoff policy."),
        validation_requirements=("Test residue-set and cutoff sensitivity."),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="protein.backbone_torsion_geometry",
        question="Which protein backbone torsional conformations are sampled?",
        formulation="Phi, psi, and omega dihedrals represented by sine and cosine components.",
        physical_interpretation="Local backbone conformation descriptors that respect angular periodicity.",
        assumptions=("Standard protein backbone connectivity is available.",),
        applicability_conditions=("Protein backbone atoms and bond topology are available.",),
        limitations=("Terminal and incomplete residues do not yield every torsion.", "Torsions alone do not define a conformational state."),
        validation_requirements=("Compare named-residue torsions with a trusted dihedral implementation.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="protein.ca_distance_contact_geometry",
        question="How do nonlocal protein Cα residue pairs separate and contact under a declared convention?",
        formulation="Pairwise Cα distances and binary contacts at an 8.0 Å cutoff after a same-chain sequence-separation filter.",
        physical_interpretation="Coarse residue geometry and contact-map descriptors, not a complete structural interaction classification.",
        assumptions=("Cα atoms represent the intended residue scope.", "The distance cutoff and sequence-separation policy are suitable for the question."),
        applicability_conditions=("Protein coordinates contain at least two Cα atoms and at least one retained pair.",),
        limitations=("Initial implementation is not PBC-aware.", "Cα contacts omit side-chain geometry and do not establish physical interactions."),
        validation_requirements=("Compare selected residue-pair distances and contacts against a trusted geometry implementation with matched cutoff and sequence policy.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="protein.secondary_structure_dssp",
        question="What fraction of supported protein residues are assigned helix, sheet, or coil by the declared DSSP backend?",
        formulation="MDAnalysis native DSSP assignment summarized as framewise H, E, and '-' fractions.",
        physical_interpretation="A backend-specific secondary-structure descriptor, not direct evidence of folding free energy or functional state.",
        assumptions=("Protein backbone atom naming and geometry are supported by the selected DSSP backend.",),
        applicability_conditions=("Protein coordinates and the MDAnalysis DSSP module are available with complete supported backbone geometry.",),
        limitations=("Initial implementation is not PBC-aware.", "Only aggregate H/E/coil fractions are emitted; assignments depend on backend version and structure preparation."),
        validation_requirements=("Compare per-frame aggregate fractions with the same MDAnalysis DSSP backend and a benchmark structure set.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="structural.solvent_accessible_surface_area",
        question="What solvent-accessible area is exposed by the selected structural scope?",
        formulation="Shrake-Rupley surface sampling with a declared probe radius and sampling density.",
        physical_interpretation="Geometric solvent exposure, not solvation free energy.",
        assumptions=("Atomic radii and the non-solvent/non-ion scope are appropriate.",),
        applicability_conditions=("Coordinates and the optional FreeSASA backend are available.",),
        limitations=("Initial implementation is not PBC-aware.", "Values depend on radii, probe radius, and structure preparation."),
        validation_requirements=("Compare total and residue values with a trusted SASA calculation using matched parameters.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="interactions.protein_hydrogen_bond_geometry",
        question="Which supported protein donor-hydrogen-acceptor geometries meet declared hydrogen-bond criteria?",
        formulation="Donor-acceptor distance and donor-H-acceptor angle filters over named protein atoms.",
        physical_interpretation="Directional geometric hydrogen-bond evidence, not an interaction energy.",
        assumptions=("Explicit hydrogens and bond connectivity are correct.", "Atom naming and geometry thresholds are suitable."),
        applicability_conditions=("Protein coordinates, bonds, and explicit hydrogens are available.",),
        limitations=("The supported atom list is intentionally limited.", "Initial implementation is not PBC-aware."),
        validation_requirements=("Compare occupancies and geometry against a trusted hydrogen-bond analysis with matched rules.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="interactions.protein_aromatic_stacking_geometry",
        question="Which supported aromatic residue pairs satisfy declared stacking geometry?",
        formulation="PHE/TYR ring-center distances and ring-plane angular filters.",
        physical_interpretation="A conservative aromatic packing geometry proxy.",
        assumptions=("Complete PHE/TYR rings are present and ring-plane criteria are appropriate."),
        applicability_conditions=("Protein coordinates with at least two complete supported aromatic rings are available.",),
        limitations=("TRP, HIS, cation-pi interactions, and offset geometry are not represented.", "Initial implementation is not PBC-aware."),
        validation_requirements=("Check ring identification and compare pair calls with a trusted structural analysis.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="solvation.water_shell_geometry",
        question="How many identified water oxygens occupy declared molecular-component shells?",
        formulation="Counts water oxygens within a declared distance of each non-solvent/non-ion component, including shared shells.",
        physical_interpretation="Local solvent occupancy and operational water-bridge evidence, not hydration free energy.",
        assumptions=("Solvent and water oxygen naming are correctly identified.",),
        applicability_conditions=("Coordinates and topology-identified solvent are available.",),
        limitations=("Initial implementation is not PBC-aware.", "Counts depend on shell cutoff and imaging."),
        validation_requirements=("Test cutoff sensitivity and compare shell counts with an imaged reference trajectory.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="electrostatics.molecular_dipole_moment",
        question="What is the instantaneous dipole magnitude of a neutral selected molecular scope?",
        formulation="Sum of supplied partial charges times positions for the non-solvent/non-ion scope.",
        physical_interpretation="Charge-distribution orientation and magnitude descriptor.",
        assumptions=("Partial charges are finite and compatible with the force field.", "The selected scope is neutral."),
        applicability_conditions=("Coordinates and finite partial charges are available for a neutral selected scope.",),
        limitations=("Dipoles of non-neutral scopes are origin-dependent and rejected.", "Initial implementation is not PBC-aware."),
        validation_requirements=("Verify units, neutrality, and values against an independent charge-weighted calculation.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="networks.protein_residue_contact_topology",
        question="How does the topology of a declared protein residue-contact graph vary over frames?",
        formulation="Nonlocal heavy-atom residue contacts define an unweighted graph with frame metrics and edge persistence.",
        physical_interpretation="Contact-network topology descriptors, not evidence of communication or allostery.",
        assumptions=("Protein residue identity and contact cutoff are appropriate.",),
        applicability_conditions=("Protein coordinates with at least two residues containing heavy atoms are available.",),
        limitations=("Initial implementation is not PBC-aware.", "Exact metrics are skipped above the configured size policy."),
        validation_requirements=("Test cutoff sensitivity and compare graph edges and metrics with a reference implementation.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
    ScientificMethod(
        method_id="nucleic_acid.base_center_proximity",
        question="Which nucleic-acid base centers are geometrically proximal under a declared cutoff?",
        formulation="Distances between centers of recognized nucleobase atoms, with proximity occupancy across frames.",
        physical_interpretation="A coarse base-geometry descriptor, not a base-pair, stacking, or hydrogen-bond classification.",
        assumptions=("Recognized atom names identify a meaningful base-atom subset.",),
        applicability_conditions=("At least two nucleic residues with recognized base atoms have coordinates.",),
        limitations=("Initial implementation is not PBC-aware.", "Center proximity does not establish pairing, stacking, or sequence-specific interactions."),
        validation_requirements=("Compare recognized base centers and distances with a trusted nucleic-acid geometry workflow.",),
        citations=("doi:10.1002/jcc.21787",),
    ),
))


DEFAULT_FEATURE_REGISTRY = FeatureRegistry((
    FeatureDefinition(
        feature_id="geometry.global_radius_of_gyration",
        name="Global geometric radius of gyration",
        category="geometry",
        description="Unweighted radius of gyration of all selected atoms.",
        method_id="geometry.radius_of_gyration",
        units="angstrom",
        granularity="frame",
        molecular_levels=("system",),
        applicable_systems=("any_coordinate_system",),
        required_capabilities=frozenset({"coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        cost_estimate="low_unmeasured",
        known_limitations=("Default scope excludes solvent and ions; use an explicit environment-inclusive scope when that is the scientific question.",),
        implementation="molscope.generic_features.GenericFeatureExtractor",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="dynamics.kabsch_rmsd_to_first_frame",
        name="Kabsch-aligned RMSD to first frame",
        category="dynamics",
        description="Reference-relative RMSD using the first analyzed frame as the reference.",
        method_id="dynamics.aligned_displacement",
        units="angstrom",
        granularity="frame",
        molecular_levels=("system",),
        applicable_systems=("any_coordinate_system",),
        required_capabilities=frozenset({"coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        cost_estimate="low_unmeasured",
        known_limitations=("The first-frame reference is a convenience choice, not an equilibrium reference.",),
        implementation="molscope.generic_features.GenericFeatureExtractor",
        citations=("doi:10.1107/S0567739476001873",),
    ),
    FeatureDefinition(
        feature_id="geometry.component_contact_fraction",
        name="Component all-atom contact-pair fraction",
        category="geometry",
        description="Fraction of cross-component atom pairs inside the configured distance cutoff.",
        method_id="geometry.component_proximity",
        units="fraction",
        granularity="frame",
        molecular_levels=("component_pair",),
        applicable_systems=("multi_component_coordinate_system",),
        required_capabilities=frozenset({"coordinates", "multiple_molecular_components"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"contact_cutoff_angstrom": 4.5},
        cost_estimate="low_unmeasured",
        implementation="molscope.generic_features.GenericFeatureExtractor",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="dynamics.global_rmsf_mean",
        name="Global mean RMSF",
        category="dynamics",
        description="Mean all-atom RMSF across the analyzed trajectory.",
        method_id="dynamics.rmsf",
        units="angstrom",
        granularity="trajectory",
        molecular_levels=("system",),
        applicable_systems=("any_coordinate_system",),
        required_capabilities=frozenset({"coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        cost_estimate="low_unmeasured",
        implementation="molscope.generic_features.GenericFeatureExtractor",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="interactions.protein_ligand_residue_contact_fraction",
        name="Protein-ligand residue contact fraction",
        category="interactions",
        description="Per-residue fraction of cross protein-ligand atom pairs inside the declared cutoff.",
        method_id="interactions.protein_ligand_geometric_contacts",
        units="fraction",
        granularity="frame",
        molecular_levels=("protein_residue", "ligand"),
        applicable_systems=("single_ligand_protein_system",),
        required_capabilities=frozenset({"coordinates", "protein", "small_molecule_or_polymer"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"contact_cutoff_angstrom": 4.5, "pbc": False, "single_ligand_component_required": True},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Not applicable when ligand scope is ambiguous.", "Geometric proximity is not a chemical interaction type or binding energy."),
        validation_status="matched_policy_reference_fixture",
        implementation="molscope.plugins.ProteinLigandContactPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="interactions.protein_salt_bridge_pair_count",
        name="Protein salt-bridge atom-pair count",
        category="interactions",
        description="Declared acidic-basic side-chain atom pairs within 4.0 Å.",
        method_id="interactions.protein_salt_bridge_geometry",
        units="count",
        granularity="frame",
        molecular_levels=("protein_residue_pair",),
        applicable_systems=("protein",),
        required_capabilities=frozenset({"coordinates", "protein"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"cutoff_angstrom": 4.0, "histidine_included": False, "pbc": False},
        cost_estimate="low_unmeasured",
        known_limitations=("A geometric proxy, not electrostatic energy.",),
        validation_status="matched_policy_reference_fixture",
        implementation="molscope.plugins.ProteinSaltBridgePlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="interactions.protein_hydrophobic_contact_pair_count",
        name="Protein hydrophobic contact-pair count",
        category="interactions",
        description="Nonlocal declared hydrophobic carbon-carbon pairs within 4.5 Å.",
        method_id="interactions.protein_hydrophobic_contact_geometry",
        units="count",
        granularity="frame",
        molecular_levels=("protein_residue_pair",),
        applicable_systems=("protein",),
        required_capabilities=frozenset({"coordinates", "protein"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"cutoff_angstrom": 4.5, "minimum_sequence_separation": 3, "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("A packing proxy, not hydrophobic free energy.",),
        implementation="molscope.plugins.ProteinHydrophobicContactPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="protein.backbone_torsions",
        name="Protein backbone torsion sine/cosine components",
        category="protein_chemistry",
        description="Per-residue phi, psi, and omega values encoded as sine and cosine components.",
        method_id="protein.backbone_torsion_geometry",
        units="unitless",
        granularity="frame",
        molecular_levels=("protein_residue",),
        applicable_systems=("protein_with_bond_topology",),
        required_capabilities=frozenset({"protein_backbone", "bonds"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"torsions": ["phi", "psi", "omega"], "encoding": "sin_cos"},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Termini and incomplete residues are omitted.",),
        implementation="molscope.plugins.ProteinBackboneTorsionPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="protein.ca_distances_contacts",
        name="Protein nonlocal Cα distances and contacts",
        category="protein_geometry",
        description="Per-pair Cα distances, binary contacts, and trajectory contact occupancies under declared nonlocality and cutoff policies.",
        method_id="protein.ca_distance_contact_geometry",
        units="angstrom_and_fraction",
        granularity="frame_and_trajectory",
        molecular_levels=("protein_residue_pair",),
        applicable_systems=("protein_with_ca_atoms",),
        required_capabilities=frozenset({"protein", "coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"atom_selection": "protein and name CA", "contact_cutoff_angstrom": 8.0, "minimum_sequence_separation": 3, "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("No PBC correction.", "Cα contacts are coarse geometric proxies and can be high-dimensional for large proteins."),
        implementation="molscope.plugins.ProteinCADistanceContactPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="protein.secondary_structure_dssp",
        name="Protein DSSP secondary-structure fractions",
        category="protein_structure",
        description="Framewise helix, sheet, and coil fractions from the MDAnalysis native DSSP backend.",
        method_id="protein.secondary_structure_dssp",
        units="fraction",
        granularity="frame_and_trajectory",
        molecular_levels=("protein",),
        applicable_systems=("protein_with_supported_backbone_geometry",),
        required_capabilities=frozenset({"protein", "coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"backend": "MDAnalysis.analysis.dssp.DSSP", "helix_code": "H", "sheet_code": "E", "coil_code": "-", "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("DSSP assignment depends on complete backbone geometry and backend version.", "No PBC correction; aggregate fractions are not a residue-level mechanistic claim."),
        implementation="molscope.plugins.ProteinDSSPPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="structural.sasa",
        name="Non-solvent/non-ion solvent-accessible surface area",
        category="structural",
        description="Total and per-residue SASA from the optional FreeSASA backend.",
        method_id="structural.solvent_accessible_surface_area",
        units="angstrom_squared",
        granularity="frame",
        molecular_levels=("system", "residue"),
        applicable_systems=("any_coordinate_system",),
        required_capabilities=frozenset({"coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        external_data=frozenset({"freesasa_backend"}),
        parameters={"probe_radius_angstrom": 1.4, "n_points": 100, "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Requires optional FreeSASA installation.", "Trajectories must be imaged before PBC-sensitive interpretation."),
        validation_status="matched_policy_reference_fixture",
        implementation="molscope.plugins.SASAPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="interactions.protein_hydrogen_bonds",
        name="Protein directional hydrogen-bond count",
        category="interactions",
        description="Supported protein donor-H-acceptor geometries satisfying declared distance and angle criteria.",
        method_id="interactions.protein_hydrogen_bond_geometry",
        units="count",
        granularity="frame",
        molecular_levels=("protein_residue_pair",),
        applicable_systems=("protein_with_explicit_hydrogens_and_bonds",),
        required_capabilities=frozenset({"protein", "coordinates", "bonds", "hydrogens"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"donor_acceptor_cutoff_angstrom": 3.5, "minimum_angle_degrees": 150.0, "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Supported atom names are limited.", "This is not an interaction-energy calculation."),
        implementation="molscope.plugins.ProteinHydrogenBondPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="interactions.protein_aromatic_stacking",
        name="Protein PHE/TYR aromatic-stacking count",
        category="interactions",
        description="PHE/TYR ring pairs satisfying declared center-distance and plane-angle geometry.",
        method_id="interactions.protein_aromatic_stacking_geometry",
        units="count",
        granularity="frame",
        molecular_levels=("protein_residue_pair",),
        applicable_systems=("protein",),
        required_capabilities=frozenset({"protein", "coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"center_distance_angstrom": [3.5, 7.0], "maximum_plane_angle_degrees": 30.0, "residue_types": ["PHE", "TYR"], "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Other aromatic residue types and cation-pi interactions are excluded.",),
        implementation="molscope.plugins.ProteinAromaticStackingPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="solvation.water_shells",
        name="Component water-shell occupancy",
        category="solvation",
        description="Water-oxygen counts around non-solvent/non-ion components and shared shell counts.",
        method_id="solvation.water_shell_geometry",
        units="count",
        granularity="frame",
        molecular_levels=("component", "component_pair"),
        applicable_systems=("solvated_coordinate_system",),
        required_capabilities=frozenset({"coordinates", "solvent"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"cutoff_angstrom": 3.5, "water_oxygen_names": ["O", "OW", "OH2"], "pbc": False},
        cost_estimate="moderate_unmeasured",
        known_limitations=("Water naming and trajectory imaging affect results.", "This is not a hydration free-energy calculation."),
        validation_status="matched_policy_reference_fixture",
        implementation="molscope.plugins.SolvationShellPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="electrostatics.molecular_dipole",
        name="Neutral-scope molecular dipole magnitude",
        category="electrostatics",
        description="Charge-weighted dipole magnitude for the selected neutral non-solvent/non-ion scope.",
        method_id="electrostatics.molecular_dipole_moment",
        units="elementary_charge_angstrom",
        granularity="frame",
        molecular_levels=("system",),
        applicable_systems=("neutral_charged_coordinate_system",),
        required_capabilities=frozenset({"coordinates", "partial_charges"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"neutrality_tolerance_e": 1e-6, "pbc": False},
        cost_estimate="low_unmeasured",
        known_limitations=("Non-neutral scopes are rejected because their dipole is origin-dependent.",),
        implementation="molscope.plugins.MolecularDipolePlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="networks.protein_contact_network",
        name="Protein residue-contact network metrics",
        category="networks",
        description="Framewise nonlocal heavy-atom residue-contact graph metrics and trajectory edge persistence.",
        method_id="networks.protein_residue_contact_topology",
        units="mixed",
        granularity="frame_and_trajectory",
        molecular_levels=("protein_residue_pair", "protein"),
        applicable_systems=("protein",),
        required_capabilities=frozenset({"protein", "coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"cutoff_angstrom": 4.5, "minimum_sequence_separation": 3, "max_exact_network_nodes": 400, "pbc": False},
        cost_estimate="high_unmeasured",
        known_limitations=("Exact metrics are skipped above the node-size policy.", "Network topology does not establish allostery or causality."),
        implementation="molscope.plugins.ProteinContactNetworkPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
    FeatureDefinition(
        feature_id="nucleic_acid.base_proximities",
        name="Nucleic-acid base-center proximity",
        category="nucleic_acid",
        description="Recognized nucleobase-center distances, proximal-pair counts, and trajectory occupancies.",
        method_id="nucleic_acid.base_center_proximity",
        units="angstrom",
        granularity="frame_and_trajectory",
        molecular_levels=("nucleic_residue_pair",),
        applicable_systems=("nucleic_acid",),
        required_capabilities=frozenset({"nucleic_acid", "coordinates"}),
        required_inputs=frozenset({"topology", "trajectory"}),
        parameters={"maximum_center_distance_angstrom": 7.0, "minimum_recognized_base_atoms": 3, "pbc": False},
        cost_estimate="low_unmeasured",
        known_limitations=("Not a base-pairing or stacking classifier.", "Recognized atom names and imaging affect results."),
        validation_status="matched_policy_reference_fixture",
        implementation="molscope.plugins.NucleicAcidBaseProximityPlugin",
        citations=("doi:10.1002/jcc.21787",),
    ),
), METHODS)
