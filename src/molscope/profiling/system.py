"""Topology-driven system profiling without molecule-specific assumptions."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
import numpy as np


WATER_RESNAMES = frozenset({"HOH", "WAT", "TIP3", "TIP3P", "SOL", "H2O", "SPC", "SPCE"})
ION_RESNAMES = frozenset({
    "NA", "CL", "K", "MG", "CA", "ZN", "FE", "MN", "CU", "CO", "NI", "BR", "I",
    "NA+", "CL-", "K+", "MG2+", "CA2+", "ZN2+",
})
NUCLEIC_RESNAMES = frozenset({
    "A", "C", "G", "U", "T", "DA", "DC", "DG", "DT", "DU", "ADE", "CYT", "GUA", "THY", "URA",
})
COMMON_LIPID_RESNAMES = frozenset({
    "POPC", "POPE", "POPG", "POPS", "POPA", "DPPC", "DPPE", "DOPC", "DOPE", "DMPC", "DLPC",
    "CHOL", "CARD", "CER", "LPS", "LIPID",
})


@dataclass(frozen=True)
class ComponentProfile:
    """A topology component grouped by identifier and broad chemistry class."""

    component_id: str
    kind: str
    atom_count: int
    residue_count: int
    residue_names: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ComponentSelection:
    """Stable atom membership for a broad topology component."""

    profile: ComponentProfile
    atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class SystemProfile:
    """Capabilities inferred from topology and, optionally, a trajectory frame."""

    topology_path: str
    atom_count: int
    residue_count: int
    components: tuple[ComponentProfile, ...]
    capabilities: frozenset[str]
    has_periodic_cell: bool
    has_bonds: bool
    warnings: tuple[str, ...] = ()
    schema_version: str = "1"

    def has(self, capability: str) -> bool:
        return capability in self.capabilities

    def component_kinds(self) -> tuple[str, ...]:
        return tuple(sorted({component.kind for component in self.components}))

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["components"] = [component.to_dict() for component in self.components]
        payload["capabilities"] = sorted(self.capabilities)
        return payload


def _chain_or_segment(atom_group: mda.core.groups.AtomGroup) -> str:
    """Choose an existing topology identifier without assuming chain A/B."""
    chain_ids = {value.strip() for value in atom_group.chainIDs if value and value.strip()}
    if len(chain_ids) == 1:
        return f"chain:{next(iter(chain_ids))}"
    segids = {value.strip() for value in atom_group.segids if value and value.strip()}
    if len(segids) == 1:
        return f"segment:{next(iter(segids))}"
    return "unlabeled"


def _residue_kind(residue) -> str:
    name = residue.resname.strip().upper()
    if name in WATER_RESNAMES:
        return "solvent"
    if name in ION_RESNAMES:
        return "ion"
    if name in NUCLEIC_RESNAMES:
        return "nucleic_acid"
    if name in COMMON_LIPID_RESNAMES:
        return "lipid"
    # MDAnalysis' protein selection captures standard and many modified amino acids.
    if len(residue.atoms.select_atoms("protein")) == len(residue.atoms) and len(residue.atoms):
        return "protein"
    return "small_molecule_or_polymer"


def _capabilities(components: Iterable[ComponentProfile], universe, *, inspect_trajectory: bool) -> set[str]:
    kinds = {component.kind for component in components}
    capabilities = {"coordinates", "topology", "atom_identity", "residue_identity"}
    if "protein" in kinds:
        capabilities.update({"protein", "protein_backbone", "protein_sidechains"})
    if "nucleic_acid" in kinds:
        capabilities.update({"nucleic_acid", "nucleic_backbone", "nucleobases"})
    if "lipid" in kinds:
        capabilities.add("lipid")
    if "solvent" in kinds:
        capabilities.add("solvent")
    if "ion" in kinds:
        capabilities.add("ions")
    if "small_molecule_or_polymer" in kinds:
        capabilities.add("small_molecule_or_polymer")
    if len([component for component in components if component.kind not in {"solvent", "ion"}]) >= 2:
        capabilities.add("multiple_molecular_components")
    try:
        if len(universe.bonds):
            capabilities.add("bonds")
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    try:
        charges = np.asarray(universe.atoms.charges, dtype=float)
        if len(charges) and np.isfinite(charges).all():
            capabilities.add("partial_charges")
    except (AttributeError, mda.exceptions.NoDataError):
        pass
    elements = getattr(universe.atoms, "elements", ())
    names = getattr(universe.atoms, "names", ())
    if any(str(value).strip().upper() == "H" for value in elements) or any(str(value).strip().upper().startswith("H") for value in names):
        capabilities.add("hydrogens")
    # A PDB CRYST1 record can describe crystallography rather than the MD box.
    # Only a supplied trajectory is accepted as evidence for PBC capability.
    dimensions = getattr(universe, "dimensions", None)
    if inspect_trajectory and dimensions is not None and len(dimensions) >= 3 and all(value > 0 for value in dimensions[:3]):
        capabilities.add("periodic_cell")
    return capabilities


def discover_component_selections(universe) -> tuple[ComponentSelection, ...]:
    """Group atoms by available chain/segment identifier and broad chemistry class."""
    grouped = defaultdict(list)
    for residue in universe.residues:
        kind = _residue_kind(residue)
        identifier = _chain_or_segment(residue.atoms)
        grouped[(identifier, kind)].append(residue)

    selections = []
    for (identifier, kind), residues in sorted(grouped.items()):
        atoms = sum((list(residue.atoms) for residue in residues), [])
        names = tuple(sorted(Counter(residue.resname.strip() for residue in residues)))
        profile = ComponentProfile(
            component_id=f"{identifier}:{kind}",
            kind=kind,
            atom_count=len(atoms),
            residue_count=len(residues),
            residue_names=names,
        )
        selections.append(ComponentSelection(profile=profile, atom_indices=tuple(atom.index for atom in atoms)))
    return tuple(selections)


def profile_system(topology: str | Path, trajectory: str | Path | None = None) -> SystemProfile:
    """Profile a topology, optionally loading a trajectory to inspect box metadata.

    The profile intentionally uses broad component classes.  It reports unknown
    organic components as ``small_molecule_or_polymer`` rather than pretending
    to know their chemistry from a residue name alone.
    """
    topology_path = Path(topology)
    universe = (
        mda.Universe(str(topology_path), str(trajectory))
        if trajectory is not None
        else mda.Universe(str(topology_path))
    )
    if trajectory is not None:
        universe.trajectory[0]

    selections = discover_component_selections(universe)
    components = [selection.profile for selection in selections]

    caps = _capabilities(components, universe, inspect_trajectory=trajectory is not None)
    warnings = []
    if "bonds" not in caps:
        warnings.append("No bond topology is available; bond-dependent analyses require supplied connectivity.")
    if "periodic_cell" not in caps:
        warnings.append("No periodic cell is available; periodic-boundary analyses are not applicable.")
    else:
        warnings.append(
            "Periodic-cell metadata is present; confirm that the simulation used PBC and that coordinates are "
            "properly imaged before interpreting PBC-sensitive analyses."
        )
    if not components:
        warnings.append("No residues were detected in the supplied topology.")

    return SystemProfile(
        topology_path=str(topology_path),
        atom_count=len(universe.atoms),
        residue_count=len(universe.residues),
        components=tuple(components),
        capabilities=frozenset(caps),
        has_periodic_cell="periodic_cell" in caps,
        has_bonds="bonds" in caps,
        warnings=tuple(warnings),
    )
