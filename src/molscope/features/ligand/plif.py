"""Protein-Ligand Interaction Fingerprints (PLIF) feature extraction plugin."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
import networkx as nx
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.profiling.system import SystemProfile, profile_system
from molscope.features.base import PluginResult, FeaturePlugin, _safe_id, _chain_or_segment_id


PROTEIN_DONOR_SELECTION = (
    "protein and (name N NE NH1 NH2 NZ ND1 NE2 ND2 SG OG OG1 OH "
    "or (element N O and not name O OXT OD1 OD2 OE1 OE2))"
)
PROTEIN_ACCEPTOR_SELECTION = (
    "protein and (name O OXT OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH SG SD "
    "or element O)"
)
PROTEIN_ACIDIC_SELECTION = "protein and resname ASP GLU and name OD1 OD2 OE1 OE2"
PROTEIN_BASIC_SELECTION = "protein and resname LYS ARG and name NZ NH1 NH2 NE"

PHE_TYR_RING_NAMES = frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"})
HIS_RING_NAMES = frozenset({"CG", "ND1", "CD2", "CE1", "NE2"})
TRP_5RING_NAMES = frozenset({"CG", "CD1", "NE1", "CE2", "CD2"})
TRP_6RING_NAMES = frozenset({"CD2", "CE2", "CZ2", "CH2", "CZ3", "CE3"})
HIS_RESNAMES = frozenset({"HIS", "HSD", "HSE", "HSP", "HIE", "HID", "HIP"})


def _get_heavy_atoms(atom_group: mda.core.groups.AtomGroup) -> mda.core.groups.AtomGroup:
    """Filter out hydrogen atoms."""
    try:
        return atom_group.select_atoms("not (element H or name H*)")
    except Exception:
        elements = getattr(atom_group, "elements", None)
        if elements is not None and len(elements) == len(atom_group):
            mask = np.array([str(e).strip().upper() != "H" for e in elements], dtype=bool)
            return atom_group[mask]
        mask = np.array([not atom.name.strip().upper().startswith("H") for atom in atom_group], dtype=bool)
        return atom_group[mask]


def _build_covalent_adjacency(atoms: mda.core.groups.AtomGroup, max_bond_distance: float = 1.85) -> nx.Graph:
    """Build covalent bond graph for an atom group using topology bonds or distance."""
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    has_bonds = hasattr(atoms.universe, "bonds") and len(atoms.universe.bonds) > 0
    if has_bonds:
        atom_set = set(atoms.indices)
        index_map = {idx: i for i, idx in enumerate(atoms.indices)}
        for atom in atoms:
            u_idx = index_map[atom.index]
            for neighbor in atom.bonded_atoms:
                if neighbor.index in atom_set:
                    graph.add_edge(u_idx, index_map[neighbor.index])
    else:
        if len(atoms) >= 2:
            pairs, _ = capped_distance(atoms.positions, atoms.positions, min_cutoff=0.4, max_cutoff=max_bond_distance)
            for left, right in pairs:
                if left < right:
                    graph.add_edge(int(left), int(right))
    return graph


def _find_nonpolar_carbons(
    atom_group: mda.core.groups.AtomGroup,
    context_atoms: mda.core.groups.AtomGroup,
) -> mda.core.groups.AtomGroup:
    """Identify carbon atoms not covalently bonded to electronegative heteroatoms."""
    try:
        carbons = atom_group.select_atoms("element C")
        if len(carbons) == 0:
            carbons = atom_group.select_atoms("name C*")
    except Exception:
        carbons = atom_group[[i for i, a in enumerate(atom_group) if a.name.strip().upper().startswith("C")]]
    if len(carbons) == 0:
        return atom_group[[]]

    try:
        heteroatoms = context_atoms.select_atoms(
            "(element N O S P F Cl Br I) or (not element C and (name N* O* S* P* F* CL* BR* I*))"
        )
    except Exception:
        hetero_names = ("N", "O", "S", "P", "F", "CL", "BR", "I")
        heteroatoms = context_atoms[[
            i for i, a in enumerate(context_atoms)
            if getattr(a, "element", "").strip().upper() != "C"
            and any(a.name.strip().upper().startswith(h) for h in hetero_names)
        ]]

    if len(heteroatoms) == 0:
        return carbons

    pairs, _ = capped_distance(carbons.positions, heteroatoms.positions, max_cutoff=1.7)
    polar_carbon_indices = set(pairs[:, 0]) if len(pairs) > 0 else set()
    nonpolar_indices = [i for i in range(len(carbons)) if i not in polar_carbon_indices]
    return carbons[nonpolar_indices]


def _find_ligand_donors_and_acceptors(
    ligand: mda.core.groups.AtomGroup,
) -> tuple[mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """Identify hydrogen-bond donor and acceptor atoms in ligand."""
    heavy_ligand = _get_heavy_atoms(ligand)
    if len(heavy_ligand) == 0:
        return ligand[[]], ligand[[]]

    adj = _build_covalent_adjacency(heavy_ligand)
    has_hydrogens = len(ligand) > len(heavy_ligand)

    donor_indices = []
    acceptor_indices = []

    for i, atom in enumerate(heavy_ligand):
        element = getattr(atom, "element", "").strip().upper()
        name = atom.name.strip().upper()
        is_n = element == "N" or (not element and name.startswith("N"))
        is_o = element == "O" or (not element and name.startswith("O"))
        is_s = element == "S" or (not element and name.startswith("S"))

        if not (is_n or is_o or is_s):
            continue

        degree = adj.degree[i]

        # Acceptors: O, S, or N with <= 3 heavy neighbors
        if is_o or is_s:
            acceptor_indices.append(atom.index)
        elif is_n and degree <= 3:
            acceptor_indices.append(atom.index)

        # Donors:
        if has_hydrogens:
            # Check for attached hydrogen
            h_count = 0
            if hasattr(atom, "bonded_atoms") and len(atom.universe.bonds) > 0:
                h_count = sum(
                    1 for nb in atom.bonded_atoms
                    if getattr(nb, "element", "").strip().upper() == "H" or nb.name.strip().upper().startswith("H")
                )
            else:
                h_pairs, _ = capped_distance(np.array([atom.position]), ligand.positions, max_cutoff=1.2)
                for _, h_idx in h_pairs:
                    h_atom = ligand[h_idx]
                    if getattr(h_atom, "element", "").strip().upper() == "H" or h_atom.name.strip().upper().startswith("H"):
                        h_count += 1
            if h_count > 0:
                donor_indices.append(atom.index)
        else:
            # Heavy-atom topology heuristic
            if is_n and degree <= 2:
                donor_indices.append(atom.index)
            elif is_o and degree <= 1:
                donor_indices.append(atom.index)
            elif is_s and degree <= 1:
                donor_indices.append(atom.index)

    donors = ligand.universe.atoms[donor_indices] if donor_indices else ligand[[]]
    acceptors = ligand.universe.atoms[acceptor_indices] if acceptor_indices else ligand[[]]
    return donors, acceptors


def _find_ligand_aromatic_rings(
    ligand: mda.core.groups.AtomGroup,
    max_planarity_deviation: float = 0.35,
) -> list[mda.core.groups.AtomGroup]:
    """Find planar 5- and 6-membered aromatic rings in ligand."""
    heavy_ligand = _get_heavy_atoms(ligand)
    if len(heavy_ligand) < 5:
        return []

    adj = _build_covalent_adjacency(heavy_ligand)
    cycles = nx.cycle_basis(adj)
    candidate_cycles = [c for c in cycles if len(c) in (5, 6)]

    aromatic_rings = []
    for cycle in candidate_cycles:
        ring_atoms = heavy_ligand[cycle]
        # Check elements (must be aromatic-compatible: C, N, O, S)
        valid_elements = True
        for a in ring_atoms:
            elem = getattr(a, "element", "").strip().upper()
            name = a.name.strip().upper()
            is_valid = (
                elem in {"C", "N", "O", "S"}
                or any(name.startswith(sym) for sym in ("C", "N", "O", "S"))
            )
            if not is_valid:
                valid_elements = False
                break
        if not valid_elements:
            continue

        # Check planarity using SVD
        positions = ring_atoms.positions
        center = positions.mean(axis=0)
        centered = positions - center
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]
        out_of_plane = np.max(np.abs(np.dot(centered, normal)))
        if out_of_plane <= max_planarity_deviation:
            aromatic_rings.append(ring_atoms)

    return aromatic_rings


def _find_ligand_charged_atoms(
    ligand: mda.core.groups.AtomGroup,
) -> tuple[mda.core.groups.AtomGroup, mda.core.groups.AtomGroup]:
    """Detect positively and negatively charged ligand atoms or groups."""
    positive_indices: list[int] = []
    negative_indices: list[int] = []

    # Check partial/formal charges if assigned
    charges = getattr(ligand, "charges", None)
    if charges is not None and len(charges) == len(ligand) and np.any(np.abs(charges) >= 0.3):
        for atom, charge in zip(ligand, charges):
            if charge >= 0.3:
                positive_indices.append(atom.index)
            elif charge <= -0.3:
                negative_indices.append(atom.index)

    # Chemical functional group recognition (from topology / geometry)
    heavy = _get_heavy_atoms(ligand)
    if len(heavy) > 0:
        adj = _build_covalent_adjacency(heavy)
        for i, atom in enumerate(heavy):
            elem = getattr(atom, "element", "").strip().upper()
            name = atom.name.strip().upper()
            degree = adj.degree[i]

            # Carboxylate detection: C bonded to 2 oxygens
            if elem == "C" or (not elem and name.startswith("C")):
                neighbor_oxygens = [
                    heavy[nb].index for nb in adj.neighbors(i)
                    if getattr(heavy[nb], "element", "").strip().upper() == "O"
                    or heavy[nb].name.strip().upper().startswith("O")
                ]
                if len(neighbor_oxygens) == 2:
                    for o_idx in neighbor_oxygens:
                        if o_idx not in negative_indices:
                            negative_indices.append(o_idx)

            # Phosphate / Sulfonate detection
            elif elem in {"P", "S"} or (not elem and (name.startswith("P") or name.startswith("S"))):
                neighbor_oxygens = [
                    heavy[nb].index for nb in adj.neighbors(i)
                    if getattr(heavy[nb], "element", "").strip().upper() == "O"
                    or heavy[nb].name.strip().upper().startswith("O")
                ]
                if len(neighbor_oxygens) >= 3:
                    for o_idx in neighbor_oxygens:
                        if o_idx not in negative_indices:
                            negative_indices.append(o_idx)

            # Basic aliphatic amine / guanidinium / amidine: N without carbonyl adjacent
            elif elem == "N" or (not elem and name.startswith("N")):
                # Check neighbors: if bonded to C=O (amide), not basic
                is_amide = False
                c_neighbors = [
                    nb for nb in adj.neighbors(i)
                    if getattr(heavy[nb], "element", "").strip().upper() == "C"
                    or heavy[nb].name.strip().upper().startswith("C")
                ]
                for c_nb in c_neighbors:
                    # check if c_nb has an O neighbor
                    o_of_c = [
                        heavy[o_nb] for o_nb in adj.neighbors(c_nb)
                        if o_nb != i and (
                            getattr(heavy[o_nb], "element", "").strip().upper() == "O"
                            or heavy[o_nb].name.strip().upper().startswith("O")
                        )
                    ]
                    if len(o_of_c) > 0:
                        is_amide = True
                        break

                if not is_amide and 1 <= degree <= 4:
                    if atom.index not in positive_indices:
                        positive_indices.append(atom.index)

    pos_group = ligand.universe.atoms[positive_indices] if positive_indices else ligand[[]]
    neg_group = ligand.universe.atoms[negative_indices] if negative_indices else ligand[[]]
    return pos_group, neg_group


class ProteinLigandInteractionFingerprintPlugin:
    """Protein-Ligand Interaction Fingerprints (PLIF) feature extraction plugin."""

    plugin_id = "ligand.plif"
    version = "1"
    required_capabilities = frozenset({"protein", "small_molecule_or_polymer", "coordinates"})

    def __init__(
        self,
        ligand_selection: str | None = None,
        hydrophobic_cutoff_angstrom: float = 4.5,
        hbond_cutoff_angstrom: float = 3.5,
        aromatic_cutoff_angstrom: float = 5.0,
        salt_bridge_cutoff_angstrom: float = 4.5,
        protein_selection: str = "protein",
    ):
        if ligand_selection is not None:
            if not isinstance(ligand_selection, str) or not ligand_selection.strip():
                raise ValueError("ligand_selection must be a non-empty string if provided.")
            self.ligand_selection: str | None = ligand_selection.strip()
        else:
            self.ligand_selection = None

        if not isinstance(protein_selection, str) or not protein_selection.strip():
            raise ValueError("protein_selection must be a non-empty string.")
        self.protein_selection = protein_selection.strip()

        for name, val in (
            ("hydrophobic_cutoff_angstrom", hydrophobic_cutoff_angstrom),
            ("hbond_cutoff_angstrom", hbond_cutoff_angstrom),
            ("aromatic_cutoff_angstrom", aromatic_cutoff_angstrom),
            ("salt_bridge_cutoff_angstrom", salt_bridge_cutoff_angstrom),
        ):
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not np.isfinite(val) or val <= 0:
                raise ValueError(f"{name} must be a positive finite number.")

        self.hydrophobic_cutoff_angstrom = float(hydrophobic_cutoff_angstrom)
        self.hbond_cutoff_angstrom = float(hbond_cutoff_angstrom)
        self.aromatic_cutoff_angstrom = float(aromatic_cutoff_angstrom)
        self.salt_bridge_cutoff_angstrom = float(salt_bridge_cutoff_angstrom)

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        req = frozenset({"protein", "coordinates"})
        if self.ligand_selection is None:
            req = req | {"small_molecule_or_polymer"}
        missing = tuple(sorted(req - profile.capabilities))
        if missing:
            return FeatureAssessment(
                self.plugin_id,
                "protein_ligand_interactions",
                ApplicabilityStatus.NOT_APPLICABLE,
                "Protein–ligand interaction fingerprints require protein coordinates and a ligand component.",
                "frame",
                "indicator",
                missing_capabilities=missing,
            )
        return FeatureAssessment(
            self.plugin_id,
            "protein_ligand_interactions",
            ApplicabilityStatus.APPLICABLE,
            "Protein coordinates and ligand component are available.",
            "frame",
            "indicator",
        )

    def extract(
        self,
        topology: str | Path,
        trajectory: str | Path,
        *,
        max_frames: int | None = None,
        frame_stride: int = 1,
    ) -> PluginResult:
        if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")

        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)

        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))

        ligand_sel = self.ligand_selection if self.ligand_selection is not None else "not protein and not resname TIP3 HOH WAT NA CL"
        ligand = universe.select_atoms(ligand_sel)
        if len(ligand) == 0:
            unavailable = FeatureAssessment(
                self.plugin_id,
                "protein_ligand_interactions",
                ApplicabilityStatus.NOT_APPLICABLE,
                f"No ligand atoms found for selection {ligand_sel!r}.",
                "frame",
                "indicator",
            )
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)

        protein = universe.select_atoms(self.protein_selection)
        protein_residues = list(protein.residues)
        if len(protein_residues) == 0:
            unavailable = FeatureAssessment(
                self.plugin_id,
                "protein_ligand_interactions",
                ApplicabilityStatus.NOT_APPLICABLE,
                f"No protein residues found for selection {self.protein_selection!r}.",
                "frame",
                "indicator",
            )
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)

        # Build stable, unique residue identifiers
        base_names = [f"{_safe_id(r.resname.strip())}_{r.resid}" for r in protein_residues]
        base_counts = Counter(base_names)
        has_duplicates = any(c > 1 for c in base_counts.values())

        res_identifiers: dict[int, str] = {}
        for r in protein_residues:
            base = f"{_safe_id(r.resname.strip())}_{r.resid}"
            if has_duplicates:
                scope = _chain_or_segment_id(r.atoms)
                res_identifiers[r.ix] = f"{_safe_id(scope)}__{base}"
            else:
                res_identifiers[r.ix] = base

        # Pre-select protein atoms for interactions
        protein_donors = protein.select_atoms(PROTEIN_DONOR_SELECTION.removeprefix("protein and "))
        protein_acceptors = protein.select_atoms(PROTEIN_ACCEPTOR_SELECTION.removeprefix("protein and "))
        protein_acidic = protein.select_atoms(PROTEIN_ACIDIC_SELECTION.removeprefix("protein and "))
        protein_basic = protein.select_atoms(PROTEIN_BASIC_SELECTION.removeprefix("protein and "))

        # Pre-detect protein aromatic rings
        protein_rings: list[tuple[int, mda.core.groups.AtomGroup]] = []
        for r in protein_residues:
            resname = r.resname.strip().upper()
            if resname in {"PHE", "TYR"}:
                atoms = r.atoms.select_atoms("name CG CD1 CD2 CE1 CE2 CZ")
                if {a.name.strip().upper() for a in atoms} == PHE_TYR_RING_NAMES:
                    protein_rings.append((r.ix, atoms))
            elif resname in HIS_RESNAMES:
                atoms = r.atoms.select_atoms("name CG ND1 CD2 CE1 NE2")
                if {a.name.strip().upper() for a in atoms} == HIS_RING_NAMES:
                    protein_rings.append((r.ix, atoms))
            elif resname == "TRP":
                five = r.atoms.select_atoms("name CG CD1 NE1 CE2 CD2")
                if {a.name.strip().upper() for a in five} == TRP_5RING_NAMES:
                    protein_rings.append((r.ix, five))
                six = r.atoms.select_atoms("name CD2 CE2 CZ2 CH2 CZ3 CE3")
                if {a.name.strip().upper() for a in six} == TRP_6RING_NAMES:
                    protein_rings.append((r.ix, six))

        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]

        frame_records = []
        residue_contact_history: dict[int, list[int]] = {r.ix: [] for r in protein_residues}
        residue_hydrophobic_history: dict[int, list[int]] = {r.ix: [] for r in protein_residues}
        residue_hbond_history: dict[int, list[int]] = {r.ix: [] for r in protein_residues}
        residue_aromatic_history: dict[int, list[int]] = {r.ix: [] for r in protein_residues}
        residue_salt_bridge_history: dict[int, list[int]] = {r.ix: [] for r in protein_residues}

        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]

            res_has_hydrophobic: dict[int, int] = {r.ix: 0 for r in protein_residues}
            res_has_hbond: dict[int, int] = {r.ix: 0 for r in protein_residues}
            res_has_aromatic: dict[int, int] = {r.ix: 0 for r in protein_residues}
            res_has_salt_bridge: dict[int, int] = {r.ix: 0 for r in protein_residues}
            res_has_contact: dict[int, int] = {r.ix: 0 for r in protein_residues}

            hydrophobic_distances: list[float] = []
            hbond_distances: list[float] = []
            aromatic_distances: list[float] = []
            salt_bridge_distances: list[float] = []

            # 1. Hydrophobic Contacts
            protein_nonpolar = _find_nonpolar_carbons(protein, protein)
            ligand_nonpolar = _find_nonpolar_carbons(ligand, ligand)
            if len(protein_nonpolar) > 0 and len(ligand_nonpolar) > 0:
                h_pairs, h_dists = capped_distance(
                    protein_nonpolar.positions,
                    ligand_nonpolar.positions,
                    max_cutoff=self.hydrophobic_cutoff_angstrom,
                    return_distances=True,
                )
                for (p_idx, _), dist in zip(h_pairs, h_dists, strict=False):
                    prot_atom = protein_nonpolar[p_idx]
                    res_ix = prot_atom.residue.ix
                    res_has_hydrophobic[res_ix] = 1
                    hydrophobic_distances.append(float(dist))

            # 2. Hydrogen Bonds
            ligand_donors, ligand_acceptors = _find_ligand_donors_and_acceptors(ligand)
            # Protein donor -> Ligand acceptor
            if len(protein_donors) > 0 and len(ligand_acceptors) > 0:
                hb_pairs1, hb_dists1 = capped_distance(
                    protein_donors.positions,
                    ligand_acceptors.positions,
                    max_cutoff=self.hbond_cutoff_angstrom,
                    return_distances=True,
                )
                for (p_idx, _), dist in zip(hb_pairs1, hb_dists1, strict=False):
                    prot_atom = protein_donors[p_idx]
                    res_ix = prot_atom.residue.ix
                    res_has_hbond[res_ix] = 1
                    hbond_distances.append(float(dist))

            # Ligand donor -> Protein acceptor
            if len(ligand_donors) > 0 and len(protein_acceptors) > 0:
                hb_pairs2, hb_dists2 = capped_distance(
                    protein_acceptors.positions,
                    ligand_donors.positions,
                    max_cutoff=self.hbond_cutoff_angstrom,
                    return_distances=True,
                )
                for (p_idx, _), dist in zip(hb_pairs2, hb_dists2, strict=False):
                    prot_atom = protein_acceptors[p_idx]
                    res_ix = prot_atom.residue.ix
                    res_has_hbond[res_ix] = 1
                    hbond_distances.append(float(dist))

            # 3. Aromatic Stacking
            ligand_rings = _find_ligand_aromatic_rings(ligand)
            if len(protein_rings) > 0 and len(ligand_rings) > 0:
                ligand_ring_centers = [r.positions.mean(axis=0) for r in ligand_rings]
                for res_ix, p_ring in protein_rings:
                    p_center = p_ring.positions.mean(axis=0)
                    for l_center in ligand_ring_centers:
                        center_dist = float(np.linalg.norm(p_center - l_center))
                        if center_dist <= self.aromatic_cutoff_angstrom:
                            res_has_aromatic[res_ix] = 1
                            aromatic_distances.append(center_dist)

            # 4. Salt Bridges
            ligand_pos, ligand_neg = _find_ligand_charged_atoms(ligand)
            # Protein acidic (-) with Ligand positive (+)
            if len(protein_acidic) > 0 and len(ligand_pos) > 0:
                sb_pairs1, sb_dists1 = capped_distance(
                    protein_acidic.positions,
                    ligand_pos.positions,
                    max_cutoff=self.salt_bridge_cutoff_angstrom,
                    return_distances=True,
                )
                for (p_idx, _), dist in zip(sb_pairs1, sb_dists1, strict=False):
                    prot_atom = protein_acidic[p_idx]
                    res_ix = prot_atom.residue.ix
                    res_has_salt_bridge[res_ix] = 1
                    salt_bridge_distances.append(float(dist))

            # Protein basic (+) with Ligand negative (-)
            if len(protein_basic) > 0 and len(ligand_neg) > 0:
                sb_pairs2, sb_dists2 = capped_distance(
                    protein_basic.positions,
                    ligand_neg.positions,
                    max_cutoff=self.salt_bridge_cutoff_angstrom,
                    return_distances=True,
                )
                for (p_idx, _), dist in zip(sb_pairs2, sb_dists2, strict=False):
                    prot_atom = protein_basic[p_idx]
                    res_ix = prot_atom.residue.ix
                    res_has_salt_bridge[res_ix] = 1
                    salt_bridge_distances.append(float(dist))

            # 5. Combined contacts & direct geometric proximity
            contact_pairs, _ = capped_distance(
                protein.positions,
                ligand.positions,
                max_cutoff=self.hydrophobic_cutoff_angstrom,
            )
            for p_idx, _ in contact_pairs:
                res_ix = protein[p_idx].residue.ix
                res_has_contact[res_ix] = 1

            for r in protein_residues:
                if (
                    res_has_hydrophobic[r.ix]
                    or res_has_hbond[r.ix]
                    or res_has_aromatic[r.ix]
                    or res_has_salt_bridge[r.ix]
                ):
                    res_has_contact[r.ix] = 1

            all_distances = hydrophobic_distances + hbond_distances + aromatic_distances + salt_bridge_distances
            total_count = (
                len(hydrophobic_distances)
                + len(hbond_distances)
                + len(aromatic_distances)
                + len(salt_bridge_distances)
            )

            row: dict[str, Any] = {
                "frame_index": frame_index,
                "frame_order": output_order,
                "time_ps": float(universe.trajectory.ts.time),
                "PLIF_hydrophobic_count": len(hydrophobic_distances),
                "PLIF_hydrophobic_min_distance": float(np.min(hydrophobic_distances)) if len(hydrophobic_distances) else np.nan,
                "PLIF_hbond_count": len(hbond_distances),
                "PLIF_hbond_min_distance": float(np.min(hbond_distances)) if len(hbond_distances) else np.nan,
                "PLIF_aromatic_count": len(aromatic_distances),
                "PLIF_aromatic_min_distance": float(np.min(aromatic_distances)) if len(aromatic_distances) else np.nan,
                "PLIF_salt_bridge_count": len(salt_bridge_distances),
                "PLIF_salt_bridge_min_distance": float(np.min(salt_bridge_distances)) if len(salt_bridge_distances) else np.nan,
                "PLIF_total_count": total_count,
                "PLIF_min_distance": float(np.min(all_distances)) if len(all_distances) else np.nan,
            }

            for r in protein_residues:
                ident = res_identifiers[r.ix]
                c_val = res_has_contact[r.ix]
                h_val = res_has_hydrophobic[r.ix]
                hb_val = res_has_hbond[r.ix]
                ar_val = res_has_aromatic[r.ix]
                sb_val = res_has_salt_bridge[r.ix]

                row[f"PLIF_contact__{ident}"] = c_val
                row[f"PLIF_hydrophobic__{ident}"] = h_val
                row[f"PLIF_hbond__{ident}"] = hb_val
                row[f"PLIF_aromatic__{ident}"] = ar_val
                row[f"PLIF_salt_bridge__{ident}"] = sb_val

                residue_contact_history[r.ix].append(c_val)
                residue_hydrophobic_history[r.ix].append(h_val)
                residue_hbond_history[r.ix].append(hb_val)
                residue_aromatic_history[r.ix].append(ar_val)
                residue_salt_bridge_history[r.ix].append(sb_val)

            frame_records.append(row)

        frame_df = pd.DataFrame.from_records(frame_records)

        # Build trajectory summary
        traj_data: dict[str, Any] = {
            "PLIF_any_contact_occupancy": float(np.mean([r["PLIF_total_count"] > 0 for r in frame_records])),
            "PLIF_hydrophobic_occupancy": float(np.mean([r["PLIF_hydrophobic_count"] > 0 for r in frame_records])),
            "PLIF_hbond_occupancy": float(np.mean([r["PLIF_hbond_count"] > 0 for r in frame_records])),
            "PLIF_aromatic_occupancy": float(np.mean([r["PLIF_aromatic_count"] > 0 for r in frame_records])),
            "PLIF_salt_bridge_occupancy": float(np.mean([r["PLIF_salt_bridge_count"] > 0 for r in frame_records])),
        }

        for r in protein_residues:
            ident = res_identifiers[r.ix]
            traj_data[f"PLIF_contact_occupancy__{ident}"] = float(np.mean(residue_contact_history[r.ix]))
            traj_data[f"PLIF_hydrophobic_occupancy__{ident}"] = float(np.mean(residue_hydrophobic_history[r.ix]))
            traj_data[f"PLIF_hbond_occupancy__{ident}"] = float(np.mean(residue_hbond_history[r.ix]))
            traj_data[f"PLIF_aromatic_occupancy__{ident}"] = float(np.mean(residue_aromatic_history[r.ix]))
            traj_data[f"PLIF_salt_bridge_occupancy__{ident}"] = float(np.mean(residue_salt_bridge_history[r.ix]))

        traj_df = pd.DataFrame([traj_data])

        provenance = FeatureProvenance(
            self.plugin_id,
            self.version,
            {
                "ligand_selection": self.ligand_selection,
                "protein_selection": self.protein_selection,
                "hydrophobic_cutoff_angstrom": self.hydrophobic_cutoff_angstrom,
                "hbond_cutoff_angstrom": self.hbond_cutoff_angstrom,
                "aromatic_cutoff_angstrom": self.aromatic_cutoff_angstrom,
                "salt_bridge_cutoff_angstrom": self.salt_bridge_cutoff_angstrom,
                "pbc": False,
                "max_frames": max_frames,
                "frame_stride": frame_stride,
            },
            str(topology),
            str(trajectory),
            perf_counter() - start,
        )

        return PluginResult(frame_df, traj_df, assessment, provenance)
