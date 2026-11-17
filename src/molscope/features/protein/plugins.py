"""Optional, capability-gated domain feature plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Protocol

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.lib.distances import distance_array, calc_bonds, calc_dihedrals
import networkx as nx
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.profiling.system import SystemProfile, discover_component_selections, profile_system

from molscope.features.base import PluginResult, FeaturePlugin, _safe_id, _chain_or_segment_id


class ProteinBackboneTorsionPlugin:
    """Protein-only phi, psi, and omega torsions with explicit applicability."""

    plugin_id = "protein.backbone_torsions"
    version = "2"
    required_capabilities = frozenset({"protein_backbone", "bonds"})

    def __init__(self, selection: str = "protein", torsions: tuple[str, ...] = ("phi", "psi", "omega")):
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("selection must be a non-empty MDAnalysis selection.")
        if not isinstance(torsions, (list, tuple)) or not torsions or any(name not in {"phi", "psi", "omega"} for name in torsions) or len(set(torsions)) != len(torsions):
            raise ValueError("torsions must be a non-empty unique list from phi, psi, omega.")
        self.selection = selection
        self.torsions = tuple(torsions)

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(
                feature_id=self.plugin_id,
                family="protein_chemistry",
                status=ApplicabilityStatus.NOT_APPLICABLE,
                reason="Protein backbone torsions require standard protein backbone and bond topology.",
                granularity="frame",
                units="degrees",
                missing_capabilities=missing,
            )
        return FeatureAssessment(
            feature_id=self.plugin_id,
            family="protein_chemistry",
            status=ApplicabilityStatus.APPLICABLE,
            reason="Protein backbone and bond topology are available.",
            granularity="frame",
            units="degrees",
        )

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        if max_frames is not None and max_frames < 1:
            raise ValueError("max_frames must be positive when supplied.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)

        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        universe.trajectory[0]
        torsions = []
        for residue in universe.select_atoms(f"protein and ({self.selection})").residues:
            scope = _chain_or_segment_id(residue.atoms)
            residue_id = f"{_safe_id(scope)}__{residue.resname.strip()}_{residue.resid}"
            for torsion_name, selection in (("phi", residue.phi_selection()), ("psi", residue.psi_selection()), ("omega", residue.omega_selection())):
                if torsion_name in self.torsions and selection is not None and len(selection) == 4:
                    torsions.append((f"TORSION__{residue_id}__{torsion_name}", selection))

        if not torsions:
            unavailable = FeatureAssessment(self.plugin_id, "protein_chemistry", ApplicabilityStatus.NOT_APPLICABLE,
                                            "No complete requested torsions exist in the selected residues.", "frame", "unitless")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)

        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records = []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time)}
            for feature_id, selection in torsions:
                radians = np.radians(selection.dihedral.value())
                row[f"{feature_id}__sin"] = float(np.sin(radians))
                row[f"{feature_id}__cos"] = float(np.cos(radians))
            records.append(row)
        provenance = FeatureProvenance(
            analyzer_id=self.plugin_id,
            analyzer_version=self.version,
            parameters={"max_frames": max_frames, "frame_stride": frame_stride, "torsion_encoding": "sin_cos", "selection": self.selection, "torsions": self.torsions},
            source_topology=str(topology),
            source_trajectory=str(trajectory),
            elapsed_seconds=perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), pd.DataFrame(), assessment, provenance)


class ProteinCADistanceContactPlugin:
    """Bounded Cα pair distances and a conventional residue contact map."""

    plugin_id = "protein.ca_distances_contacts"
    version = "3"
    contact_cutoff_angstrom = 8.0
    minimum_sequence_separation = 3
    required_capabilities = frozenset({"protein", "coordinates"})

    def __init__(self, selection: str = "protein and name CA", contact_cutoff_angstrom: float = 8.0,
                 minimum_sequence_separation: int = 3, max_pairs: int = 5000,
                 residue_pairs: list[list[int]] | None = None, include_contacts: bool = True,
                 use_pbc: bool = False):
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("selection must be a non-empty MDAnalysis selection.")
        if isinstance(contact_cutoff_angstrom, bool) or not isinstance(contact_cutoff_angstrom, (int, float)) or not np.isfinite(contact_cutoff_angstrom) or contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be finite and positive.")
        for name, value, minimum in (("minimum_sequence_separation", minimum_sequence_separation, 0), ("max_pairs", max_pairs, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if not isinstance(include_contacts, bool):
            raise ValueError("include_contacts must be boolean.")
        if not isinstance(use_pbc, bool):
            raise ValueError("use_pbc must be boolean.")
        if residue_pairs is not None:
            if not isinstance(residue_pairs, (list, tuple)) or not residue_pairs:
                raise ValueError("residue_pairs must be a non-empty list of two-residue pairs.")
            normalized = []
            for pair in residue_pairs:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2 or any(isinstance(value, bool) or not isinstance(value, int) for value in pair) or pair[0] == pair[1]:
                    raise ValueError("Each residue pair must contain two distinct integer residue IDs.")
                normalized.append(tuple(sorted(pair)))
            if len(set(normalized)) != len(normalized):
                raise ValueError("residue_pairs must be unique.")
            if len(normalized) > max_pairs:
                raise ValueError("residue_pairs exceeds max_pairs.")
            residue_pairs = normalized
        self.selection = selection
        self.contact_cutoff_angstrom = float(contact_cutoff_angstrom)
        self.minimum_sequence_separation = minimum_sequence_separation
        self.max_pairs = max_pairs
        self.residue_pairs = residue_pairs
        self.include_contacts = include_contacts
        self.use_pbc = use_pbc

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE, "Cα distances require protein coordinates.", "frame", "angstrom", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.APPLICABLE, "Protein coordinates are available; Cα availability is checked during extraction.", "frame", "angstrom")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        if self.use_pbc and not profile.has_periodic_cell:
            raise ValueError("PBC-aware Cα distances require positive periodic-cell metadata in the trajectory.")
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")
        alpha_carbons = universe.select_atoms(f"protein and name CA and ({self.selection})")
        if len(alpha_carbons) < 2:
            unavailable = FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE, "Fewer than two protein Cα atoms were found.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if self.residue_pairs is None:
            candidate_indices = combinations(range(len(alpha_carbons)), 2)
        else:
            by_resid = {}
            for index, atom in enumerate(alpha_carbons):
                by_resid.setdefault(int(atom.resid), []).append(index)
            missing = [pair for pair in self.residue_pairs if any(len(by_resid.get(resid, ())) != 1 for resid in pair)]
            if missing:
                raise ValueError(f"Selected residue pairs require one Cα per residue ID; invalid pairs: {missing[:5]}")
            candidate_indices = ((by_resid[left][0], by_resid[right][0]) for left, right in self.residue_pairs)
        pairs = []
        for left_index, right_index in candidate_indices:
            left, right = alpha_carbons[left_index], alpha_carbons[right_index]
            left_scope = _chain_or_segment_id(left.residue.atoms)
            right_scope = _chain_or_segment_id(right.residue.atoms)
            same_chain = left_scope == right_scope
            if same_chain and abs(left.resid - right.resid) < self.minimum_sequence_separation:
                if self.residue_pairs is not None:
                    raise ValueError(f"Declared pair {left.resid}, {right.resid} violates minimum_sequence_separation.")
                continue
            left_chain = left_scope.removeprefix("chain_").removeprefix("segment_")
            right_chain = right_scope.removeprefix("chain_").removeprefix("segment_")
            left_id = _safe_id(f"{left_chain}__{left.resname}_{left.resid}")
            right_id = _safe_id(f"{right_chain}__{right.resname}_{right.resid}")
            pairs.append((left_index, right_index, f"{left_id}__{right_id}"))
            if len(pairs) > self.max_pairs:
                raise ValueError(f"Cα pair count exceeds max_pairs={self.max_pairs}; narrow selection or explicitly raise the limit.")
        if not pairs:
            unavailable = FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE, "No Cα pairs remain after the declared sequence-separation policy.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        if len({pair_id for _, _, pair_id in pairs}) != len(pairs):
            raise ValueError("Ambiguous Cα residue identities; resolve duplicate chain/residue labels before extraction.")
        contact_presence = {pair_id: [] for _, _, pair_id in pairs} if self.include_contacts else {}
        left_indices, right_indices = np.asarray([(left, right) for left, right, _ in pairs]).T
        records = []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            distances = calc_bonds(
                alpha_carbons.positions[left_indices], alpha_carbons.positions[right_indices],
                box=universe.dimensions if self.use_pbc else None,
            )
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time)}
            for pair_index, (_, _, pair_id) in enumerate(pairs):
                distance = float(distances[pair_index])
                row[f"CA_DISTANCE_angstrom__{pair_id}"] = distance
                if self.include_contacts:
                    contact = distance <= self.contact_cutoff_angstrom
                    row[f"CA_CONTACT__{pair_id}"] = float(contact)
                    contact_presence[pair_id].append(contact)
            records.append(row)
        trajectory_features = pd.DataFrame([{
            f"CA_CONTACT_occupancy__{pair_id}": float(np.mean(values))
            for pair_id, values in contact_presence.items()
        }]) if self.include_contacts else pd.DataFrame()
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"contact_cutoff_angstrom": self.contact_cutoff_angstrom, "minimum_sequence_separation": self.minimum_sequence_separation, "atom_selection": self.selection, "max_pairs": self.max_pairs, "residue_pairs": self.residue_pairs, "include_contacts": self.include_contacts, "pbc": self.use_pbc, "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinResidueHeavyDistancePlugin:
    """Bounded closest-heavy-atom residue distances, matching a common MD feature convention."""

    plugin_id = "protein.residue_heavy_distances_contacts"
    version = "1"
    required_capabilities = frozenset({"protein", "coordinates"})

    def __init__(self, selection: str = "protein", contact_cutoff_angstrom: float = 4.5,
                 minimum_sequence_separation: int = 1, max_pairs: int = 5_000,
                 include_contacts: bool = True, use_pbc: bool = False):
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("selection must be a non-empty MDAnalysis selection.")
        if isinstance(contact_cutoff_angstrom, bool) or not isinstance(contact_cutoff_angstrom, (int, float)) or not np.isfinite(contact_cutoff_angstrom) or contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be finite and positive.")
        if isinstance(minimum_sequence_separation, bool) or not isinstance(minimum_sequence_separation, int) or minimum_sequence_separation < 0:
            raise ValueError("minimum_sequence_separation must be an integer >= 0.")
        if isinstance(max_pairs, bool) or not isinstance(max_pairs, int) or max_pairs < 1:
            raise ValueError("max_pairs must be positive.")
        if not isinstance(include_contacts, bool) or not isinstance(use_pbc, bool):
            raise ValueError("include_contacts and use_pbc must be boolean.")
        self.selection = selection
        self.contact_cutoff_angstrom = float(contact_cutoff_angstrom)
        self.minimum_sequence_separation = minimum_sequence_separation
        self.max_pairs = max_pairs
        self.include_contacts = include_contacts
        self.use_pbc = use_pbc

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE,
                                     "Closest-heavy-atom residue distances require protein coordinates.", "frame", "angstrom",
                                     missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.APPLICABLE,
                                 "Protein coordinates are available; complete heavy-atom residues are checked during extraction.",
                                 "frame", "angstrom")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None,
                frame_stride: int = 1) -> PluginResult:
        if isinstance(frame_stride, bool) or not isinstance(frame_stride, int) or frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        if self.use_pbc and not profile.has_periodic_cell:
            raise ValueError("PBC-aware residue distances require positive periodic-cell metadata in the trajectory.")
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        selected_residues = list(universe.select_atoms(f"protein and ({self.selection})").residues)
        residues = []
        for residue in selected_residues:
            heavy = residue.atoms.select_atoms("not name H*")
            if len(heavy):
                residues.append((residue, np.asarray(heavy.indices, dtype=int)))
        pairs = []
        for left_index, right_index in combinations(range(len(residues)), 2):
            left, left_atoms = residues[left_index]
            right, right_atoms = residues[right_index]
            if _chain_or_segment_id(left.atoms) == _chain_or_segment_id(right.atoms) and abs(int(left.resid) - int(right.resid)) < self.minimum_sequence_separation:
                continue
            left_id = _safe_id(f"{_chain_or_segment_id(left.atoms)}__{left.resname}_{left.resid}")
            right_id = _safe_id(f"{_chain_or_segment_id(right.atoms)}__{right.resname}_{right.resid}")
            pairs.append((left_atoms, right_atoms, f"{left_id}__{right_id}"))
            if len(pairs) > self.max_pairs:
                raise ValueError(f"Residue heavy-atom pair count exceeds max_pairs={self.max_pairs}; narrow selection or raise the limit.")
        if not pairs:
            unavailable = FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE,
                                            "No complete residue-heavy-atom pairs remain after the declared selection policy.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if len({pair_id for _, _, pair_id in pairs}) != len(pairs):
            raise ValueError("Ambiguous residue identities; resolve duplicate chain/residue labels before extraction.")
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        contact_presence = {pair_id: [] for _, _, pair_id in pairs} if self.include_contacts else {}
        records = []
        for order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            box = universe.dimensions if self.use_pbc else None
            row = {"frame_index": frame_index, "frame_order": order, "time_ps": float(universe.trajectory.ts.time)}
            for left_atoms, right_atoms, pair_id in pairs:
                distance = float(np.min(distance_array(universe.atoms.positions[left_atoms], universe.atoms.positions[right_atoms], box=box)))
                row[f"RESIDUE_HEAVY_DISTANCE_angstrom__{pair_id}"] = distance
                if self.include_contacts:
                    contact = distance <= self.contact_cutoff_angstrom
                    row[f"RESIDUE_HEAVY_CONTACT__{pair_id}"] = float(contact)
                    contact_presence[pair_id].append(contact)
            records.append(row)
        trajectory_features = pd.DataFrame([{
            f"RESIDUE_HEAVY_CONTACT_occupancy__{pair_id}": float(np.mean(values))
            for pair_id, values in contact_presence.items()
        }]) if self.include_contacts else pd.DataFrame()
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"atom_selection": self.selection, "distance_convention": "closest_heavy_atom", "contact_cutoff_angstrom": self.contact_cutoff_angstrom,
             "minimum_sequence_separation": self.minimum_sequence_separation, "max_pairs": self.max_pairs,
             "include_contacts": self.include_contacts, "pbc": self.use_pbc, "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinSidechainTorsionPlugin:
    """Selected protein side-chain chi angles from atom geometry, without bonds."""

    plugin_id = "protein.sidechain_torsions"
    version = "2"
    required_capabilities = frozenset({"protein", "coordinates"})
    chi1_end = {
        "ARG": "CG", "ASN": "CG", "ASP": "CG", "GLN": "CG", "GLU": "CG",
        "HIS": "CG", "HSD": "CG", "HSE": "CG", "ILE": "CG1", "LEU": "CG",
        "LYS": "CG", "MET": "CG", "PHE": "CG", "PRO": "CG", "SER": "OG",
        "THR": "OG1", "TRP": "CG", "TYR": "CG", "VAL": "CG1",
    }
    chi2_end = {
        "ARG": "CD", "ASN": "OD1", "ASP": "OD1", "GLN": "CD", "GLU": "CD",
        "HIS": "ND1", "HSD": "ND1", "HSE": "ND1", "ILE": "CD1", "LEU": "CD1",
        "LYS": "CD", "MET": "SD", "PHE": "CD1", "PRO": "CD", "TRP": "CD1",
        "TYR": "CD1",
    }
    chi3_atoms = {
        "ARG": ("CB", "CG", "CD", "NE"), "GLN": ("CB", "CG", "CD", "OE1"),
        "GLU": ("CB", "CG", "CD", "OE1"), "LYS": ("CB", "CG", "CD", "CE"),
        "MET": ("CB", "CG", "SD", "CE"),
    }
    chi4_atoms = {
        "ARG": ("CG", "CD", "NE", "CZ"), "LYS": ("CG", "CD", "CE", "NZ"),
    }

    def __init__(self, residue_ids: list[int] | None = None, include_chi2: bool = True,
                 include_chi3: bool = True, include_chi4: bool = True, max_torsions: int = 100):
        if residue_ids is not None and (not isinstance(residue_ids, (list, tuple)) or not residue_ids
                                        or any(isinstance(value, bool) or not isinstance(value, int) for value in residue_ids)
                                        or len(set(residue_ids)) != len(residue_ids)):
            raise ValueError("residue_ids must be a non-empty unique list of integers.")
        if not isinstance(include_chi2, bool):
            raise ValueError("include_chi2 must be boolean.")
        if not isinstance(include_chi3, bool) or not isinstance(include_chi4, bool):
            raise ValueError("include_chi3 and include_chi4 must be boolean.")
        if isinstance(max_torsions, bool) or not isinstance(max_torsions, int) or max_torsions < 1:
            raise ValueError("max_torsions must be positive.")
        self.residue_ids = tuple(residue_ids) if residue_ids is not None else None
        self.include_chi2 = include_chi2
        self.include_chi3 = include_chi3
        self.include_chi4 = include_chi4
        self.max_torsions = max_torsions

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_chemistry", ApplicabilityStatus.NOT_APPLICABLE,
                                     "Side-chain torsions need protein coordinates.", "frame", "unitless",
                                     missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_chemistry", ApplicabilityStatus.APPLICABLE,
                                 "Side-chain torsions use named atoms; availability is checked during extraction.",
                                 "frame", "unitless")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None,
                frame_stride: int = 1) -> PluginResult:
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
        residues = list(universe.select_atoms("protein").residues)
        if self.residue_ids is not None:
            selected = [residue for resid in self.residue_ids for residue in residues if int(residue.resid) == resid]
            if len(selected) != len(self.residue_ids) or len({int(residue.resid) for residue in selected}) != len(selected):
                raise ValueError("Each selected residue ID must identify exactly one protein residue.")
            residues = selected
        torsions = []
        for residue in residues:
            atom_indices = {}
            for atom in residue.atoms:
                atom_indices.setdefault(atom.name.strip(), []).append(int(atom.ix))
            def indices(names):
                if any(len(atom_indices.get(name, ())) != 1 for name in names):
                    return None
                return tuple(atom_indices[name][0] for name in names)
            name = residue.resname.strip()
            scope = _chain_or_segment_id(residue.atoms)
            identifier = f"{_safe_id(scope)}__{name}_{int(residue.resid)}"
            if name in self.chi1_end:
                group = indices(("N", "CA", "CB", self.chi1_end[name]))
                if group is not None:
                    torsions.append((f"SIDECHAIN_chi1__{identifier}", group))
            if self.include_chi2 and name in self.chi2_end:
                group = indices(("CA", "CB", "CG" if name != "ILE" else "CG1", self.chi2_end[name]))
                if group is not None:
                    torsions.append((f"SIDECHAIN_chi2__{identifier}", group))
            if self.include_chi3 and name in self.chi3_atoms:
                group = indices(self.chi3_atoms[name])
                if group is not None:
                    torsions.append((f"SIDECHAIN_chi3__{identifier}", group))
            if self.include_chi4 and name in self.chi4_atoms:
                group = indices(self.chi4_atoms[name])
                if group is not None:
                    torsions.append((f"SIDECHAIN_chi4__{identifier}", group))
        if not torsions:
            unavailable = FeatureAssessment(self.plugin_id, "protein_chemistry", ApplicabilityStatus.NOT_APPLICABLE,
                                            "No complete selected side-chain torsions were found.", "frame", "unitless")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if len(torsions) > self.max_torsions:
            raise ValueError(f"Side-chain torsion count exceeds max_torsions={self.max_torsions}.")
        if len({name for name, _ in torsions}) != len(torsions):
            raise ValueError("Ambiguous selected side-chain torsion identities.")
        indices_by_atom = np.array([item[1] for item in torsions], dtype=int)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records = []
        for order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            positions = universe.atoms.positions
            angles = calc_dihedrals(*(positions[indices_by_atom[:, column]] for column in range(4)))
            if not np.isfinite(angles).all():
                raise ValueError(f"Nonfinite side-chain torsion at frame {frame_index}.")
            row = {"frame_index": frame_index, "frame_order": order, "time_ps": float(universe.trajectory.ts.time)}
            for (feature_id, _), angle in zip(torsions, angles, strict=True):
                row[f"{feature_id}__sin"] = float(np.sin(angle))
                row[f"{feature_id}__cos"] = float(np.cos(angle))
            records.append(row)
        provenance = FeatureProvenance(self.plugin_id, self.version,
                                      {"residue_ids": self.residue_ids, "include_chi2": self.include_chi2,
                                       "include_chi3": self.include_chi3, "include_chi4": self.include_chi4,
                                       "max_torsions": self.max_torsions, "selected_atom_quartets":
                                       {name: indices for name, indices in torsions}, "pbc": False,
                                       "max_frames": max_frames, "frame_stride": frame_stride},
                                      str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), pd.DataFrame(), assessment, provenance)


class ProteinDSSPPlugin:
    """Protein secondary-structure fractions using MDAnalysis' native DSSP."""

    plugin_id = "protein.secondary_structure_dssp"
    version = "2"
    required_capabilities = frozenset({"protein", "coordinates"})

    def __init__(self, per_residue: bool = False):
        if not isinstance(per_residue, bool):
            raise ValueError("per_residue must be boolean.")
        self.per_residue = per_residue

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_structure", ApplicabilityStatus.NOT_APPLICABLE, "DSSP requires protein coordinates.", "frame", "fraction", missing_capabilities=missing)
        try:
            from MDAnalysis.analysis.dssp import DSSP  # noqa: F401
        except ImportError:
            return FeatureAssessment(self.plugin_id, "protein_structure", ApplicabilityStatus.UNSUPPORTED, "The MDAnalysis DSSP module is unavailable in this environment.", "frame", "fraction")
        return FeatureAssessment(self.plugin_id, "protein_structure", ApplicabilityStatus.APPLICABLE, "Protein coordinates and the MDAnalysis DSSP module are available; complete backbone availability is checked during extraction.", "frame", "fraction")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        from MDAnalysis.analysis.dssp import DSSP
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        protein = universe.select_atoms("protein")
        if not len(protein.residues):
            unavailable = FeatureAssessment(self.plugin_id, "protein_structure", ApplicabilityStatus.NOT_APPLICABLE, "No protein residues were found.", "frame", "fraction")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        try:
            result = DSSP(protein).run(frames=frame_indices)
        except (ValueError, KeyError, IndexError) as error:
            unavailable = FeatureAssessment(self.plugin_id, "protein_structure", ApplicabilityStatus.NOT_APPLICABLE, f"DSSP could not establish complete supported backbone geometry: {error}", "frame", "fraction")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        codes = np.asarray(result.results.dssp)
        if codes.shape != (len(frame_indices), len(protein.residues)) or not np.isin(codes, ["H", "E", "-"]).all():
            raise ValueError("DSSP returned unexpected frame/residue mapping or secondary-structure codes.")
        records = []
        for output_order, (frame_index, frame_codes) in enumerate(zip(frame_indices, codes, strict=True)):
            universe.trajectory[frame_index]
            records.append({
                "frame_index": frame_index,
                "frame_order": output_order,
                "time_ps": float(universe.trajectory.ts.time),
                "DSSP_helix_fraction": float(np.mean(frame_codes == "H")),
                "DSSP_sheet_fraction": float(np.mean(frame_codes == "E")),
                "DSSP_coil_fraction": float(np.mean(frame_codes == "-")),
            })
            if self.per_residue:
                for residue, code in zip(protein.residues, frame_codes, strict=True):
                    identifier = f"resindex_{residue.ix}"
                    for state, token in (("helix", "H"), ("sheet", "E"), ("coil", "-")):
                        records[-1][f"DSSP_{state}__{identifier}"] = float(code == token)
        frame_features = pd.DataFrame.from_records(records)
        trajectory_features = pd.DataFrame([{
            "DSSP_helix_fraction_mean": float(frame_features["DSSP_helix_fraction"].mean()),
            "DSSP_sheet_fraction_mean": float(frame_features["DSSP_sheet_fraction"].mean()),
            "DSSP_coil_fraction_mean": float(frame_features["DSSP_coil_fraction"].mean()),
        }])
        if self.per_residue:
            occupancies = {f"{column}__occupancy": float(frame_features[column].mean())
                           for column in frame_features if column.startswith("DSSP_") and "__resindex_" in column}
            trajectory_features = pd.concat([trajectory_features, pd.DataFrame([occupancies])], axis=1)
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"backend": "MDAnalysis.analysis.dssp.DSSP", "backend_version": mda.__version__, "per_residue": self.per_residue, "residue_mapping": [{"resindex": r.ix, "resid": int(r.resid), "resname": r.resname, "segid": r.segid} for r in protein.residues], "secondary_structure_codes": {"helix": "H", "sheet": "E", "coil": "-"}, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(frame_features, trajectory_features, assessment, provenance)


class ProteinSaltBridgePlugin:
    """Protein salt-bridge geometry using declared charged side-chain atoms."""

    plugin_id = "protein.salt_bridges"
    version = "1"
    cutoff_angstrom = 4.0
    required_capabilities = frozenset({"protein", "coordinates"})
    acidic_atoms = frozenset({"OD1", "OD2", "OE1", "OE2"})
    basic_atoms = frozenset({"NZ", "NH1", "NH2", "NE"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Salt-bridge geometry requires protein coordinates.", "frame", "angstrom", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.APPLICABLE, "Protein coordinates are available; charged-atom availability is checked during extraction.", "frame", "angstrom")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        acidic = universe.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
        basic = universe.select_atoms("resname LYS ARG and name NZ NH1 NH2 NE")
        if not len(acidic) or not len(basic):
            unavailable = FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "No standard acidic/basic side-chain atom sets were found.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records, bridge_presence = [], []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            pairs, distances = capped_distance(acidic.positions, basic.positions, max_cutoff=self.cutoff_angstrom, return_distances=True)
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "SALT_BRIDGE_pair_count": int(len(pairs)), "SALT_BRIDGE_min_distance": float(np.min(distances)) if len(distances) else np.nan}
            bridge_presence.append(bool(len(pairs)))
            records.append(row)
        trajectory_features = pd.DataFrame([{"SALT_BRIDGE_any_occupancy": float(np.mean(bridge_presence))}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"cutoff_angstrom": self.cutoff_angstrom, "acidic_atoms": sorted(self.acidic_atoms), "basic_atoms": sorted(self.basic_atoms), "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinHydrophobicContactPlugin:
    """Nonlocal contacts between declared hydrophobic protein side chains."""

    plugin_id = "protein.hydrophobic_contacts"
    version = "1"
    cutoff_angstrom = 4.5
    minimum_sequence_separation = 3
    hydrophobic_resnames = frozenset({"ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR", "PRO"})
    required_capabilities = frozenset({"protein", "coordinates"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Hydrophobic contacts require protein coordinates.", "frame", "count", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.APPLICABLE, "Protein coordinates are available; hydrophobic side-chain atom availability is checked during extraction.", "frame", "count")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        hydrophobic = universe.select_atoms("protein and resname ALA VAL ILE LEU MET PHE TRP TYR PRO and element C")
        if len(hydrophobic) < 2:
            unavailable = FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Fewer than two declared hydrophobic side-chain carbon atoms were found.", "frame", "count")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        contact_presence, records = [], []
        atom_residue_indices = np.asarray([atom.residue.ix for atom in hydrophobic], dtype=int)
        atom_resids = np.asarray([atom.residue.resid for atom in hydrophobic], dtype=int)
        atom_chains = np.asarray([atom.chainID.strip() for atom in hydrophobic], dtype=object)
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            pairs, distances = capped_distance(hydrophobic.positions, hydrophobic.positions, max_cutoff=self.cutoff_angstrom, return_distances=True)
            kept = []
            for pair_index, (left, right) in enumerate(pairs):
                if left >= right or atom_residue_indices[left] == atom_residue_indices[right]:
                    continue
                if atom_chains[left] == atom_chains[right] and abs(atom_resids[left] - atom_resids[right]) < self.minimum_sequence_separation:
                    continue
                kept.append(float(distances[pair_index]))
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "HYDROPHOBIC_contact_pair_count": len(kept), "HYDROPHOBIC_min_contact_distance": min(kept) if kept else np.nan}
            records.append(row)
            contact_presence.append(bool(kept))
        trajectory_features = pd.DataFrame([{"HYDROPHOBIC_any_occupancy": float(np.mean(contact_presence))}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"cutoff_angstrom": self.cutoff_angstrom, "minimum_sequence_separation": self.minimum_sequence_separation, "hydrophobic_resnames": sorted(self.hydrophobic_resnames), "carbon_only": True, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinHydrogenBondPlugin:
    """Protein hydrogen bonds with explicit donor-H bond and angle requirements."""

    plugin_id = "protein.hydrogen_bonds"
    version = "1"
    donor_acceptor_cutoff_angstrom = 3.5
    minimum_angle_degrees = 150.0
    required_capabilities = frozenset({"protein", "coordinates", "bonds", "hydrogens"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Hydrogen-bond geometry requires protein coordinates, donor-H bond topology, and explicit hydrogens.", "frame", "count", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.APPLICABLE, "Required protein, bond, and hydrogen information is available.", "frame", "count")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        donors = universe.select_atoms("protein and name N NE NH1 NH2 NZ ND1 NE2 OG OG1 OH")
        acceptors = universe.select_atoms("protein and name O OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH")
        donor_hydrogens = []
        for donor in donors:
            hydrogens = [atom for atom in donor.bonded_atoms if str(getattr(atom, "element", "")).upper() == "H" or atom.name.strip().upper().startswith("H")]
            donor_hydrogens.extend((donor, hydrogen) for hydrogen in hydrogens)
        if not donor_hydrogens or not len(acceptors):
            unavailable = FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "No supported donor-H bonds or acceptor atoms were found.", "frame", "count")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        counts, records = [], []
        cosine_limit = np.cos(np.radians(self.minimum_angle_degrees))
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            count, shortest = 0, np.inf
            for donor, hydrogen in donor_hydrogens:
                for acceptor in acceptors:
                    if acceptor.residue.ix == donor.residue.ix:
                        continue
                    donor_acceptor = np.linalg.norm(donor.position - acceptor.position)
                    if donor_acceptor > self.donor_acceptor_cutoff_angstrom:
                        continue
                    first = donor.position - hydrogen.position
                    second = acceptor.position - hydrogen.position
                    cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
                    if cosine <= cosine_limit:
                        count += 1
                        shortest = min(shortest, donor_acceptor)
            records.append({"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "HBOND_count": count, "HBOND_min_donor_acceptor_distance": shortest if np.isfinite(shortest) else np.nan})
            counts.append(bool(count))
        trajectory_features = pd.DataFrame([{"HBOND_any_occupancy": float(np.mean(counts))}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"donor_acceptor_cutoff_angstrom": self.donor_acceptor_cutoff_angstrom, "minimum_angle_degrees": self.minimum_angle_degrees, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinAromaticStackingPlugin:
    """PHE/TYR ring-plane stacking geometry with conservative criteria."""

    plugin_id = "protein.aromatic_stacking"
    version = "1"
    minimum_center_distance_angstrom = 3.5
    maximum_center_distance_angstrom = 7.0
    maximum_plane_angle_degrees = 30.0
    required_capabilities = frozenset({"protein", "coordinates"})
    ring_names = frozenset({"CG", "CD1", "CD2", "CE1", "CE2", "CZ"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Aromatic stacking geometry requires protein coordinates.", "frame", "count", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.APPLICABLE, "Protein coordinates are available; complete PHE/TYR rings are checked during extraction.", "frame", "count")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        rings = []
        for residue in universe.select_atoms("protein and resname PHE TYR").residues:
            atoms = residue.atoms.select_atoms("name CG CD1 CD2 CE1 CE2 CZ")
            if {atom.name.strip() for atom in atoms} == self.ring_names:
                rings.append((residue, atoms))
        if len(rings) < 2:
            unavailable = FeatureAssessment(self.plugin_id, "protein_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Fewer than two complete PHE/TYR aromatic rings were found.", "frame", "count")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        occupancy, records = [], []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            geometries = []
            for residue, atoms in rings:
                positions = atoms.positions
                center = positions.mean(axis=0)
                _, _, right = np.linalg.svd(positions - center)
                geometries.append((residue, center, right[-1]))
            count, closest = 0, np.inf
            for (left_residue, left_center, left_normal), (right_residue, right_center, right_normal) in combinations(geometries, 2):
                if left_residue.segid == right_residue.segid and abs(left_residue.resid - right_residue.resid) < 3:
                    continue
                distance = float(np.linalg.norm(left_center - right_center))
                angle = float(np.degrees(np.arccos(np.clip(abs(np.dot(left_normal, right_normal)), -1.0, 1.0))))
                if self.minimum_center_distance_angstrom <= distance <= self.maximum_center_distance_angstrom and angle <= self.maximum_plane_angle_degrees:
                    count += 1
                    closest = min(closest, distance)
            records.append({"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "AROMATIC_stack_count": count, "AROMATIC_min_center_distance": closest if np.isfinite(closest) else np.nan})
            occupancy.append(bool(count))
        trajectory_features = pd.DataFrame([{"AROMATIC_any_stack_occupancy": float(np.mean(occupancy))}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"minimum_center_distance_angstrom": self.minimum_center_distance_angstrom, "maximum_center_distance_angstrom": self.maximum_center_distance_angstrom, "maximum_plane_angle_degrees": self.maximum_plane_angle_degrees, "residue_types": ["PHE", "TYR"], "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class ProteinContactNetworkPlugin:
    """Dynamic residue-contact networks with explicit size safeguards."""

    plugin_id = "networks.protein_contact_network"
    version = "1"
    cutoff_angstrom = 4.5
    minimum_sequence_separation = 3
    max_exact_network_nodes = 400
    required_capabilities = frozenset({"protein", "coordinates"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "networks", ApplicabilityStatus.NOT_APPLICABLE, "Residue contact networks require protein coordinates.", "frame", "count", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "networks", ApplicabilityStatus.APPLICABLE, "Protein coordinates are available; exact network metrics may be skipped by the size policy.", "frame", "count")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        protein = universe.select_atoms("protein and not name H*")
        residues = list(protein.residues)
        if len(residues) < 2:
            unavailable = FeatureAssessment(self.plugin_id, "networks", ApplicabilityStatus.NOT_APPLICABLE, "Fewer than two protein residues with heavy atoms were found.", "frame", "count")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        atom_node = np.asarray([atom.residue.ix for atom in protein], dtype=int)
        node_resids = {residue.ix: residue.resid for residue in residues}
        node_chains = {residue.ix: (residue.atoms.chainIDs[0].strip() if len(residue.atoms.chainIDs) and residue.atoms.chainIDs[0].strip() else residue.segid.strip()) for residue in residues}
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records, edge_occurrences = [], {}
        exact = len(residues) <= self.max_exact_network_nodes
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            pairs, _ = capped_distance(protein.positions, protein.positions, max_cutoff=self.cutoff_angstrom, return_distances=True)
            edges = set()
            for left_atom, right_atom in pairs:
                left, right = atom_node[left_atom], atom_node[right_atom]
                if left >= right:
                    continue
                if node_chains[left] == node_chains[right] and abs(node_resids[left] - node_resids[right]) < self.minimum_sequence_separation:
                    continue
                edges.add((int(left), int(right)))
            graph = nx.Graph()
            graph.add_nodes_from(node_resids)
            graph.add_edges_from(edges)
            for edge in edges:
                edge_occurrences[edge] = edge_occurrences.get(edge, 0) + 1
            row = {
                "frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time),
                "NETWORK_node_count": graph.number_of_nodes(), "NETWORK_edge_count": graph.number_of_edges(),
                "NETWORK_density": float(nx.density(graph)),
                "NETWORK_mean_degree": float(np.mean([degree for _, degree in graph.degree()])),
            }
            if exact:
                betweenness = nx.betweenness_centrality(graph, normalized=True)
                row["NETWORK_mean_betweenness"] = float(np.mean(list(betweenness.values())))
                communities = list(nx.algorithms.community.greedy_modularity_communities(graph)) if graph.number_of_edges() else []
                row["NETWORK_community_count"] = len(communities)
                row["NETWORK_metric_status"] = "exact"
            else:
                row["NETWORK_metric_status"] = "skipped_by_cost_policy"
            records.append(row)
        persistence = np.asarray(list(edge_occurrences.values()), dtype=float) / len(frame_indices) if edge_occurrences else np.asarray([])
        trajectory_features = pd.DataFrame([{
            "NETWORK_unique_edge_count": len(edge_occurrences),
            "NETWORK_edge_persistence_mean": float(np.mean(persistence)) if len(persistence) else np.nan,
            "NETWORK_edge_persistence_median": float(np.median(persistence)) if len(persistence) else np.nan,
        }])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"cutoff_angstrom": self.cutoff_angstrom, "minimum_sequence_separation": self.minimum_sequence_separation, "max_exact_network_nodes": self.max_exact_network_nodes, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)
