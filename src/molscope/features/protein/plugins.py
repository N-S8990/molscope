"""Optional, capability-gated domain feature plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Protocol

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.lib.distances import distance_array, calc_bonds
import networkx as nx
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.profiling.system import SystemProfile, discover_component_selections, profile_system

from molscope.features.base import PluginResult, FeaturePlugin, _safe_id


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
            chain_ids = {value.strip() for value in residue.atoms.chainIDs if value and value.strip()}
            segment_ids = {value.strip() for value in residue.atoms.segids if value and value.strip()}
            scope = f"chain_{next(iter(chain_ids))}" if len(chain_ids) == 1 else f"segment_{next(iter(segment_ids))}" if len(segment_ids) == 1 else "unlabeled"
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
    """Non-PBC Cα pair distances and a conventional residue contact map."""

    plugin_id = "protein.ca_distances_contacts"
    version = "2"
    contact_cutoff_angstrom = 8.0
    minimum_sequence_separation = 3
    required_capabilities = frozenset({"protein", "coordinates"})

    def __init__(self, selection: str = "protein and name CA", contact_cutoff_angstrom: float = 8.0,
                 minimum_sequence_separation: int = 3, max_pairs: int = 5000):
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("selection must be a non-empty MDAnalysis selection.")
        if isinstance(contact_cutoff_angstrom, bool) or not isinstance(contact_cutoff_angstrom, (int, float)) or not np.isfinite(contact_cutoff_angstrom) or contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be finite and positive.")
        for name, value, minimum in (("minimum_sequence_separation", minimum_sequence_separation, 0), ("max_pairs", max_pairs, 1)):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        self.selection = selection
        self.contact_cutoff_angstrom = float(contact_cutoff_angstrom)
        self.minimum_sequence_separation = minimum_sequence_separation
        self.max_pairs = max_pairs

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
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        if max_frames is not None and (isinstance(max_frames, bool) or not isinstance(max_frames, int) or max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")
        alpha_carbons = universe.select_atoms(f"protein and name CA and ({self.selection})")
        if len(alpha_carbons) < 2:
            unavailable = FeatureAssessment(self.plugin_id, "protein_geometry", ApplicabilityStatus.NOT_APPLICABLE, "Fewer than two protein Cα atoms were found.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        pairs = []
        for left_index, right_index in combinations(range(len(alpha_carbons)), 2):
            left, right = alpha_carbons[left_index], alpha_carbons[right_index]
            same_chain = (left.chainID or left.segid) == (right.chainID or right.segid)
            if same_chain and abs(left.resid - right.resid) < self.minimum_sequence_separation:
                continue
            left_id = _safe_id(f"{left.chainID or left.segid or 'unlabeled'}__{left.resname}_{left.resid}")
            right_id = _safe_id(f"{right.chainID or right.segid or 'unlabeled'}__{right.resname}_{right.resid}")
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
        contact_presence = {pair_id: [] for _, _, pair_id in pairs}
        left_indices, right_indices = np.asarray([(left, right) for left, right, _ in pairs]).T
        records = []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            distances = calc_bonds(alpha_carbons.positions[left_indices], alpha_carbons.positions[right_indices])
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time)}
            for pair_index, (_, _, pair_id) in enumerate(pairs):
                distance = float(distances[pair_index])
                contact = distance <= self.contact_cutoff_angstrom
                row[f"CA_DISTANCE_angstrom__{pair_id}"] = distance
                row[f"CA_CONTACT__{pair_id}"] = float(contact)
                contact_presence[pair_id].append(contact)
            records.append(row)
        trajectory_features = pd.DataFrame([{
            f"CA_CONTACT_occupancy__{pair_id}": float(np.mean(values))
            for pair_id, values in contact_presence.items()
        }])
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"contact_cutoff_angstrom": self.contact_cutoff_angstrom, "minimum_sequence_separation": self.minimum_sequence_separation, "atom_selection": self.selection, "max_pairs": self.max_pairs, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


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
