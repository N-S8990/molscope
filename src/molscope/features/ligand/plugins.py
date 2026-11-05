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


class ProteinLigandContactPlugin:
    """Protein–single-ligand geometric contacts with explicit scope checks."""

    plugin_id = "protein_ligand.contacts"
    version = "1"
    contact_cutoff_angstrom = 4.5
    required_capabilities = frozenset({"protein", "small_molecule_or_polymer", "coordinates"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        ligand_like = [item for item in profile.components if item.kind == "small_molecule_or_polymer"]
        if missing or len(ligand_like) != 1:
            return FeatureAssessment(self.plugin_id, "protein_ligand_interactions", ApplicabilityStatus.NOT_APPLICABLE, "Protein–ligand contacts require protein coordinates and exactly one detected ligand-like component.", "frame", "fraction", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "protein_ligand_interactions", ApplicabilityStatus.APPLICABLE, "Protein coordinates and one ligand-like component are available.", "frame", "fraction")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        universe.trajectory[0]
        ligand_selection = next(item for item in discover_component_selections(universe) if item.profile.kind == "small_molecule_or_polymer")
        ligand = universe.atoms[list(ligand_selection.atom_indices)]
        protein_residues = list(universe.select_atoms("protein").residues)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records, contacted = [], {residue.ix: [] for residue in protein_residues}
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            protein = universe.select_atoms("protein")
            pairs, distances = capped_distance(protein.positions, ligand.positions, max_cutoff=self.contact_cutoff_angstrom, return_distances=True)
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "LIGAND_contact_pair_count": int(len(pairs)), "LIGAND_min_contact_distance": float(np.min(distances)) if len(distances) else np.nan}
            for residue in protein_residues:
                residue_pairs, _ = capped_distance(residue.atoms.positions, ligand.positions, max_cutoff=self.contact_cutoff_angstrom, return_distances=True)
                identifier = f"{_safe_id(residue.resname)}_{residue.resid}"
                row[f"LIGAND_residue_contact_fraction__{identifier}"] = float(len(residue_pairs) / (len(residue.atoms) * len(ligand)))
                contacted[residue.ix].append(bool(len(residue_pairs)))
            records.append(row)
        trajectory_row = {f"LIGAND_residue_occupancy__{_safe_id(residue.resname)}_{residue.resid}": float(np.mean(contacted[residue.ix])) for residue in protein_residues}
        provenance = FeatureProvenance(self.plugin_id, self.version, {"contact_cutoff_angstrom": self.contact_cutoff_angstrom, "max_frames": max_frames, "frame_stride": frame_stride, "pbc": False}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), pd.DataFrame([trajectory_row]), assessment, provenance)
