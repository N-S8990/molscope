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


class NucleicAcidBaseProximityPlugin:
    """Nucleic-acid base-center proximity descriptors without base-pair claims."""

    plugin_id = "nucleic_acid.base_proximities"
    version = "1"
    maximum_center_distance_angstrom = 7.0
    required_capabilities = frozenset({"nucleic_acid", "coordinates"})
    _base_atom_names = frozenset({"N1", "C2", "N3", "C4", "C5", "C6", "N7", "C8", "N9"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(
                self.plugin_id, "nucleic_acid", ApplicabilityStatus.NOT_APPLICABLE,
                "Base-center proximity requires nucleic-acid coordinates.", "frame", "count",
                missing_capabilities=missing,
            )
        return FeatureAssessment(
            self.plugin_id, "nucleic_acid", ApplicabilityStatus.APPLICABLE,
            "Nucleic-acid coordinates are available; identifiable base atoms are checked during extraction.",
            "frame", "count",
        )

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        base_groups = []
        for residue in universe.select_atoms("nucleic").residues:
            atoms = residue.atoms[[atom.name.strip().upper() in self._base_atom_names for atom in residue.atoms]]
            if len(atoms) >= 3:
                first_atom = residue.atoms[0]
                scope = getattr(first_atom, "chainID", "") or getattr(first_atom, "segid", "") or "unlabeled"
                identifier = _safe_id(f"{scope}__{residue.resname}_{residue.resid}")
                base_groups.append((identifier, atoms))
        if len(base_groups) < 2:
            unavailable = FeatureAssessment(
                self.plugin_id, "nucleic_acid", ApplicabilityStatus.NOT_APPLICABLE,
                "Fewer than two nucleic residues with at least three recognized base atoms were found.", "frame", "count",
            )
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        pair_presence = {f"{left[0]}__{right[0]}": [] for left, right in combinations(base_groups, 2)}
        records = []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            count, minimum = 0, np.inf
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time)}
            for left, right in combinations(base_groups, 2):
                pair_id = f"{left[0]}__{right[0]}"
                distance = float(np.linalg.norm(left[1].positions.mean(axis=0) - right[1].positions.mean(axis=0)))
                proximal = distance <= self.maximum_center_distance_angstrom
                row[f"NUCLEIC_base_center_distance__{pair_id}"] = distance
                pair_presence[pair_id].append(proximal)
                count += int(proximal)
                minimum = min(minimum, distance)
            row["NUCLEIC_base_proximal_pair_count"] = count
            row["NUCLEIC_min_base_center_distance"] = minimum
            records.append(row)
        trajectory_features = pd.DataFrame([{
            f"NUCLEIC_base_proximity_occupancy__{pair_id}": float(np.mean(values))
            for pair_id, values in pair_presence.items()
        }])
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"maximum_center_distance_angstrom": self.maximum_center_distance_angstrom, "recognized_base_atoms": sorted(self._base_atom_names), "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)
