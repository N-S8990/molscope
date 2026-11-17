"""Explicit selected Cartesian-coordinate features for trajectory analysis."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import MDAnalysis as mda
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.features.base import PluginResult, _chain_or_segment_id, _safe_id
from molscope.profiling.system import SystemProfile, profile_system


class SelectedCoordinatePlugin:
    """Export a bounded named atom selection as its stored Cartesian coordinates.

    Coordinates are intentionally not aligned, unwrapped, or PBC-transformed.
    Those operations change the observable and must be performed explicitly
    upstream with their own provenance.
    """

    plugin_id = "coordinates.selected_atoms"
    version = "1"
    required_capabilities = frozenset({"coordinates"})

    def __init__(self, selection: str = "protein and name CA", max_atoms: int = 500):
        if not isinstance(selection, str) or not selection.strip():
            raise ValueError("selection must be a non-empty MDAnalysis selection.")
        if isinstance(max_atoms, bool) or not isinstance(max_atoms, int) or max_atoms < 1:
            raise ValueError("max_atoms must be a positive integer.")
        self.selection = selection
        self.max_atoms = max_atoms

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "coordinates", ApplicabilityStatus.NOT_APPLICABLE,
                                     "Selected coordinate features require coordinates.", "frame", "angstrom",
                                     missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "coordinates", ApplicabilityStatus.APPLICABLE,
                                 "Coordinates are available; the requested atom selection is checked during extraction.",
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
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        atoms = universe.select_atoms(self.selection)
        if not len(atoms):
            unavailable = FeatureAssessment(self.plugin_id, "coordinates", ApplicabilityStatus.NOT_APPLICABLE,
                                            "The requested atom selection is empty.", "frame", "angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if len(atoms) > self.max_atoms:
            raise ValueError(f"Selected atom count exceeds max_atoms={self.max_atoms}; narrow selection or raise the limit.")
        labels = []
        for atom in atoms:
            scope = _chain_or_segment_id(atom.residue.atoms)
            labels.append(_safe_id(f"{scope}__{atom.resname}_{atom.resid}__{atom.name}"))
        if len(set(labels)) != len(labels):
            raise ValueError("Selected coordinate atoms have ambiguous chain/residue/atom identities.")
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records = []
        for order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            row = {"frame_index": frame_index, "frame_order": order, "time_ps": float(universe.trajectory.ts.time)}
            for label, position in zip(labels, atoms.positions, strict=True):
                row[f"COORD_x_angstrom__{label}"] = float(position[0])
                row[f"COORD_y_angstrom__{label}"] = float(position[1])
                row[f"COORD_z_angstrom__{label}"] = float(position[2])
            records.append(row)
        provenance = FeatureProvenance(
            self.plugin_id, self.version,
            {"atom_selection": self.selection, "max_atoms": self.max_atoms, "coordinate_policy": "as_stored_no_alignment_or_pbc_transform",
             "max_frames": max_frames, "frame_stride": frame_stride},
            str(topology), str(trajectory), perf_counter() - start,
        )
        return PluginResult(pd.DataFrame.from_records(records), pd.DataFrame(), assessment, provenance)
