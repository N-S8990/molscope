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


class SASAPlugin:
    """All-atom SASA using the optional, validated FreeSASA backend."""

    plugin_id = "structural.sasa"
    version = "1"
    probe_radius_angstrom = 1.4
    n_points = 100
    required_capabilities = frozenset({"coordinates"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "structural", ApplicabilityStatus.NOT_APPLICABLE, "SASA requires coordinates.", "frame", "angstrom_squared", missing_capabilities=missing)
        try:
            import freesasa  # noqa: F401
        except ImportError:
            return FeatureAssessment(self.plugin_id, "structural", ApplicabilityStatus.UNSUPPORTED, "FreeSASA is not installed; install the optional SASA dependency.", "frame", "angstrom_squared")
        return FeatureAssessment(self.plugin_id, "structural", ApplicabilityStatus.APPLICABLE, "Coordinates and the FreeSASA backend are available.", "frame", "angstrom_squared")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        import freesasa
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        universe.trajectory[0]
        selections = discover_component_selections(universe)
        atom_indices = sorted({index for selection in selections if selection.profile.kind not in {"solvent", "ion"} for index in selection.atom_indices})
        atoms = universe.atoms[atom_indices] if atom_indices else universe.atoms
        residue_atoms: dict[str, list[int]] = {}
        atom_identifiers = []
        for local_index, atom in enumerate(atoms):
            identifier = _safe_id(f"{atom.chainID or atom.segid or 'unlabeled'}__{atom.resname}_{atom.resid}")
            residue_atoms.setdefault(identifier, []).append(local_index)
            atom_identifiers.append(identifier)
        parameters = freesasa.Parameters({"probe-radius": self.probe_radius_angstrom, "n-points": self.n_points})
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records = []
        previous_verbosity = freesasa.getVerbosity()
        freesasa.setVerbosity(freesasa.nowarnings)
        try:
            for output_order, frame_index in enumerate(frame_indices):
                universe.trajectory[frame_index]
                structure = freesasa.Structure()
                for atom in atoms:
                    chain = (atom.chainID or atom.segid or " ").strip()[:1] or " "
                    structure.addAtom(atom.name.strip(), atom.resname.strip(), str(atom.resid), chain, *map(float, atom.position))
                result = freesasa.calc(structure, parameters)
                row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "SASA_total": float(result.totalArea())}
                for identifier, local_indices in residue_atoms.items():
                    row[f"SASA_residue__{identifier}"] = float(sum(result.atomArea(index) for index in local_indices))
                records.append(row)
        finally:
            freesasa.setVerbosity(previous_verbosity)
        frame_features = pd.DataFrame.from_records(records)
        trajectory_features = pd.DataFrame([{
            "SASA_total_mean": float(frame_features["SASA_total"].mean()),
            "SASA_total_std": float(frame_features["SASA_total"].std(ddof=1)) if len(frame_features) > 1 else np.nan,
        }])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"backend": "freesasa", "backend_version": getattr(freesasa, "__version__", "unknown"), "probe_radius_angstrom": self.probe_radius_angstrom, "n_points": self.n_points, "scope": "non_solvent_non_ion_when_detected", "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(frame_features, trajectory_features, assessment, provenance)


class SolvationShellPlugin:
    """Water-shell and water-bridge counts for topology-identified solvent."""

    plugin_id = "solvation.water_shells"
    version = "1"
    cutoff_angstrom = 3.5
    required_capabilities = frozenset({"coordinates", "solvent"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "solvation", ApplicabilityStatus.NOT_APPLICABLE, "Water-shell analysis requires coordinates and topology-identified solvent.", "frame", "count", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "solvation", ApplicabilityStatus.APPLICABLE, "Coordinates and solvent are available; solvent oxygen availability is checked during extraction.", "frame", "count")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        start = perf_counter()
        universe = mda.Universe(str(topology), str(trajectory))
        selections = discover_component_selections(universe)
        molecular = [item for item in selections if item.profile.kind not in {"solvent", "ion"}]
        solvent_indices = [index for item in selections if item.profile.kind == "solvent" for index in item.atom_indices]
        water_atoms = universe.atoms[solvent_indices]
        oxygen_indices = [atom.index for atom in water_atoms if str(getattr(atom, "element", "")).strip().upper() == "O" or atom.name.strip().upper() in {"O", "OW", "OH2"}]
        water_oxygen = universe.atoms[oxygen_indices]
        if not molecular or not len(water_oxygen):
            unavailable = FeatureAssessment(self.plugin_id, "solvation", ApplicabilityStatus.NOT_APPLICABLE, "No molecular component or solvent oxygen atoms were found.", "frame", "count")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records, total_counts = [], []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            shell_sets = {}
            row = {"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time)}
            for component in molecular:
                component_atoms = universe.atoms[list(component.atom_indices)]
                pairs, _ = capped_distance(component_atoms.positions, water_oxygen.positions, max_cutoff=self.cutoff_angstrom, return_distances=True)
                shell = set(pairs[:, 1].tolist()) if len(pairs) else set()
                identifier = _safe_id(component.profile.component_id)
                shell_sets[identifier] = shell
                row[f"SOLVATION_shell_water_count__{identifier}"] = len(shell)
            for left, right in combinations(shell_sets, 2):
                row[f"SOLVATION_water_bridge_count__{left}__{right}"] = len(shell_sets[left] & shell_sets[right])
            total = len(set().union(*shell_sets.values()))
            row["SOLVATION_total_shell_water_count"] = total
            records.append(row)
            total_counts.append(total)
        trajectory_features = pd.DataFrame([{"SOLVATION_total_shell_water_mean": float(np.mean(total_counts)), "SOLVATION_total_shell_water_std": float(np.std(total_counts, ddof=1)) if len(total_counts) > 1 else np.nan}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"cutoff_angstrom": self.cutoff_angstrom, "water_oxygen_names": ["O", "OW", "OH2"], "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)


class MolecularDipolePlugin:
    """Dipole magnitude for a neutral selected molecular scope with charges."""

    plugin_id = "electrostatics.molecular_dipole"
    version = "1"
    required_capabilities = frozenset({"coordinates", "partial_charges"})

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        missing = tuple(sorted(self.required_capabilities - profile.capabilities))
        if missing:
            return FeatureAssessment(self.plugin_id, "electrostatics", ApplicabilityStatus.NOT_APPLICABLE, "Dipole calculation requires finite partial charges and coordinates.", "frame", "elementary_charge_angstrom", missing_capabilities=missing)
        return FeatureAssessment(self.plugin_id, "electrostatics", ApplicabilityStatus.APPLICABLE, "Finite partial charges and coordinates are available; scope neutrality is checked during extraction.", "frame", "elementary_charge_angstrom")

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        profile = profile_system(topology, trajectory)
        assessment = self.assess(profile)
        if assessment.status is not ApplicabilityStatus.APPLICABLE:
            return PluginResult(pd.DataFrame(), pd.DataFrame(), assessment, None)
        universe = mda.Universe(str(topology), str(trajectory))
        selections = discover_component_selections(universe)
        indices = sorted({index for item in selections if item.profile.kind not in {"solvent", "ion"} for index in item.atom_indices})
        atoms = universe.atoms[indices] if indices else universe.atoms
        charges = np.asarray(atoms.charges, dtype=float)
        net_charge = float(charges.sum())
        if not np.isclose(net_charge, 0.0, atol=1e-6):
            unavailable = FeatureAssessment(self.plugin_id, "electrostatics", ApplicabilityStatus.NOT_APPLICABLE, "Dipole is origin-dependent for the selected non-neutral scope.", "frame", "elementary_charge_angstrom")
            return PluginResult(pd.DataFrame(), pd.DataFrame(), unavailable, None)
        if frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        start = perf_counter()
        frame_indices = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            frame_indices = frame_indices[:max_frames]
        records = []
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            dipole = np.sum(charges[:, None] * atoms.positions, axis=0)
            records.append({"frame_index": frame_index, "frame_order": output_order, "time_ps": float(universe.trajectory.ts.time), "DIPOLE_magnitude_e_angstrom": float(np.linalg.norm(dipole)), "DIPOLE_net_charge_e": net_charge})
        values = [row["DIPOLE_magnitude_e_angstrom"] for row in records]
        trajectory_features = pd.DataFrame([{"DIPOLE_magnitude_mean_e_angstrom": float(np.mean(values)), "DIPOLE_magnitude_std_e_angstrom": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan}])
        provenance = FeatureProvenance(self.plugin_id, self.version, {"scope": "non_solvent_non_ion_when_detected", "neutrality_tolerance_e": 1e-6, "pbc": False, "max_frames": max_frames, "frame_stride": frame_stride}, str(topology), str(trajectory), perf_counter() - start)
        return PluginResult(pd.DataFrame.from_records(records), trajectory_features, assessment, provenance)
