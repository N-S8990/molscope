"""Validated starting-point feature families for arbitrary coordinate systems."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.lib.distances import distance_array
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.profiling.system import ComponentSelection, SystemProfile, discover_component_selections, profile_system


@dataclass(frozen=True)
class GenericFeatureConfig:
    contact_cutoff_angstrom: float = 4.5
    use_pbc: bool = False
    include_environment_in_global_features: bool = False
    max_frames: int | None = None
    frame_stride: int = 1

    def __post_init__(self):
        if self.contact_cutoff_angstrom <= 0:
            raise ValueError("contact_cutoff_angstrom must be positive.")
        if self.frame_stride < 1:
            raise ValueError("frame_stride must be positive.")
        if self.max_frames is not None and self.max_frames < 1:
            raise ValueError("max_frames must be positive when supplied.")


@dataclass(frozen=True)
class GenericFeatureResult:
    frame_features: pd.DataFrame
    trajectory_features: pd.DataFrame
    assessments: tuple[FeatureAssessment, ...]
    provenance: tuple[FeatureProvenance, ...]

    def assessments_payload(self) -> list[dict]:
        return [assessment.to_dict() for assessment in self.assessments]

    def provenance_payload(self) -> list[dict]:
        return [item.to_dict() for item in self.provenance]


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")


def _kabsch_align(mobile: np.ndarray, reference: np.ndarray) -> np.ndarray:
    mobile_center = mobile.mean(axis=0)
    reference_center = reference.mean(axis=0)
    mobile_centered = mobile - mobile_center
    reference_centered = reference - reference_center
    left, _, right = np.linalg.svd(mobile_centered.T @ reference_centered)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1.0
        rotation = left @ right
    return mobile_centered @ rotation + reference_center


def _rmsd(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.sum((left - right) ** 2, axis=1))))


def _geometric_radius_of_gyration(positions: np.ndarray) -> float:
    centered = positions - positions.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))


def _asphericity(positions: np.ndarray) -> float:
    centered = positions - positions.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(positions))
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))
    denominator = float(np.sum(eigenvalues))
    return float((eigenvalues[-1] - 0.5 * (eigenvalues[0] + eigenvalues[1])) / denominator) if denominator else 0.0


def _molecular_components(selections: tuple[ComponentSelection, ...]) -> tuple[ComponentSelection, ...]:
    """Exclude environment components from inter-component molecular metrics."""
    return tuple(item for item in selections if item.profile.kind not in {"solvent", "ion"})


def _minimum_interatomic_distance(left: np.ndarray, right: np.ndarray, box: np.ndarray | None = None) -> float:
    """Calculate an exact minimum without allocating a full pairwise matrix."""
    minimum = np.inf
    for start in range(0, len(left), 2_048):
        block = left[start:start + 2_048]
        minimum = min(minimum, float(np.min(distance_array(block, right, box=box))))
    return minimum


def _longest_run_fraction(values: list[bool]) -> float:
    longest = current = 0
    for value in values:
        current = current + 1 if value else 0
        longest = max(longest, current)
    return float(longest / len(values)) if values else np.nan


def _lag_one_autocorrelation(values: list[float]) -> float:
    """A sampling-interval-specific dynamic descriptor, not a timescale."""
    series = np.asarray(values, dtype=float)
    if len(series) < 3 or np.ptp(series) <= 1e-12:
        return np.nan
    return float(np.corrcoef(series[:-1], series[1:])[0, 1])


def _residue_feature_id(residue) -> str:
    atoms = residue.atoms
    chain_ids = {value.strip() for value in atoms.chainIDs if value and value.strip()}
    segids = {value.strip() for value in atoms.segids if value and value.strip()}
    scope = f"chain_{next(iter(chain_ids))}" if len(chain_ids) == 1 else f"segment_{next(iter(segids))}" if len(segids) == 1 else "unlabeled"
    return f"{_safe_id(scope)}__{_safe_id(residue.resname)}_{residue.resid}"


class GenericFeatureExtractor:
    """Extract small, generic feature families with explicit output granularity."""

    version = "2"

    def __init__(self, config: GenericFeatureConfig = GenericFeatureConfig()):
        self.config = config

    def extract(self, topology: str | Path, trajectory: str | Path) -> GenericFeatureResult:
        profile = profile_system(topology, trajectory)
        if self.config.use_pbc and not profile.has_periodic_cell:
            raise ValueError(
                "PBC-aware geometry was requested, but the trajectory does not provide a positive periodic cell. "
                "Do not infer PBC from topology alone."
            )
        universe = mda.Universe(str(topology), str(trajectory))
        universe.trajectory[0]
        selections = discover_component_selections(universe)
        molecular_components = _molecular_components(selections)
        analysis_atoms = universe.atoms
        if not self.config.include_environment_in_global_features and molecular_components:
            molecular_indices = sorted({index for component in molecular_components for index in component.atom_indices})
            analysis_atoms = universe.atoms[molecular_indices]
        if not len(analysis_atoms):
            raise ValueError("No atoms are available for global feature calculation.")
        reference = analysis_atoms.positions.copy()
        frame_indices = list(range(0, len(universe.trajectory), self.config.frame_stride))
        if self.config.max_frames is not None:
            frame_indices = frame_indices[:self.config.max_frames]

        assessments = [
            FeatureAssessment("geometry.global_shape", "geometry", ApplicabilityStatus.APPLICABLE, "Coordinates are available.", "frame", "angstrom"),
            FeatureAssessment("dynamics.kabsch_displacement", "dynamics", ApplicabilityStatus.APPLICABLE, "Coordinates are available.", "frame", "angstrom"),
            FeatureAssessment("dynamics.rmsf", "dynamics", ApplicabilityStatus.APPLICABLE, "Coordinates are available.", "trajectory", "angstrom"),
        ]
        component_status = ApplicabilityStatus.APPLICABLE if len(molecular_components) >= 2 else ApplicabilityStatus.NOT_APPLICABLE
        component_reason = "At least two non-environment molecular components are available." if component_status == ApplicabilityStatus.APPLICABLE else "Fewer than two non-environment molecular components were detected."
        assessments.extend([
            FeatureAssessment("geometry.component_distances", "geometry", component_status, component_reason, "frame", "angstrom"),
            FeatureAssessment("interactions.generic_contacts", "interactions", component_status, component_reason, "frame", "fraction"),
        ])

        start = perf_counter()
        frames = []
        aligned_positions = []
        rg_values, rmsd_values = [], []
        component_pairs = [(left, right) for index, left in enumerate(molecular_components) for right in molecular_components[index + 1:]]
        contact_presence = {f"{_safe_id(left.profile.component_id)}__{_safe_id(right.profile.component_id)}": [] for left, right in component_pairs}
        for output_order, frame_index in enumerate(frame_indices):
            universe.trajectory[frame_index]
            positions = analysis_atoms.positions.copy()
            aligned = _kabsch_align(positions, reference)
            aligned_positions.append(aligned)
            row = {
                "frame_index": frame_index,
                "frame_order": output_order,
                "time_ps": float(universe.trajectory.ts.time),
                "GEOM_global_radius_gyration": _geometric_radius_of_gyration(positions),
                "GEOM_global_asphericity": _asphericity(positions),
                "DYN_kabsch_rmsd_to_first_frame": _rmsd(aligned, reference),
            }
            rg_values.append(row["GEOM_global_radius_gyration"])
            rmsd_values.append(row["DYN_kabsch_rmsd_to_first_frame"])
            for left, right in component_pairs:
                left_atoms = universe.atoms[list(left.atom_indices)]
                right_atoms = universe.atoms[list(right.atom_indices)]
                pair_id = f"{_safe_id(left.profile.component_id)}__{_safe_id(right.profile.component_id)}"
                box = universe.dimensions if self.config.use_pbc else None
                row[f"GEOM_component_com_distance__{pair_id}"] = float(distance_array(
                    left_atoms.positions.mean(axis=0)[None, :], right_atoms.positions.mean(axis=0)[None, :], box=box,
                )[0, 0])
                row[f"GEOM_component_min_atom_distance__{pair_id}"] = _minimum_interatomic_distance(left_atoms.positions, right_atoms.positions, box=box)
                pairs, _ = capped_distance(left_atoms.positions, right_atoms.positions, max_cutoff=self.config.contact_cutoff_angstrom, box=box, return_distances=True)
                row[f"CONTACT_pair_count__{pair_id}"] = int(len(pairs))
                row[f"CONTACT_pair_fraction__{pair_id}"] = float(len(pairs) / (len(left_atoms) * len(right_atoms)))
                contact_presence[pair_id].append(bool(len(pairs)))
            frames.append(row)
        frame_features = pd.DataFrame.from_records(frames)
        aligned_stack = np.asarray(aligned_positions)
        mean_positions = aligned_stack.mean(axis=0)
        rmsf = np.sqrt(np.mean(np.sum((aligned_stack - mean_positions) ** 2, axis=2), axis=0))
        trajectory_row = {
            "DYN_global_rmsf_mean": float(np.mean(rmsf)),
            "DYN_global_rmsf_max": float(np.max(rmsf)),
            "frames_analyzed": len(frame_indices),
            "DYN_radius_gyration_std": float(np.std(rg_values, ddof=1)) if len(rg_values) > 1 else np.nan,
            "DYN_radius_gyration_lag1_autocorrelation": _lag_one_autocorrelation(rg_values),
            "DYN_rmsd_lag1_autocorrelation": _lag_one_autocorrelation(rmsd_values),
        }
        local_atom_indices = {atom.index: local_index for local_index, atom in enumerate(analysis_atoms)}
        for residue in analysis_atoms.residues:
            local_indices = [local_atom_indices[atom.index] for atom in residue.atoms if atom.index in local_atom_indices]
            if not local_indices:
                continue
            residue_rmsf = rmsf[local_indices]
            identifier = _residue_feature_id(residue)
            trajectory_row[f"DYN_residue_rmsf_mean__{identifier}"] = float(np.mean(residue_rmsf))
            trajectory_row[f"DYN_residue_rmsf_max__{identifier}"] = float(np.max(residue_rmsf))
        for pair_id, values in contact_presence.items():
            trajectory_row[f"CONTACT_occupancy__{pair_id}"] = float(np.mean(values)) if values else np.nan
            trajectory_row[f"CONTACT_longest_run_fraction__{pair_id}"] = _longest_run_fraction(values)
        trajectory_features = pd.DataFrame([trajectory_row])
        elapsed = perf_counter() - start
        provenance = (
            FeatureProvenance(
                analyzer_id="generic.coordinate_features",
                analyzer_version=self.version,
                parameters=asdict(self.config),
                source_topology=str(topology),
                source_trajectory=str(trajectory),
                elapsed_seconds=elapsed,
            ),
        )
        return GenericFeatureResult(
            frame_features=frame_features,
            trajectory_features=trajectory_features,
            assessments=tuple(assessments),
            provenance=provenance,
        )
