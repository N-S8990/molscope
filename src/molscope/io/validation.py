"""Molecular identity and integrity checks for trajectory ingestion.

This module deliberately separates what is *observed* in a topology or
trajectory from chemistry inferred by a later analyzer.  In particular, an
atom index is a source-file identity, not a universal chemical identifier.
The complete identity table makes that limitation explicit and gives later
stages an auditable mapping from feature values back to their input atoms.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any

import MDAnalysis as mda
import numpy as np


class InputIntegrityStatus(str, Enum):
    """Outcome of a read-only topology/trajectory integrity check."""

    VALID = "valid"
    VALID_WITH_WARNINGS = "valid_with_warnings"
    INVALID = "invalid"


@dataclass(frozen=True)
class InputFileEvidence:
    """Stable file facts recorded without assuming content equivalence."""

    path: str
    format_hint: str | None
    size_bytes: int
    modified_ns: int
    sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResidueIdentity:
    """Topology-resident residue/equivalent-unit identity.

    ``index`` is the zero-based MDAnalysis topology index. ``resid`` and
    ``insertion_code`` preserve source labels and are not assumed unique.
    """

    index: int
    resname: str
    resid: int | str
    insertion_code: str
    segment_id: str
    chain_id: str
    atom_indices: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AtomIdentity:
    """Topology-resident atom identity linked to its containing residue."""

    index: int
    name: str
    element: str | None
    atom_type: str | None
    mass: float | None
    charge: float | None
    residue_index: int
    segment_id: str
    chain_id: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TopologyIdentity:
    """Identity table and connectivity availability from a parsed topology."""

    source: InputFileEvidence
    atom_count: int
    residue_count: int
    has_bonds: bool
    residues: tuple[ResidueIdentity, ...]
    atoms: tuple[AtomIdentity, ...]

    def to_dict(self, *, include_identities: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source.to_dict(),
            "atom_count": self.atom_count,
            "residue_count": self.residue_count,
            "has_bonds": self.has_bonds,
        }
        if include_identities:
            payload["residues"] = [item.to_dict() for item in self.residues]
            payload["atoms"] = [item.to_dict() for item in self.atoms]
        return payload


@dataclass(frozen=True)
class TrajectoryIntegrity:
    """Facts from reading trajectory coordinates under an explicit audit scope."""

    source: InputFileEvidence
    topology_atom_count: int
    trajectory_atom_count: int | None
    frame_count: int | None
    frames_checked: int
    frame_stride: int
    time_start_ps: float | None
    time_end_ps: float | None
    timestep_ps: float | None
    time_monotonic: bool | None
    finite_coordinates: bool | None
    periodic_cell_frame_fraction: float | None
    complete_scan: bool
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InputIntegrityReport:
    """Combined topology identity and trajectory compatibility result."""

    status: InputIntegrityStatus
    topology: TopologyIdentity | None
    trajectory: TrajectoryIntegrity | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    schema_version: str = "1"

    def to_dict(self, *, include_identities: bool = False) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "topology": self.topology.to_dict(include_identities=include_identities) if self.topology else None,
            "trajectory": self.trajectory.to_dict() if self.trajectory else None,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "schema_version": self.schema_version,
        }


def _text(value: object) -> str:
    return str(value).strip() if value is not None else ""


def _safe_attribute(item: object, name: str) -> object | None:
    try:
        return getattr(item, name)
    except (AttributeError, mda.exceptions.NoDataError):
        return None


def _number_or_none(value: object | None) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if np.isfinite(number) else None


def _file_evidence(path: str | Path, *, include_checksum: bool = False) -> InputFileEvidence:
    resolved = Path(path).resolve()
    stat = resolved.stat()
    checksum = None
    if include_checksum:
        digest = hashlib.sha256()
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1_048_576), b""):
                digest.update(chunk)
        checksum = digest.hexdigest()
    suffix = resolved.suffix.lower().lstrip(".") or None
    return InputFileEvidence(
        path=str(resolved),
        format_hint=suffix,
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=checksum,
    )


def topology_identity(topology: str | Path, *, include_checksum: bool = False) -> TopologyIdentity:
    """Read a topology and return source-preserving atom and residue identities."""
    evidence = _file_evidence(topology, include_checksum=include_checksum)
    universe = mda.Universe(evidence.path)
    residues = []
    for residue in universe.residues:
        atoms = residue.atoms
        chain_values = _safe_attribute(atoms, "chainIDs")
        segment_values = _safe_attribute(atoms, "segids")
        chain_ids = {_text(value) for value in chain_values if _text(value)} if chain_values is not None else set()
        segment_ids = {_text(value) for value in segment_values if _text(value)} if segment_values is not None else set()
        residue_id = _safe_attribute(residue, "resid")
        residues.append(ResidueIdentity(
            index=int(residue.ix),
            resname=_text(residue.resname),
            resid=residue_id if residue_id is not None else "",
            insertion_code=_text(_safe_attribute(residue, "icode")),
            segment_id=next(iter(segment_ids)) if len(segment_ids) == 1 else "",
            chain_id=next(iter(chain_ids)) if len(chain_ids) == 1 else "",
            atom_indices=tuple(int(atom.index) for atom in atoms),
        ))
    atoms = []
    for atom in universe.atoms:
        atoms.append(AtomIdentity(
            index=int(atom.index),
            name=_text(atom.name),
            element=_text(_safe_attribute(atom, "element")) or None,
            atom_type=_text(_safe_attribute(atom, "type")) or None,
            mass=_number_or_none(_safe_attribute(atom, "mass")),
            charge=_number_or_none(_safe_attribute(atom, "charge")),
            residue_index=int(atom.residue.ix),
            segment_id=_text(_safe_attribute(atom, "segid")),
            chain_id=_text(_safe_attribute(atom, "chainID")),
        ))
    try:
        has_bonds = bool(len(universe.bonds))
    except (AttributeError, mda.exceptions.NoDataError):
        has_bonds = False
    return TopologyIdentity(
        source=evidence,
        atom_count=len(universe.atoms),
        residue_count=len(universe.residues),
        has_bonds=has_bonds,
        residues=tuple(residues),
        atoms=tuple(atoms),
    )


def inspect_md_inputs(
    topology: str | Path,
    trajectory: str | Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    include_checksum: bool = False,
) -> InputIntegrityReport:
    """Validate topology/trajectory compatibility without modifying the inputs.

    By default every trajectory frame is inspected. ``max_frames`` makes the
    audit explicitly partial; the report records this so it cannot be mistaken
    for a complete coordinate-integrity scan.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when supplied.")
    try:
        topology_info = topology_identity(topology, include_checksum=include_checksum)
    except Exception as exc:  # Reader exceptions vary across supported formats.
        return InputIntegrityReport(
            status=InputIntegrityStatus.INVALID,
            topology=None,
            trajectory=None,
            errors=(f"Unable to read topology: {type(exc).__name__}: {exc}",),
        )
    try:
        trajectory_evidence = _file_evidence(trajectory, include_checksum=include_checksum)
        universe = mda.Universe(topology_info.source.path, trajectory_evidence.path)
        trajectory_atom_count = int(getattr(universe.trajectory, "n_atoms", len(universe.atoms)))
        frame_count = len(universe.trajectory)
    except Exception as exc:  # Reader exceptions vary across supported formats.
        return InputIntegrityReport(
            status=InputIntegrityStatus.INVALID,
            topology=topology_info,
            trajectory=None,
            errors=(f"Unable to read topology with trajectory: {type(exc).__name__}: {exc}",),
        )

    errors = []
    warnings = []
    if trajectory_atom_count != topology_info.atom_count:
        errors.append(
            f"Topology has {topology_info.atom_count} atoms but trajectory reports {trajectory_atom_count}."
        )
    indices = list(range(0, frame_count, frame_stride))
    if max_frames is not None:
        indices = indices[:max_frames]
    if not indices:
        errors.append("Trajectory contains no frames selected for inspection.")

    times = []
    finite_coordinates = True
    periodic_frames = 0
    for frame_index in indices:
        timestep = universe.trajectory[frame_index]
        positions = universe.atoms.positions
        if not np.isfinite(positions).all():
            finite_coordinates = False
        time_value = _number_or_none(getattr(timestep, "time", None))
        if time_value is not None:
            times.append(time_value)
        raw_dimensions = getattr(timestep, "dimensions", None)
        dimensions = np.asarray(raw_dimensions if raw_dimensions is not None else [], dtype=float).reshape(-1)
        if len(dimensions) >= 3 and np.isfinite(dimensions[:3]).all() and np.all(dimensions[:3] > 0):
            periodic_frames += 1

    if not finite_coordinates:
        errors.append("One or more inspected frames contain non-finite coordinates.")
    time_monotonic = None
    if times:
        time_monotonic = bool(np.all(np.diff(times) >= 0))
        if not time_monotonic:
            warnings.append("Frame times are not monotonic over the inspected frames.")
    else:
        warnings.append("Trajectory reader did not provide finite frame times.")
    if max_frames is not None and len(indices) < frame_count:
        warnings.append("Coordinate integrity check used a limited frame subset.")
    if periodic_frames not in {0, len(indices)}:
        warnings.append("Periodic-cell metadata is present for only some inspected frames.")

    timestep_ps = _number_or_none(getattr(universe.trajectory, "dt", None))
    trajectory_info = TrajectoryIntegrity(
        source=trajectory_evidence,
        topology_atom_count=topology_info.atom_count,
        trajectory_atom_count=trajectory_atom_count,
        frame_count=frame_count,
        frames_checked=len(indices),
        frame_stride=frame_stride,
        time_start_ps=times[0] if times else None,
        time_end_ps=times[-1] if times else None,
        timestep_ps=timestep_ps,
        time_monotonic=time_monotonic,
        finite_coordinates=finite_coordinates if indices else None,
        periodic_cell_frame_fraction=(periodic_frames / len(indices)) if indices else None,
        complete_scan=len(indices) == frame_count,
        warnings=tuple(warnings),
    )
    status = (
        InputIntegrityStatus.INVALID if errors else
        InputIntegrityStatus.VALID_WITH_WARNINGS if warnings else
        InputIntegrityStatus.VALID
    )
    return InputIntegrityReport(
        status=status,
        topology=topology_info,
        trajectory=trajectory_info,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )
