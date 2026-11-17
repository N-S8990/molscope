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
    has_pbc: bool | None = None
    wrapped_coordinates_detected: bool = False
    broken_bonds_detected: bool = False
    broken_bond_count: int = 0
    imaging_recommendations: tuple[str, ...] = ()
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


RECOMMENDED_IMAGING_FIXES: str = (
    "Recommended imaging fixes:\n"
    "1. MolScope / MDAnalysis on-the-fly transformations (in Python):\n"
    "   from molscope.io.transformations import apply_imaging_transformations, ImagingPolicy\n"
    "   apply_imaging_transformations(universe, ImagingPolicy(unwrap=True, center_selection='protein', wrap=True))\n"
    "2. GROMACS (gmx trjconv):\n"
    "   # Step 1: Make molecules whole across periodic boundaries:\n"
    "   gmx trjconv -s topol.tpr -f traj.xtc -o traj_whole.xtc -pbc whole\n"
    "   # Step 2: Center solute and wrap solvent around it:\n"
    "   gmx trjconv -s topol.tpr -f traj_whole.xtc -o traj_imaged.xtc -center -pbc mol -ur compact\n"
    "3. MDTraj / mdconvert:\n"
    "   mdconvert traj.xtc -o traj_imaged.xtc -t topol.pdb"
)


def detect_pbc_wrapping_artifacts(
    universe: mda.Universe,
    *,
    frame_index: int | None = None,
    bond_distance_cutoff: float = 3.0,
    pbc_intact_cutoff: float = 2.5,
    ca_distance_cutoff: float = 5.0,
    ca_pbc_intact_cutoff: float = 4.5,
) -> dict[str, Any]:
    """Detect wrapped or un-imaged coordinates across periodic boundary conditions.

    Identifies:
    1. Broken covalent bonds where raw Cartesian distance exceeds physical limits
       (> 3.0 Å) but minimum-image PBC distance is physical (< 2.5 Å).
    2. Broken protein backbone Cα-Cα virtual bonds (> 5.0 Å raw vs <= 4.5 Å PBC).
    3. Multi-atom residues split across opposite unit cell boundaries.
    """
    dims = getattr(universe, "dimensions", None)
    if dims is None or len(dims) < 3 or not np.isfinite(dims[:3]).all() or not np.all(np.asarray(dims[:3]) > 0):
        return {
            "has_pbc": False,
            "wrapped_coordinates_detected": False,
            "broken_bonds_detected": False,
            "broken_bond_count": 0,
            "issues": [],
            "warnings": (),
        }

    from MDAnalysis.lib.distances import calc_bonds

    pos = universe.atoms.positions
    if not np.isfinite(pos).all():
        return {
            "has_pbc": True,
            "wrapped_coordinates_detected": False,
            "broken_bonds_detected": False,
            "broken_bond_count": 0,
            "issues": ["Non-finite coordinates present"],
            "warnings": (),
        }

    issues: list[dict[str, Any]] = []
    broken_bond_count = 0
    box = np.asarray(dims, dtype=float)

    # 1. Explicit covalent bonds if available
    has_bonds = hasattr(universe, "bonds") and len(universe.bonds) > 0
    if has_bonds:
        indices = universe.bonds.indices
        idx1 = indices[:, 0]
        idx2 = indices[:, 1]
        raw_diff = pos[idx1] - pos[idx2]
        raw_dists = np.sqrt(np.sum(raw_diff ** 2, axis=-1))
        pbc_dists = calc_bonds(pos[idx1], pos[idx2], box=box)

        # Wrapped bonds: raw distance > cutoff, but PBC distance is small
        wrapped_mask = (raw_dists > bond_distance_cutoff) & (pbc_dists < pbc_intact_cutoff)
        # Stretched/broken bonds: raw distance > cutoff, and even PBC distance is large
        stretched_mask = (raw_dists > bond_distance_cutoff) & (pbc_dists >= pbc_intact_cutoff)

        wrapped_indices = np.where(wrapped_mask)[0]
        stretched_indices = np.where(stretched_mask)[0]
        broken_bond_count = len(wrapped_indices) + len(stretched_indices)

        for b_idx in wrapped_indices[:10]:
            a1 = universe.atoms[idx1[b_idx]]
            a2 = universe.atoms[idx2[b_idx]]
            issues.append({
                "kind": "wrapped_covalent_bond",
                "atom1_index": int(a1.index),
                "atom1_name": str(a1.name),
                "atom1_resname": str(a1.resname),
                "atom2_index": int(a2.index),
                "atom2_name": str(a2.name),
                "atom2_resname": str(a2.resname),
                "raw_distance_angstrom": float(raw_dists[b_idx]),
                "pbc_distance_angstrom": float(pbc_dists[b_idx]),
            })
        for b_idx in stretched_indices[:5]:
            a1 = universe.atoms[idx1[b_idx]]
            a2 = universe.atoms[idx2[b_idx]]
            issues.append({
                "kind": "broken_covalent_bond",
                "atom1_index": int(a1.index),
                "atom1_name": str(a1.name),
                "atom1_resname": str(a1.resname),
                "atom2_index": int(a2.index),
                "atom2_name": str(a2.name),
                "atom2_resname": str(a2.resname),
                "raw_distance_angstrom": float(raw_dists[b_idx]),
                "pbc_distance_angstrom": float(pbc_dists[b_idx]),
            })

    # 2. Protein Cα-Cα backbone continuity check (especially critical when no explicit bonds)
    try:
        ca_atoms = universe.select_atoms("name CA")
    except Exception:
        ca_atoms = []
    if len(ca_atoms) >= 2:
        for i in range(len(ca_atoms) - 1):
            a1 = ca_atoms[i]
            a2 = ca_atoms[i + 1]
            same_chain = (
                getattr(a1, "chainID", "") == getattr(a2, "chainID", "")
                and getattr(a1, "segid", "") == getattr(a2, "segid", "")
            )
            is_consecutive_resid = False
            try:
                is_consecutive_resid = (int(a2.resid) == int(a1.resid) + 1)
            except (ValueError, TypeError):
                is_consecutive_resid = (a2.resindex == a1.resindex + 1)

            if same_chain and is_consecutive_resid:
                raw_d = float(np.linalg.norm(pos[a1.index] - pos[a2.index]))
                pbc_d = float(calc_bonds(pos[a1.index], pos[a2.index], box=box))
                if raw_d > ca_distance_cutoff and pbc_d <= ca_pbc_intact_cutoff:
                    broken_bond_count += 1
                    issues.append({
                        "kind": "wrapped_protein_backbone",
                        "atom1_index": int(a1.index),
                        "atom1_name": "CA",
                        "atom1_resname": str(a1.resname),
                        "atom2_index": int(a2.index),
                        "atom2_name": "CA",
                        "atom2_resname": str(a2.resname),
                        "raw_distance_angstrom": raw_d,
                        "pbc_distance_angstrom": pbc_d,
                    })

    # 3. Residue intra-molecular coordinate span (for multi-atom non-solvent residues)
    min_box_dim = float(np.min(box[:3]))
    half_box = min_box_dim / 2.0
    try:
        sample_residues = [
            r for r in universe.residues[:200]
            if len(r.atoms) >= 3 and r.resname.strip().upper() not in {"HOH", "WAT", "TIP3", "TIP3P", "SOL"}
        ]
        for r in sample_residues:
            r_pos = r.atoms.positions
            spans = np.ptp(r_pos, axis=0)
            if np.any(spans > max(15.0, half_box)):
                issues.append({
                    "kind": "split_residue",
                    "residue_name": str(r.resname),
                    "residue_id": str(r.resid),
                    "coordinate_span": float(np.max(spans)),
                    "half_box_dimension": half_box,
                })
                broken_bond_count += 1
                break
    except Exception:
        pass

    wrapped_detected = any(item["kind"] in ("wrapped_covalent_bond", "wrapped_protein_backbone", "split_residue") for item in issues)
    broken_detected = broken_bond_count > 0

    warnings_list = []
    if broken_detected or wrapped_detected:
        frame_str = f" in frame {frame_index}" if frame_index is not None else ""
        issue_examples = []
        for issue in issues[:3]:
            if "raw_distance_angstrom" in issue:
                issue_examples.append(
                    f"{issue['atom1_resname']}:{issue['atom1_name']}#{issue['atom1_index']}–"
                    f"{issue['atom2_resname']}:{issue['atom2_name']}#{issue['atom2_index']} "
                    f"(raw: {issue['raw_distance_angstrom']:.2f} Å, PBC: {issue['pbc_distance_angstrom']:.2f} Å)"
                )
            elif "coordinate_span" in issue:
                issue_examples.append(
                    f"residue {issue['residue_name']}:{issue['residue_id']} "
                    f"(coordinate span: {issue['coordinate_span']:.2f} Å > half-box {issue['half_box_dimension']:.2f} Å)"
                )
        example_str = "; ".join(issue_examples) if issue_examples else "multiple broken bonds"
        box_str = ", ".join(f"{v:.1f}" for v in box[:3])

        msg = (
            f"Periodic boundary condition (PBC) coordinate wrapping / un-imaged coordinates detected{frame_str}: "
            f"{broken_bond_count} broken bond(s) or split residue(s) across unit cell boundaries "
            f"(box dimensions: [{box_str}] Å; e.g. {example_str}). "
            f"Computing distances, contacts, Rg, or RMSD on wrapped coordinates without unwrapping produces "
            f"catastrophic box-jump artifacts.\n{RECOMMENDED_IMAGING_FIXES}"
        )
        warnings_list.append(msg)

    return {
        "has_pbc": True,
        "wrapped_coordinates_detected": wrapped_detected,
        "broken_bonds_detected": broken_detected,
        "broken_bond_count": broken_bond_count,
        "issues": issues,
        "warnings": tuple(warnings_list),
    }


def inspect_md_inputs(
    topology: str | Path,
    trajectory: str | Path,
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
    include_checksum: bool = False,
    check_pbc_wrapping: bool = True,
    require_imaged: bool = False,
) -> InputIntegrityReport:
    """Validate topology/trajectory compatibility without modifying the inputs.

    By default every trajectory frame is inspected. ``max_frames`` makes the
    audit explicitly partial; the report records this so it cannot be mistaken
    for a complete coordinate-integrity scan.

    When ``check_pbc_wrapping`` is True and periodic box dimensions are present,
    inspects coordinates for broken covalent bonds or wrapped molecules across
    periodic unit cell boundaries. If detected, emits loud warnings detailing
    recommended imaging fixes (e.g. apply_imaging_transformations, gmx trjconv,
    mdconvert). Set ``require_imaged=True`` to treat wrapped coordinates as fatal
    errors (InputIntegrityStatus.INVALID).
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
    frames_checked = 0
    wrapped_frames = 0
    total_broken_bonds = 0
    pbc_warnings: list[str] = []

    for frame_index in indices:
        try:
            timestep = universe.trajectory[frame_index]
            positions = universe.atoms.positions
        except Exception as exc:
            errors.append(f"Unable to read trajectory frame {frame_index}: {type(exc).__name__}: {exc}")
            break
        frames_checked += 1
        if not np.isfinite(positions).all():
            finite_coordinates = False
        time_value = _number_or_none(getattr(timestep, "time", None))
        if time_value is not None:
            times.append(time_value)
        raw_dimensions = getattr(timestep, "dimensions", None)
        dimensions = np.asarray(raw_dimensions if raw_dimensions is not None else [], dtype=float).reshape(-1)
        if len(dimensions) >= 3 and np.isfinite(dimensions[:3]).all() and np.all(dimensions[:3] > 0):
            periodic_frames += 1
            if check_pbc_wrapping:
                pbc_res = detect_pbc_wrapping_artifacts(universe, frame_index=frame_index)
                if pbc_res["broken_bonds_detected"] or pbc_res["wrapped_coordinates_detected"]:
                    wrapped_frames += 1
                    total_broken_bonds += pbc_res["broken_bond_count"]
                    if not pbc_warnings:
                        pbc_warnings.extend(pbc_res["warnings"])

    if pbc_warnings:
        if require_imaged:
            errors.extend(pbc_warnings)
        else:
            warnings.extend(pbc_warnings)

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
    if periodic_frames not in {0, frames_checked}:
        warnings.append("Periodic-cell metadata is present for only some inspected frames.")

    timestep_ps = _number_or_none(getattr(universe.trajectory, "dt", None))
    trajectory_info = TrajectoryIntegrity(
        source=trajectory_evidence,
        topology_atom_count=topology_info.atom_count,
        trajectory_atom_count=trajectory_atom_count,
        frame_count=frame_count,
        frames_checked=frames_checked,
        frame_stride=frame_stride,
        time_start_ps=times[0] if times else None,
        time_end_ps=times[-1] if times else None,
        timestep_ps=timestep_ps,
        time_monotonic=time_monotonic,
        finite_coordinates=finite_coordinates if frames_checked else None,
        periodic_cell_frame_fraction=(periodic_frames / frames_checked) if frames_checked else None,
        complete_scan=frames_checked == frame_count and not errors,
        has_pbc=periodic_frames > 0,
        wrapped_coordinates_detected=wrapped_frames > 0,
        broken_bonds_detected=total_broken_bonds > 0,
        broken_bond_count=total_broken_bonds,
        imaging_recommendations=tuple(pbc_warnings) if pbc_warnings else (),
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
