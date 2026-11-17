"""Trajectory imaging and on-the-fly coordinate transformations.

This module provides reproducible, auditable coordinate transformations using
MDAnalysis on-the-fly transformations. Trajectories crossing periodic boundary
conditions (PBC) can be unwrapped, centered, and wrapped so that distance and
contact measurements reflect true physical distances rather than box-jump
artifacts.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import MDAnalysis as mda
import MDAnalysis.transformations as trans
import numpy as np


@dataclass(frozen=True)
class ImagingPolicy:
    """Declared policy for coordinate transformations and periodic imaging.

    Attributes:
        unwrap: Whether to unwrap molecules broken across periodic boundaries.
        unwrap_selection: Atom selection for unwrapping (default 'all').
        center_selection: Atom selection to center in the box (e.g. 'protein').
        center_to: Where to center: 'box' (center of box) or 'origin'.
        wrap: Whether to wrap molecules back into the unit cell around the centered solute.
        wrap_compound: Unit to preserve whole during wrapping: 'fragments', 'residues', or 'atoms'.
        fit_selection: Optional atom selection to align (superpose) to reference/initial coordinates.
        require_pbc: If True, raises ValueError if the trajectory lacks periodic box dimensions.
    """

    unwrap: bool = False
    unwrap_selection: str = "all"
    center_selection: str | None = None
    center_to: str = "box"
    wrap: bool = False
    wrap_compound: str = "fragments"
    fit_selection: str | None = None
    require_pbc: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> ImagingPolicy:
        if not data:
            return cls()
        known = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known}
        return cls(**filtered)


def has_periodic_box(universe: mda.Universe) -> bool:
    """Check if the universe has valid non-zero periodic box dimensions."""
    dims = getattr(universe, "dimensions", None)
    if dims is None:
        return False
    # Dimensions are [a, b, c, alpha, beta, gamma]
    return bool(len(dims) >= 3 and np.all(np.asarray(dims[:3]) > 0.0))


def _ensure_bonds_for_fragments(universe: mda.Universe) -> None:
    """Ensure topology has bonds so that atom fragments can be computed for unwrap/wrap."""
    if not hasattr(universe, "bonds") or len(universe.bonds) == 0:
        try:
            universe.guess_TopologyAttrs(to_guess=["bonds"])
        except Exception:
            pass


def apply_imaging_transformations(
    universe: mda.Universe,
    policy: ImagingPolicy | dict[str, Any] | None,
) -> list[str]:
    """Attach on-the-fly transformations to universe.trajectory according to policy.

    Returns:
        List of transformation descriptions applied, for provenance logging.
    """
    if policy is None:
        return []
    if isinstance(policy, dict):
        policy = ImagingPolicy.from_dict(policy)

    applied: list[str] = []
    pbc_available = has_periodic_box(universe)

    if policy.require_pbc and not pbc_available:
        raise ValueError(
            "ImagingPolicy requires periodic boundary conditions, but the trajectory "
            "has no periodic cell dimensions."
        )

    transformations = []

    # 1. Unwrapping
    if policy.unwrap:
        if pbc_available:
            _ensure_bonds_for_fragments(universe)
            ag = universe.select_atoms(policy.unwrap_selection)
            if len(ag) > 0:
                transformations.append(trans.unwrap(ag))
                applied.append(f"unwrap({policy.unwrap_selection})")
        elif policy.require_pbc:
            raise ValueError("Cannot unwrap coordinates without periodic box dimensions.")

    # 2. Centering
    if policy.center_selection:
        center_ag = universe.select_atoms(policy.center_selection)
        if len(center_ag) > 0:
            if policy.center_to == "origin":
                transformations.append(trans.center_in_box(center_ag, point=(0.0, 0.0, 0.0)))
                applied.append(f"center_to_origin({policy.center_selection})")
            elif pbc_available:
                transformations.append(trans.center_in_box(center_ag, center="geometry"))
                applied.append(f"center_in_box({policy.center_selection})")
            else:
                # If no PBC, center relative to origin
                transformations.append(trans.center_in_box(center_ag, point=(0.0, 0.0, 0.0)))
                applied.append(f"center_in_box_no_pbc_fallback({policy.center_selection})")

    # 3. Wrapping solvent/environment around centered solute
    if policy.wrap:
        if pbc_available:
            compound = policy.wrap_compound
            if compound not in ("fragments", "residues", "atoms"):
                raise ValueError(f"Invalid wrap_compound: {compound!r}. Must be 'fragments', 'residues', or 'atoms'.")
            if compound == "fragments":
                _ensure_bonds_for_fragments(universe)
            all_atoms = universe.atoms
            transformations.append(trans.wrap(all_atoms, compound=compound))
            applied.append(f"wrap(all, compound={compound})")
        elif policy.require_pbc:
            raise ValueError("Cannot wrap coordinates without periodic box dimensions.")

    # 4. Fitting / Superposition
    if policy.fit_selection:
        fit_ag = universe.select_atoms(policy.fit_selection)
        if len(fit_ag) > 0:
            # Fit to the reference coordinates (first frame / topology)
            transformations.append(trans.fit_rot_trans(fit_ag, fit_ag))
            applied.append(f"fit_rot_trans({policy.fit_selection})")

    if transformations:
        universe.trajectory.add_transformations(*transformations)

    return applied


def load_universe(
    topology: str | Path,
    trajectory: str | Path | None = None,
    imaging_policy: ImagingPolicy | dict[str, Any] | None = None,
) -> tuple[mda.Universe, list[str]]:
    """Standardized Universe loader with optional on-the-fly transformations.

    Args:
        topology: Path to topology file (e.g. PDB, PRMTOP, PSF, GRO).
        trajectory: Optional path to trajectory file (e.g. XTC, DCD, NC, TRR).
        imaging_policy: Optional imaging and transformation configuration.

    Returns:
        A tuple of (universe, applied_transformations_list).
    """
    top_str = str(topology)
    traj_str = str(trajectory) if trajectory is not None else None

    if traj_str is not None:
        universe = mda.Universe(top_str, traj_str)
    else:
        universe = mda.Universe(top_str)

    applied = apply_imaging_transformations(universe, imaging_policy)
    return universe, applied
