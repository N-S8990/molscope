"""Unit tests for trajectory imaging and transformations."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest

from molscope.io.transformations import (
    ImagingPolicy,
    apply_imaging_transformations,
    has_periodic_box,
    load_universe,
)

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"
PERIODIC_PDB = Path(__file__).parents[1] / "fixtures" / "protein_ligand_contacts.pdb"


def test_imaging_policy_defaults_and_dict():
    policy = ImagingPolicy()
    assert policy.unwrap is False
    assert policy.center_selection is None
    assert policy.center_to == "box"
    assert policy.wrap is False
    assert policy.require_pbc is False

    d = policy.to_dict()
    restored = ImagingPolicy.from_dict(d)
    assert restored == policy


def test_has_periodic_box():
    u = mda.Universe(str(FIXTURE_PDB))
    u.dimensions = None
    assert has_periodic_box(u) is False

    u.dimensions = np.zeros(6)
    assert has_periodic_box(u) is False

    u.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
    assert has_periodic_box(u) is True


def test_imaging_policy_require_pbc_raises_on_nonperiodic():
    u = mda.Universe(str(FIXTURE_PDB))
    u.dimensions = None
    policy = ImagingPolicy(require_pbc=True, unwrap=True)
    with pytest.raises(ValueError, match="requires periodic boundary conditions"):
        apply_imaging_transformations(u, policy)


def test_apply_imaging_transformations_on_periodic_universe(tmp_path):
    pdb_path = tmp_path / "periodic_test.pdb"
    pdb_path.write_text(
        "CRYST1   50.000   50.000   50.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  CA  ALA A   1       2.000   2.000   2.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2      48.000   2.000   2.000  1.00  0.00           C\nEND\n"
    )
    u = mda.Universe(str(pdb_path))
    assert has_periodic_box(u) is True

    policy = ImagingPolicy(
        unwrap=True,
        center_selection="protein",
        wrap=True,
        wrap_compound="fragments",
    )
    applied = apply_imaging_transformations(u, policy)
    assert any("unwrap" in step for step in applied)
    assert any("center_in_box" in step for step in applied)
    assert any("wrap" in step for step in applied)


def test_load_universe_helper(tmp_path):
    u, applied = load_universe(FIXTURE_PDB)
    assert len(u.atoms) > 0
    assert applied == []

    # With non-PBC center to origin
    policy = ImagingPolicy(center_selection="name CA", center_to="origin")
    u2, applied2 = load_universe(FIXTURE_PDB, imaging_policy=policy)
    assert any("center_to_origin" in step for step in applied2)
