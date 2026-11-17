"""Focused checks for the published GPR6 heavy-atom geometry panel."""

import numpy as np
import pytest

from studies.gpr6.analysis_protocol.prepare_chemical_panel import DEFINITIONS, ring_normal, value


def test_panel_has_unique_named_heavy_atom_measurements():
    assert len(DEFINITIONS) == 18
    assert len({definition[0] for definition in DEFINITIONS}) == 18
    assert all(kind in {"min", "centroid", "normal"} for _, kind, *_ in DEFINITIONS)
    assert all(len(residues) == 2 for *_, residues in DEFINITIONS)


def test_group_minimum_and_centroid_distance_are_distinct():
    left = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    right = np.array([[1.0, 0.0, 0.0]])
    assert value("min", left, right) == pytest.approx(1.0)
    assert value("centroid", left, right) == pytest.approx(1.0)
    farther = np.array([[5.0, 0.0, 0.0]])
    assert value("min", left, farther) == pytest.approx(1.0)
    assert value("centroid", left, farther) == pytest.approx(3.0)


def test_aromatic_normal_uses_unsigned_plane_orientation():
    phenyl = np.array([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0],
        [0.0, 2.0, 0.0], [-1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
    ])
    reflected = phenyl.copy()
    reflected[1], reflected[-1] = reflected[-1].copy(), reflected[1].copy()
    assert np.linalg.norm(ring_normal(phenyl)) == pytest.approx(1.0)
    assert value("normal", phenyl, reflected) == pytest.approx(1.0)
    tilted = phenyl.copy()
    tilted[:, 1], tilted[:, 2] = phenyl[:, 2], phenyl[:, 1]
    assert value("normal", phenyl, tilted) == pytest.approx(0.0)
