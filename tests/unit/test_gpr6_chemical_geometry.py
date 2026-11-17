"""Focused checks for chemical geometry calculation primitives."""

import numpy as np
import pytest

from molscope.features.geometry.generic import ring_normal, value

# (feature_id, geometry_kind, left_atom_names, right_atom_names, (left_resid, right_resid))
DEFINITIONS = (
    ("ASP118_SER155", "min", ("OD1", "OD2"), ("OG",), (118, 155)),
    ("ASP118_ASN321", "min", ("OD1", "OD2"), ("OD1", "ND2"), (118, 321)),
    ("SER155_ASN321", "centroid", ("OG",), ("OD1", "ND2"), (155, 321)),
    ("HSE128_PHE152", "min", ("ND1", "NE2"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (128, 152)),
    ("LEU148_PHE152", "min", ("CD1", "CD2"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (148, 152)),
    ("PHE152_TRP292_centroid", "centroid", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"), (152, 292)),
    ("PHE152_TRP292_normal", "normal", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"), (152, 292)),
    ("PHE152_PHE295_centroid", "centroid", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (152, 295)),
    ("PHE152_PHE295_normal", "normal", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (152, 295)),
    ("PHE234_TRP292_centroid", "centroid", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"), (234, 292)),
    ("PHE234_TRP292_normal", "normal", ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), ("CG", "CD1", "CD2", "NE1", "CE2", "CE3", "CZ2", "CZ3", "CH2"), (234, 292)),
    ("VAL156_PHE288", "min", ("CG1", "CG2"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (156, 288)),
    ("VAL237_PHE288", "min", ("CG1", "CG2"), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (237, 288)),
    ("ARG166_THR274", "min", ("NE", "CZ", "NH1", "NH2"), ("OG1",), (166, 274)),
    ("ASP165_TYR176", "min", ("OD1", "OD2"), ("OH",), (165, 176)),
    ("ARG166_GLY277", "min", ("NE", "CZ", "NH1", "NH2"), ("O",), (166, 277)),
    ("ARG166_TYR329", "centroid", ("CZ",), ("CG", "CD1", "CD2", "CE1", "CE2", "CZ"), (166, 329)),
    ("CYS209_CYS216", "min", ("SG",), ("SG",), (209, 216)),
)


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
