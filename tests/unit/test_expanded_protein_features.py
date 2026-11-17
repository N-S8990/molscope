"""Targeted checks for curated receptor distances and side-chain torsions."""

from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_dihedrals
import numpy as np
import pytest

from molscope.contracts import ApplicabilityStatus
from molscope.features.plugins import get_plugin


REFERENCE = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"
CONTACT_REFERENCE = Path(__file__).parents[1] / "fixtures" / "protein_ligand_contacts.pdb"


def test_explicit_ca_pairs_can_emit_distances_without_constant_contacts(tmp_path):
    path = tmp_path / "ca.pdb"
    path.write_text(
        "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      4  CA  ALA A   4       7.000   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    plugin = get_plugin("protein.ca_distances_contacts", {
        "residue_pairs": [[1, 4]], "include_contacts": False, "max_pairs": 1,
    })
    result = plugin.extract(path, path)
    assert result.assessment.status is ApplicabilityStatus.APPLICABLE
    measured = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"])
    assert measured.shape == (1, 1)
    assert measured.iloc[0, 0] == pytest.approx(7.0)
    assert measured.columns[0].startswith("CA_DISTANCE_angstrom__")
    assert result.trajectory_features.empty
    with pytest.raises(ValueError, match="one Cα"):
        get_plugin("protein.ca_distances_contacts", {
            "residue_pairs": [[1, 9]], "include_contacts": False,
        }).extract(path, path)


def test_sidechain_chi_uses_named_atoms_and_circular_encoding():
    plugin = get_plugin("protein.sidechain_torsions", {"residue_ids": [1]})
    result = plugin.extract(REFERENCE, REFERENCE, max_frames=1)
    assert result.assessment.status is ApplicabilityStatus.APPLICABLE
    measured = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"])
    assert measured.shape == (1, 4)
    assert np.isfinite(measured.to_numpy()).all()
    universe = mda.Universe(str(REFERENCE))
    residue = universe.select_atoms("protein and resid 1").residues[0]
    atoms = {atom.name: atom for atom in residue.atoms}
    for name, quartet in {
        "chi1": ("N", "CA", "CB", "CG"),
        "chi2": ("CA", "CB", "CG", "CD1"),
    }.items():
        expected = calc_dihedrals(*(atoms[part].position[None, :] for part in quartet))[0]
        sin_column = next(column for column in measured if f"SIDECHAIN_{name}__" in column and column.endswith("__sin"))
        cos_column = next(column for column in measured if f"SIDECHAIN_{name}__" in column and column.endswith("__cos"))
        observed_sin = measured.loc[0, sin_column]
        observed_cos = measured.loc[0, cos_column]
        assert observed_sin == pytest.approx(np.sin(expected), abs=1e-6)
        assert observed_cos == pytest.approx(np.cos(expected), abs=1e-6)
        assert observed_sin ** 2 + observed_cos ** 2 == pytest.approx(1.0, abs=1e-6)


def test_sidechain_missing_named_atoms_has_explicit_status(tmp_path):
    path = tmp_path / "incomplete.pdb"
    path.write_text(
        "ATOM      1  N   PHE A   1       0.000   0.000   0.000  1.00  0.00           N\n"
        "ATOM      2  CA  PHE A   1       1.000   0.000   0.000  1.00  0.00           C\n"
        "ATOM      3  CB  PHE A   1       1.000   1.000   0.000  1.00  0.00           C\nEND\n"
    )
    result = get_plugin("protein.sidechain_torsions", {"residue_ids": [1]}).extract(path, path)
    assert result.assessment.status is ApplicabilityStatus.NOT_APPLICABLE
    assert result.frame_features.empty


def test_sidechain_chi3_and_chi4_use_named_arg_atoms():
    plugin = get_plugin("protein.sidechain_torsions", {"residue_ids": [16]})
    result = plugin.extract(REFERENCE, REFERENCE, max_frames=1)
    measured = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"])
    assert any("SIDECHAIN_chi3__" in column for column in measured)
    assert any("SIDECHAIN_chi4__" in column for column in measured)
    assert np.isfinite(measured.to_numpy()).all()


def test_closest_heavy_residue_distance_plugin_emits_bounded_distances_and_contacts():
    plugin = get_plugin("protein.residue_heavy_distances_contacts", {
        "max_pairs": 20, "minimum_sequence_separation": 1,
    })
    result = plugin.extract(CONTACT_REFERENCE, CONTACT_REFERENCE, max_frames=1)
    assert result.assessment.status is ApplicabilityStatus.APPLICABLE
    measured = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"])
    distances = [column for column in measured if column.startswith("RESIDUE_HEAVY_DISTANCE_angstrom__")]
    contacts = [column for column in measured if column.startswith("RESIDUE_HEAVY_CONTACT__")]
    assert distances and len(distances) == len(contacts)
    assert (measured[distances].to_numpy() >= 0).all()
    assert set(measured[contacts].iloc[0]).issubset({0.0, 1.0})
    assert result.provenance.parameters["distance_convention"] == "closest_heavy_atom"


def test_pbc_features_reject_trajectories_without_a_periodic_cell():
    plugin = get_plugin("protein.ca_distances_contacts", {"use_pbc": True, "max_pairs": 20})
    with pytest.raises(ValueError, match="periodic-cell"):
        plugin.extract(CONTACT_REFERENCE, CONTACT_REFERENCE, max_frames=1)


def test_pbc_ca_distance_uses_the_declared_trajectory_cell(tmp_path):
    path = tmp_path / "periodic.pdb"
    path.write_text(
        "CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1\n"
        "ATOM      1  CA  ALA A   1       0.500   0.000   0.000  1.00  0.00           C\n"
        "ATOM      2  CA  ALA A   4       9.500   0.000   0.000  1.00  0.00           C\nEND\n"
    )
    plugin = get_plugin("protein.ca_distances_contacts", {
        "residue_pairs": [[1, 4]], "max_pairs": 1, "include_contacts": False, "use_pbc": True,
    })
    result = plugin.extract(path, path)
    value = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"]).iloc[0, 0]
    assert value == pytest.approx(1.0)


def test_selected_coordinate_plugin_exports_named_xyz_as_stored():
    plugin = get_plugin("coordinates.selected_atoms", {"selection": "protein and resid 1", "max_atoms": 20})
    result = plugin.extract(CONTACT_REFERENCE, CONTACT_REFERENCE, max_frames=1)
    measured = result.frame_features.drop(columns=["frame_index", "frame_order", "time_ps"])
    assert len(measured.columns) == 15  # five atoms in ALA 1, each with x/y/z
    assert all(column.startswith(("COORD_x_angstrom__", "COORD_y_angstrom__", "COORD_z_angstrom__")) for column in measured)
    assert result.provenance.parameters["coordinate_policy"] == "as_stored_no_alignment_or_pbc_transform"

