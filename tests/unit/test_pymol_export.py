"""Unit tests for PyMOL and 3D visualization exporter."""

from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

from molscope.reporting.pymol import export_3d_visualization, extract_residue_scores

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def test_extract_residue_scores_pairs_and_sidechains():
    df = pd.DataFrame([
        {"Feature": "CA_DISTANCE_angstrom__A__ALA_1__A__PHE_16", "PCA_Salience": 0.85},
        {"Feature": "SIDECHAIN_chi1__A__PHE_16", "PCA_Salience": 0.95},
        {"Feature": "GENERIC_rg", "PCA_Salience": 0.50},  # unmapped global feature
    ])
    scores = extract_residue_scores(df, feature_column="Feature", score_column="PCA_Salience")
    assert 1 in scores
    assert scores[1] == pytest.approx(0.85)
    assert 16 in scores
    assert scores[16] == pytest.approx(0.95)  # took max


def test_export_3d_visualization(tmp_path):
    residue_scores = {1: 0.2, 16: 0.9, 25: 0.6}
    outputs = export_3d_visualization(
        topology=FIXTURE_PDB,
        output_dir=tmp_path,
        residue_scores=residue_scores,
        basename="test_viz",
    )
    pdb_file = outputs["pdb"]
    pml_file = outputs["pymol_script"]
    cxc_file = outputs["chimerax_script"]

    assert pdb_file.is_file()
    assert pml_file.is_file()
    assert cxc_file.is_file()

    # Verify PDB has normalized B-factors assigned
    u = mda.Universe(str(pdb_file))
    res16_atoms = u.select_atoms("resid 16")
    assert len(res16_atoms) > 0
    # 0.9 / 0.9 = 1.0
    assert np.allclose(res16_atoms.tempfactors, 1.0, atol=1e-3)

    res1_atoms = u.select_atoms("resid 1")
    assert len(res1_atoms) > 0
    # 0.2 / 0.9 ≈ 0.222, formatted to 2 decimal places in PDB
    assert np.allclose(res1_atoms.tempfactors, 0.2 / 0.9, atol=0.01)

    # Verify PML contains expected commands
    pml_text = pml_file.read_text(encoding="utf-8")
    assert "reinitialize" in pml_text
    assert "spectrum b, blue_white_red" in pml_text
    assert "cartoon putty" in pml_text
    assert "resi 16" in pml_text

    # Verify CXC contains expected commands
    cxc_text = cxc_file.read_text(encoding="utf-8")
    assert "color bfactor palette" in cxc_text
    assert "open test_viz.pdb" in cxc_text
