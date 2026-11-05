"""Release regressions for bounded extraction and configured backends."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import MDAnalysis as mda
import numpy as np
import pytest

from molscope.plugins import ProteinDSSPPlugin, get_plugin


REFERENCE_PDB = str(Path(__file__).parents[1] / "fixtures" / "1A22.pdb")

def test_pair_budget_stops_large_feature_matrix_before_distance_allocation():
    plugin = get_plugin("protein.ca_distances_contacts", {"max_pairs": 2})
    with patch("molscope.features.protein.plugins.calc_bonds") as distances:
        with pytest.raises(ValueError, match="max_pairs"):
            plugin.extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        distances.assert_not_called()


def test_torsion_configuration_filters_measurements_and_empty_scope_is_explicit():
    plugin = get_plugin("protein.backbone_torsions", {"selection": "chainID A and resid 2:4", "torsions": ["phi"]})
    result = plugin.extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
    columns = result.frame_features.filter(regex="^TORSION__").columns
    assert len(columns) == 6
    assert all("__phi__" in column for column in columns)
    empty = get_plugin("protein.backbone_torsions", {"selection": "resid 99999"}).extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
    assert empty.assessment.status.value == "not_applicable"
    assert empty.frame_features.empty


def test_selection_and_cutoff_are_local_to_configured_instance():
    plugin = get_plugin("protein.ca_distances_contacts", {
        "selection": "chainID A and resid 1:4", "max_pairs": 1,
        "contact_cutoff_angstrom": 0.01,
    })
    result = plugin.extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
    contacts = result.frame_features.filter(regex="^CA_CONTACT__")
    assert contacts.shape == (1, 1)
    assert contacts.iloc[0, 0] == 0
    assert get_plugin("protein.ca_distances_contacts").contact_cutoff_angstrom == 8
    assert result.provenance.parameters["max_pairs"] == 1


def test_dssp_backend_reads_only_requested_frames_and_preserves_residue_outputs(tmp_path):
    universe = mda.Universe(REFERENCE_PDB)
    trajectory = tmp_path / "sample.dcd"
    with mda.Writer(str(trajectory), n_atoms=len(universe.atoms)) as writer:
        for _ in range(5):
            writer.write(universe.atoms)
    residues = len(universe.select_atoms("protein").residues)
    codes = np.full((2, residues), "H")
    codes[1, :] = "E"
    with patch("MDAnalysis.analysis.dssp.DSSP") as backend:
        backend.return_value.run.return_value = SimpleNamespace(results=SimpleNamespace(dssp=codes))
        result = ProteinDSSPPlugin(per_residue=True).extract(REFERENCE_PDB, trajectory, frame_stride=2, max_frames=2)
    backend.return_value.run.assert_called_once_with(frames=[0, 2])
    assert result.frame_features.frame_index.tolist() == [0, 2]
    assert result.trajectory_features["DSSP_helix__resindex_0__occupancy"].iloc[0] == 0.5
    assert len(result.provenance.parameters["residue_mapping"]) == residues


@pytest.mark.parametrize("parameters", [{"max_pairs": 0}, {"contact_cutoff_angstrom": float("nan")}, {"bogus": 1}])
def test_invalid_plugin_parameters_rejected(parameters):
    with pytest.raises(ValueError):
        get_plugin("protein.ca_distances_contacts", parameters)
