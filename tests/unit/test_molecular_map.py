"""Evidence-to-structure mapping must never silently change residue identity."""
import numpy as np

from molscope.reporting.molecular_map import _chemical_map, _mapping


def test_feature_mapping_requires_exact_residue_identity():
    positions = {
        151: ("SER", np.array([0.0, 0.0, 0.0])),
        293: ("LEU", np.array([1.0, 0.0, 0.0])),
    }
    assert _mapping("CA_DISTANCE_angstrom__X__SER_151__X__LEU_293", positions) == (151, 293)
    assert _mapping("SIDECHAIN_chi1__chain_X__SER_151__sin", positions) == (151,)
    assert _mapping("SIDECHAIN_residue_SER_151", positions) == (151,)
    assert _mapping("CA_DISTANCE_angstrom__X__ALA_151__X__LEU_293", positions) == ()
    assert _mapping("GEOM_global_radius_gyration", positions) == ()


def test_chemical_mapping_uses_declared_atom_groups(tmp_path):
    import json

    positions = {
        152: ("PHE", np.array([0.0, 0.0, 0.0])),
        292: ("TRP", np.array([1.0, 0.0, 0.0])),
    }
    path = tmp_path / "definitions.json"
    name = "CHEM_POCKET_Phe152_Trp292_normal_abs_cos"
    path.write_text(json.dumps({"features": [{"feature": name, "left_group": "PHE152",
                                                "right_group": "TRP292", "residue_ids": [152, 292]}]}))
    mapping = _chemical_map(path, positions)
    assert _mapping(name, positions, mapping) == (152, 292)
    assert _mapping("CHEM_UNKNOWN", positions, mapping) == ()
    path.write_text(json.dumps({"features": [{"feature": name, "left_group": "ALA152",
                                                "right_group": "TRP292", "residue_ids": [152, 292]}]}))
    import pytest
    with pytest.raises(ValueError, match="does not match"):
        _chemical_map(path, positions)
