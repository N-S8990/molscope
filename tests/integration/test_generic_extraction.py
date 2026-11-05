import tempfile
import unittest
from pathlib import Path

from molscope.generic_features import GenericFeatureConfig, GenericFeatureExtractor


PDB = """MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
TER
ATOM      3  C1  UNK B   1       4.000   0.000   0.000  1.00  0.00           C
ATOM      4  C2  UNK B   1       5.000   0.000   0.000  1.00  0.00           C
TER
ENDMDL
MODEL        2
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
TER
ATOM      3  C1  UNK B   1       3.500   0.000   0.000  1.00  0.00           C
ATOM      4  C2  UNK B   1       4.500   0.000   0.000  1.00  0.00           C
TER
ENDMDL
END
"""

PDB_WITH_WATER = """MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
HETATM    3  O   HOH W   1     100.000   0.000   0.000  1.00  0.00           O
ENDMDL
END
"""

PBC_PDB = """CRYST1   10.000   10.000   10.000  90.00  90.00  90.00 P 1           1
ATOM      1  CA  ALA A   1       0.500   0.000   0.000  1.00  0.00           C
TER
HETATM    2  C1  UNK B   1       9.500   0.000   0.000  1.00  0.00           C
END
"""


class GenericFeatureTests(unittest.TestCase):
    def test_generic_extractor_keeps_frame_and_trajectory_outputs_separate(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "two_components.pdb"
            path.write_text(PDB, encoding="utf-8")
            result = GenericFeatureExtractor(GenericFeatureConfig(contact_cutoff_angstrom=4.5)).extract(path, path)
            self.assertEqual(len(result.frame_features), 2)
            self.assertIn("GEOM_global_radius_gyration", result.frame_features.columns)
            self.assertIn("DYN_kabsch_rmsd_to_first_frame", result.frame_features.columns)
            self.assertTrue(any(column.startswith("GEOM_component_min_atom_distance") for column in result.frame_features.columns))
            self.assertTrue(any(column.startswith("CONTACT_pair_count") for column in result.frame_features.columns))
            self.assertEqual(len(result.trajectory_features), 1)
            self.assertIn("DYN_global_rmsf_mean", result.trajectory_features.columns)
            self.assertIn("DYN_radius_gyration_lag1_autocorrelation", result.trajectory_features.columns)
            self.assertTrue(any(column.startswith("DYN_residue_rmsf_mean") for column in result.trajectory_features.columns))
            self.assertTrue(any(column.startswith("CONTACT_occupancy") for column in result.trajectory_features.columns))
            self.assertNotIn("DYN_global_rmsf_mean", result.frame_features.columns)
            self.assertTrue(any(item.feature_id == "interactions.generic_contacts" and item.status.value == "applicable" for item in result.assessments))

    def test_global_features_exclude_environment_by_default(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "water_scope.pdb"
            path.write_text(PDB_WITH_WATER, encoding="utf-8")
            molecular = GenericFeatureExtractor().extract(path, path)
            environment = GenericFeatureExtractor(
                GenericFeatureConfig(include_environment_in_global_features=True)
            ).extract(path, path)
        self.assertLess(
            molecular.frame_features.loc[0, "GEOM_global_radius_gyration"],
            environment.frame_features.loc[0, "GEOM_global_radius_gyration"],
        )

    def test_pbc_geometry_uses_minimum_image_and_requires_a_cell(self):
        with tempfile.TemporaryDirectory() as temporary:
            pbc_path = Path(temporary) / "pbc.pdb"
            pbc_path.write_text(PBC_PDB, encoding="utf-8")
            result = GenericFeatureExtractor(GenericFeatureConfig(use_pbc=True)).extract(pbc_path, pbc_path)
            distance_column = next(column for column in result.frame_features if column.startswith("GEOM_component_com_distance"))
            self.assertAlmostEqual(result.frame_features.loc[0, distance_column], 1.0, places=5)
            no_cell_path = Path(temporary) / "no_cell.pdb"
            no_cell_path.write_text(PDB, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not provide a positive periodic cell"):
                GenericFeatureExtractor(GenericFeatureConfig(use_pbc=True)).extract(no_cell_path, no_cell_path)


if __name__ == "__main__":
    unittest.main()
