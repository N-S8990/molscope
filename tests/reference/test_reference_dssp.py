"""Reference validation tests for secondary structure assignment via DSSP.

Validates MolScope's ProteinDSSPPlugin against MDAnalysis native DSSP:
- Framewise aggregate fractions (helix, sheet, coil) vs direct DSSP assignment.
- Per-residue secondary structure assignments vs reference dictionary mapping ('H', 'E', '-').
- Completeness and partition of unity (helix + sheet + coil = 1.0).
- Trajectory mean occupancy agreement.
- Graceful handling of non-protein or incomplete backbone structures.
"""

from pathlib import Path
import unittest

import MDAnalysis as mda
from MDAnalysis.analysis.dssp import DSSP
import numpy as np

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.protein.plugins import ProteinDSSPPlugin


FIXTURES = Path(__file__).parents[1] / "fixtures"
PDB_1A22 = FIXTURES / "1A22.pdb"
PDB_NUCLEIC = FIXTURES / "nucleic_base_centers.pdb"


class ReferenceDSSPValidationTests(unittest.TestCase):
    """Validation test suite for DSSP secondary structure assignment vs reference dictionary."""

    def test_dssp_aggregate_fractions_match_mdanalysis_reference(self):
        """Verify MolScope DSSP aggregate fractions match MDAnalysis native DSSP output."""
        plugin = ProteinDSSPPlugin(per_residue=False)
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        protein = universe.select_atoms("protein")
        ref_dssp = DSSP(protein).run()
        codes = np.asarray(ref_dssp.results.dssp)[0]

        expected_helix = float(np.mean(codes == "H"))
        expected_sheet = float(np.mean(codes == "E"))
        expected_coil = float(np.mean(codes == "-"))

        reported_helix = result.frame_features.loc[0, "DSSP_helix_fraction"]
        reported_sheet = result.frame_features.loc[0, "DSSP_sheet_fraction"]
        reported_coil = result.frame_features.loc[0, "DSSP_coil_fraction"]

        self.assertAlmostEqual(reported_helix, expected_helix, places=6)
        self.assertAlmostEqual(reported_sheet, expected_sheet, places=6)
        self.assertAlmostEqual(reported_coil, expected_coil, places=6)

        # Completeness / partition of unity: fractions must sum to 1.0
        total_fraction = reported_helix + reported_sheet + reported_coil
        self.assertAlmostEqual(total_fraction, 1.0, places=6)

        # Trajectory feature means
        self.assertAlmostEqual(result.trajectory_features.loc[0, "DSSP_helix_fraction_mean"], expected_helix, places=6)
        self.assertAlmostEqual(result.trajectory_features.loc[0, "DSSP_sheet_fraction_mean"], expected_sheet, places=6)
        self.assertAlmostEqual(result.trajectory_features.loc[0, "DSSP_coil_fraction_mean"], expected_coil, places=6)

    def test_dssp_per_residue_assignments_match_reference_dictionary(self):
        """Verify per-residue binary indicators match DSSP code dictionary across all 372 residues."""
        plugin = ProteinDSSPPlugin(per_residue=True)
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        protein = universe.select_atoms("protein")
        ref_dssp = DSSP(protein).run()
        codes = np.asarray(ref_dssp.results.dssp)[0]

        # Canonical reference dictionary mapping DSSP codes to indicator tuples (helix, sheet, coil)
        reference_dict = {
            "H": (1.0, 0.0, 0.0),
            "E": (0.0, 1.0, 0.0),
            "-": (0.0, 0.0, 1.0),
        }

        self.assertEqual(len(protein.residues), len(codes))
        for residue, code in zip(protein.residues, codes, strict=True):
            expected_h, expected_e, expected_c = reference_dict[code]
            ident = f"resindex_{residue.ix}"

            h_col = f"DSSP_helix__{ident}"
            e_col = f"DSSP_sheet__{ident}"
            c_col = f"DSSP_coil__{ident}"

            self.assertIn(h_col, result.frame_features.columns)
            self.assertIn(e_col, result.frame_features.columns)
            self.assertIn(c_col, result.frame_features.columns)

            self.assertEqual(result.frame_features.loc[0, h_col], expected_h)
            self.assertEqual(result.frame_features.loc[0, e_col], expected_e)
            self.assertEqual(result.frame_features.loc[0, c_col], expected_c)

            # Check that exactly one secondary structure state is assigned per residue
            state_sum = (
                result.frame_features.loc[0, h_col]
                + result.frame_features.loc[0, e_col]
                + result.frame_features.loc[0, c_col]
            )
            self.assertEqual(state_sum, 1.0, f"Residue {ident} has invalid secondary structure indicator sum: {state_sum}")

    def test_dssp_canonical_structure_composition(self):
        """Verify 1A22 contains expected known alpha-helix and beta-sheet compositions."""
        plugin = ProteinDSSPPlugin(per_residue=False)
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)

        reported_helix = result.frame_features.loc[0, "DSSP_helix_fraction"]
        reported_sheet = result.frame_features.loc[0, "DSSP_sheet_fraction"]

        # In 1A22 (growth hormone + receptor), helix is ~35% and sheet is ~23%
        self.assertGreater(reported_helix, 0.30, "Growth hormone complex should have significant helical content")
        self.assertGreater(reported_sheet, 0.20, "Receptor extracellular domain should have beta-sheet content")

    def test_dssp_non_protein_system_gracefully_handled(self):
        """Verify non-protein structure returns NOT_APPLICABLE with explicit diagnostic reason."""
        plugin = ProteinDSSPPlugin()
        result = plugin.extract(PDB_NUCLEIC, PDB_NUCLEIC, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertIn("protein", result.assessment.reason.lower())


if __name__ == "__main__":
    unittest.main()
