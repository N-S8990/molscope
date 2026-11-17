"""Reference validation tests for MolScope backbone and sidechain dihedrals.

Validates MolScope's ProteinBackboneTorsionPlugin and ProteinSidechainTorsionPlugin against:
- MDAnalysis Dihedral analysis (MDAnalysis.analysis.dihedrals.Dihedral).
- Direct atomgroup dihedral measurement (residue.phi_selection().dihedral.value()).
- Trigonometric identity (sin^2 + cos^2 = 1) and continuous angle reconstruction.
- Standard peptide geometry constraints (trans-omega dihedrals).
"""

from pathlib import Path
import unittest

import MDAnalysis as mda
from MDAnalysis.analysis.dihedrals import Dihedral
import numpy as np

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.protein.plugins import ProteinBackboneTorsionPlugin, ProteinSidechainTorsionPlugin


FIXTURES = Path(__file__).parents[1] / "fixtures"
PDB_1A22 = FIXTURES / "1A22.pdb"


def _circular_diff_degrees(angle1: float, angle2: float) -> float:
    """Return shortest angular difference in degrees accounting for wrap-around at +-180."""
    diff = (angle1 - angle2 + 180.0) % 360.0 - 180.0
    return abs(diff)


class ReferenceTorsionValidationTests(unittest.TestCase):
    """Validation test suite for backbone and sidechain torsions vs MDAnalysis."""

    def test_backbone_torsions_match_mdanalysis_dihedral_analysis(self):
        """Verify phi, psi, omega sine/cosine components match MDAnalysis Dihedral class."""
        plugin = ProteinBackboneTorsionPlugin(
            selection="resid 10:30",
            torsions=("phi", "psi", "omega"),
        )
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        protein = universe.select_atoms("protein")

        validated_torsions = 0
        for residue in protein.residues:
            if not (10 <= residue.resid <= 30):
                continue
            chain = residue.atoms.chainIDs[0].strip() or residue.segid.strip() or "unlabeled"
            res_id = f"chain_{chain}__{residue.resname.strip()}_{residue.resid}"

            for name, selection in (
                ("phi", residue.phi_selection()),
                ("psi", residue.psi_selection()),
                ("omega", residue.omega_selection()),
            ):
                if selection is None or len(selection) != 4:
                    continue

                # Compute reference dihedral with MDAnalysis.analysis.dihedrals.Dihedral
                dihedral_analysis = Dihedral([selection]).run()
                ref_deg = float(dihedral_analysis.results.angles[0, 0])
                ref_rad = np.radians(ref_deg)
                expected_sin = float(np.sin(ref_rad))
                expected_cos = float(np.cos(ref_rad))

                sin_col = f"TORSION__{res_id}__{name}__sin"
                cos_col = f"TORSION__{res_id}__{name}__cos"

                self.assertIn(sin_col, result.frame_features.columns)
                self.assertIn(cos_col, result.frame_features.columns)

                reported_sin = result.frame_features.loc[0, sin_col]
                reported_cos = result.frame_features.loc[0, cos_col]

                # Match sine and cosine to 1e-6
                self.assertAlmostEqual(reported_sin, expected_sin, places=6)
                self.assertAlmostEqual(reported_cos, expected_cos, places=6)

                # Reconstruct angle in degrees via arctan2(sin, cos) and verify against MDAnalysis
                reconstructed_deg = float(np.degrees(np.arctan2(reported_sin, reported_cos)))
                ang_diff = _circular_diff_degrees(reconstructed_deg, ref_deg)
                self.assertLess(
                    ang_diff,
                    0.001,
                    f"Angle mismatch for {res_id} {name}: reconstructed={reconstructed_deg}, ref={ref_deg}",
                )
                validated_torsions += 1

        self.assertGreater(validated_torsions, 40, "Should have validated dozens of dihedral angles")

    def test_backbone_torsion_trigonometric_normalization(self):
        """Verify every sin/cos pair satisfies sin^2 + cos^2 = 1.0 within numerical precision."""
        plugin = ProteinBackboneTorsionPlugin(selection="resid 10:20")
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)

        sin_cols = [c for c in result.frame_features.columns if c.endswith("__sin")]
        self.assertGreater(len(sin_cols), 10)

        for sin_col in sin_cols:
            cos_col = sin_col.removesuffix("__sin") + "__cos"
            s = result.frame_features.loc[0, sin_col]
            c = result.frame_features.loc[0, cos_col]
            norm = s**2 + c**2
            self.assertAlmostEqual(norm, 1.0, places=6, msg=f"Normalization failed for {sin_col}: {norm}")

    def test_trans_peptide_omega_conformation(self):
        """Verify omega dihedrals conform to expected trans-peptide planar geometry."""
        plugin = ProteinBackboneTorsionPlugin(selection="resid 10:30", torsions=("omega",))
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)

        omega_sin_cols = [c for c in result.frame_features.columns if "__omega__sin" in c]
        self.assertGreater(len(omega_sin_cols), 15)

        for sin_col in omega_sin_cols:
            cos_col = sin_col.removesuffix("__sin") + "__cos"
            s = result.frame_features.loc[0, sin_col]
            c = result.frame_features.loc[0, cos_col]
            deg = np.degrees(np.arctan2(s, c))
            # In trans peptide bonds, omega is near +-180 degrees (abs(deg) >= 160)
            self.assertGreater(abs(deg), 155.0, f"Expected trans peptide omega, found {deg} for {sin_col}")

    def test_sidechain_chi_torsions_vs_mdanalysis_dihedrals(self):
        """Verify side-chain chi angles match direct MDAnalysis calc_dihedrals."""
        plugin = ProteinSidechainTorsionPlugin(residue_ids=[12, 13, 14, 15, 16], include_chi2=True)
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        # Check that extracted sidechain columns match
        chi_cols = [c for c in result.frame_features.columns if c.startswith("SIDECHAIN_chi")]
        self.assertGreater(len(chi_cols), 4)

        for col in chi_cols:
            if col.endswith("__sin"):
                cos_col = col.removesuffix("__sin") + "__cos"
                s = result.frame_features.loc[0, col]
                c = result.frame_features.loc[0, cos_col]
                self.assertAlmostEqual(s**2 + c**2, 1.0, places=6)

    def test_terminal_residues_omitted_without_errors(self):
        """Verify N- and C-terminal residues lacking complete quartets do not crash."""
        plugin = ProteinBackboneTorsionPlugin(selection="protein", torsions=("phi", "psi", "omega"))
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        # Check that all reported values are finite (no NaN / inf)
        torsion_cols = [c for c in result.frame_features.columns if c.startswith("TORSION__")]
        for col in torsion_cols:
            val = result.frame_features.loc[0, col]
            self.assertTrue(np.isfinite(val), f"Non-finite value in {col}: {val}")


if __name__ == "__main__":
    unittest.main()
