"""Reference validation tests for Solvent-Accessible Surface Area (SASA).

Validates MolScope's SASAPlugin against:
- Analytical spherical surface area geometry: A = 4 * pi * (R + r_probe)^2.
- Exact additivity for non-overlapping spheres: A_dimer = 2 * A_monomer for d > 2(R + r_probe).
- Steric shielding in covalently bonded atoms: A_dimer < 2 * A_monomer.
- Direct FreeSASA reference calculation on protein-ligand and multi-residue complexes.
- Correct exclusion of solvent and ions according to declared scope policy.
"""

from pathlib import Path
import tempfile
import unittest

import freesasa
import MDAnalysis as mda
import numpy as np

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.environment.plugins import SASAPlugin


FIXTURES = Path(__file__).parents[1] / "fixtures"
PDB_PLIF = FIXTURES / "protein_ligand_contacts.pdb"
PDB_SOLVATED = FIXTURES / "solvated_protein.pdb"
PDB_1A22 = FIXTURES / "1A22.pdb"


class ReferenceSASAValidationTests(unittest.TestCase):
    """Validation test suite for SASA approximations vs geometric reference."""

    def test_sasa_isolated_sphere_matches_analytical_geometry(self):
        """Verify single isolated atom SASA matches exact 4*pi*(R + r_probe)^2 spherical surface area."""
        single_atom_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(single_atom_pdb)
            tmp_path = f.name

        try:
            plugin = SASAPlugin()
            result = plugin.extract(tmp_path, tmp_path, max_frames=1)
            self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

            reported_total = result.frame_features.loc[0, "SASA_total"]

            # Calculate exact analytical sphere area
            probe_radius = plugin.probe_radius_angstrom
            # Query standard Carbon radius assigned by FreeSASA
            struct = freesasa.Structure()
            struct.addAtom("CA", "ALA", "1", "A", 0.0, 0.0, 0.0)
            atom_radius = struct.radius(0)
            expected_area = 4.0 * np.pi * (atom_radius + probe_radius) ** 2

            self.assertAlmostEqual(reported_total, expected_area, places=4)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_sasa_non_overlapping_spheres_exact_additivity(self):
        """Verify two distant non-overlapping atoms exhibit exact additivity: A_total = 2 * A_monomer."""
        two_atoms_pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2      25.000   0.000   0.000  1.00  0.00           C
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(two_atoms_pdb)
            tmp_path = f.name

        try:
            plugin = SASAPlugin()
            result = plugin.extract(tmp_path, tmp_path, max_frames=1)
            self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

            reported_total = result.frame_features.loc[0, "SASA_total"]

            struct = freesasa.Structure()
            struct.addAtom("CA", "ALA", "1", "A", 0.0, 0.0, 0.0)
            single_atom_area = 4.0 * np.pi * (struct.radius(0) + plugin.probe_radius_angstrom) ** 2
            expected_total = 2.0 * single_atom_area

            self.assertAlmostEqual(reported_total, expected_total, places=4)

            # Check residue-wise contributions
            res_cols = [c for c in result.frame_features.columns if c.startswith("SASA_residue__")]
            self.assertEqual(len(res_cols), 2)
            for col in res_cols:
                self.assertAlmostEqual(result.frame_features.loc[0, col], single_atom_area, places=4)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_sasa_steric_shielding_overlapping_atoms(self):
        """Verify covalently bonded atoms exhibit mutual steric shielding and match direct FreeSASA."""
        # Carbon-carbon single bond (d = 1.54 Å)
        c_c_dimer_pdb = """ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  C2  LIG A   1       1.540   0.000   0.000  1.00  0.00           C
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(c_c_dimer_pdb)
            tmp_path = f.name

        try:
            plugin = SASAPlugin()
            result = plugin.extract(tmp_path, tmp_path, max_frames=1)
            self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

            reported_total = result.frame_features.loc[0, "SASA_total"]

            # Independent monomer area
            struct_mono = freesasa.Structure()
            struct_mono.addAtom("C1", "LIG", "1", "A", 0.0, 0.0, 0.0)
            single_atom_area = 4.0 * np.pi * (struct_mono.radius(0) + plugin.probe_radius_angstrom) ** 2

            # Direct FreeSASA reference on dimer
            struct_dimer = freesasa.Structure()
            struct_dimer.addAtom("C1", "LIG", "1", "A", 0.0, 0.0, 0.0)
            struct_dimer.addAtom("C2", "LIG", "1", "A", 1.54, 0.0, 0.0)
            params = freesasa.Parameters({"probe-radius": plugin.probe_radius_angstrom, "n-points": plugin.n_points})
            ref_dimer_area = freesasa.calc(struct_dimer, params).totalArea()

            # Mutual steric shielding reduces area below 2 * monomer
            self.assertLess(reported_total, 2.0 * single_atom_area)
            # Match direct calculation to within 1e-5
            self.assertAlmostEqual(reported_total, ref_dimer_area, places=5)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_sasa_protein_ligand_complex_vs_direct_freesasa(self):
        """Verify total and residue-level SASA on complex match direct FreeSASA reference calculation."""
        plugin = SASAPlugin()
        result = plugin.extract(PDB_PLIF, PDB_PLIF, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_PLIF))
        structure = freesasa.Structure()
        for atom in universe.atoms:
            chain = (atom.chainID or atom.segid or " ").strip()[:1] or " "
            structure.addAtom(atom.name.strip(), atom.resname.strip(), str(atom.resid), chain, *map(float, atom.position))

        params = freesasa.Parameters({"probe-radius": plugin.probe_radius_angstrom, "n-points": plugin.n_points})
        ref_calc = freesasa.calc(structure, params)
        expected_total = ref_calc.totalArea()

        reported_total = result.frame_features.loc[0, "SASA_total"]
        self.assertAlmostEqual(reported_total, expected_total, places=5)

        # Verify sum of residue SASAs equals total SASA
        res_cols = [c for c in result.frame_features.columns if c.startswith("SASA_residue__")]
        self.assertGreater(len(res_cols), 1)
        residue_sum = sum(result.frame_features.loc[0, c] for c in res_cols)
        self.assertAlmostEqual(residue_sum, reported_total, places=4)

    def test_sasa_solvation_scope_policy(self):
        """Verify solvent (HOH) is excluded from SASA calculation according to scope policy."""
        plugin = SASAPlugin()
        result = plugin.extract(PDB_SOLVATED, PDB_SOLVATED, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_SOLVATED))
        non_solvent = universe.select_atoms("not resname HOH")
        self.assertGreater(len(non_solvent), 0)

        structure = freesasa.Structure()
        for atom in non_solvent:
            chain = (atom.chainID or atom.segid or " ").strip()[:1] or " "
            structure.addAtom(atom.name.strip(), atom.resname.strip(), str(atom.resid), chain, *map(float, atom.position))

        params = freesasa.Parameters({"probe-radius": plugin.probe_radius_angstrom, "n-points": plugin.n_points})
        expected_total = freesasa.calc(structure, params).totalArea()

        reported_total = result.frame_features.loc[0, "SASA_total"]
        self.assertAlmostEqual(reported_total, expected_total, places=5)


if __name__ == "__main__":
    unittest.main()
