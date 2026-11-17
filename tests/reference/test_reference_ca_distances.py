"""Reference validation tests for MolScope protein C-alpha distances and contacts.

Validates MolScope's ProteinCADistanceContactPlugin against MDAnalysis:
- Pairwise C-alpha Euclidean distances vs MDAnalysis distance_array / calc_bonds.
- Binary contact classification vs strict threshold inequality (d <= cutoff).
- Trajectory contact occupancy vs frame-averaged presence.
- Explicit declared residue-pair selection vs targeted MDAnalysis distance calculation.
- Minimum sequence-separation filter vs same-chain residue indexing.
- Periodic boundary conditions (minimum image convention) vs MDAnalysis calc_bonds with box.
"""

from pathlib import Path
import tempfile
import unittest

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds, distance_array
import numpy as np

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.protein.plugins import ProteinCADistanceContactPlugin


FIXTURES = Path(__file__).parents[1] / "fixtures"
PDB_1A22 = FIXTURES / "1A22.pdb"


class ReferenceCADistanceValidationTests(unittest.TestCase):
    """Validation test suite for C-alpha distances and contacts vs MDAnalysis."""

    def test_ca_pairwise_distances_match_mdanalysis_euclidean(self):
        """Verify all extracted C-alpha distances match direct MDAnalysis atom positions."""
        plugin = ProteinCADistanceContactPlugin(
            selection="resid 10:25",
            contact_cutoff_angstrom=8.0,
            minimum_sequence_separation=3,
            max_pairs=200,
            include_contacts=True,
            use_pbc=False,
        )
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        ca_atoms = universe.select_atoms("protein and name CA and (resid 10:25)")
        by_resid = {atom.resid: atom for atom in ca_atoms}

        # Validate every extracted pair against direct Euclidean distance from MDA
        distance_cols = [col for col in result.frame_features.columns if col.startswith("CA_DISTANCE_angstrom__")]
        self.assertGreater(len(distance_cols), 10, "Should extract a substantial number of C-alpha pairs")

        for col in distance_cols:
            pair_id = col.removeprefix("CA_DISTANCE_angstrom__")
            reported_dist = result.frame_features.loc[0, col]

            # Parse pair_id format: e.g. A__PHE_10__A__PHE_25
            parts = pair_id.split("__")
            self.assertEqual(len(parts), 4, f"Unexpected pair ID structure: {pair_id}")
            chain1, res_info1, chain2, res_info2 = parts
            resid1 = int(res_info1.split("_")[-1])
            resid2 = int(res_info2.split("_")[-1])

            atom1 = by_resid[resid1]
            atom2 = by_resid[resid2]

            expected_dist = float(np.linalg.norm(atom1.position - atom2.position))
            self.assertAlmostEqual(
                reported_dist,
                expected_dist,
                places=5,
                msg=f"Discrepancy in pair {pair_id}: reported={reported_dist}, expected={expected_dist}",
            )

    def test_ca_contacts_match_cutoff_threshold(self):
        """Verify binary contact flags exactly match d <= cutoff and occupancy is exact."""
        cutoff = 8.0
        plugin = ProteinCADistanceContactPlugin(
            selection="resid 10:25",
            contact_cutoff_angstrom=cutoff,
            minimum_sequence_separation=3,
            max_pairs=200,
            include_contacts=True,
        )
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)

        contact_cols = [col for col in result.frame_features.columns if col.startswith("CA_CONTACT__")]
        self.assertGreater(len(contact_cols), 10)

        for col in contact_cols:
            pair_id = col.removeprefix("CA_CONTACT__")
            dist_col = f"CA_DISTANCE_angstrom__{pair_id}"
            dist = result.frame_features.loc[0, dist_col]
            contact = result.frame_features.loc[0, col]

            expected_contact = 1.0 if dist <= cutoff else 0.0
            self.assertEqual(contact, expected_contact, f"Contact mismatch for {pair_id} at dist={dist}")

            # Verify trajectory occupancy
            occ_col = f"CA_CONTACT_occupancy__{pair_id}"
            reported_occ = result.trajectory_features.loc[0, occ_col]
            self.assertEqual(reported_occ, contact)

    def test_ca_explicit_declared_pairs_match_reference(self):
        """Verify explicitly declared residue pairs match direct MDAnalysis calc_bonds."""
        declared_pairs = [[10, 25], [12, 30], [15, 50]]
        plugin = ProteinCADistanceContactPlugin(
            residue_pairs=declared_pairs,
            contact_cutoff_angstrom=8.0,
            include_contacts=True,
            use_pbc=False,
        )
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_1A22))
        for r1, r2 in declared_pairs:
            atom1 = universe.select_atoms(f"protein and name CA and resid {r1}")[0]
            atom2 = universe.select_atoms(f"protein and name CA and resid {r2}")[0]
            ref_dist = float(calc_bonds(atom1.position[np.newaxis, :], atom2.position[np.newaxis, :])[0])

            col_match = [
                c for c in result.frame_features.columns
                if c.startswith("CA_DISTANCE_angstrom__") and f"_{r1}__" in c and f"_{r2}" in c
            ]
            self.assertEqual(len(col_match), 1, f"Expected exactly one column for pair {r1}-{r2}")
            reported_dist = result.frame_features.loc[0, col_match[0]]
            self.assertAlmostEqual(reported_dist, ref_dist, places=5)

    def test_ca_minimum_sequence_separation_enforced(self):
        """Verify pairs with separation < minimum_sequence_separation within the same chain are omitted."""
        plugin = ProteinCADistanceContactPlugin(
            selection="resid 10:20",
            minimum_sequence_separation=4,
            max_pairs=500,
        )
        result = plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        distance_cols = [col for col in result.frame_features.columns if col.startswith("CA_DISTANCE_angstrom__")]
        for col in distance_cols:
            pair_id = col.removeprefix("CA_DISTANCE_angstrom__")
            parts = pair_id.split("__")
            chain1, res1, chain2, res2 = parts
            if chain1 == chain2:
                resid1 = int(res1.split("_")[-1])
                resid2 = int(res2.split("_")[-1])
                self.assertGreaterEqual(
                    abs(resid1 - resid2), 4,
                    f"Pair {pair_id} violated minimum_sequence_separation=4",
                )

    def test_ca_pbc_minimum_image_distance_reference(self):
        """Verify PBC-aware C-alpha distances match MDAnalysis minimum image convention."""
        # Create a test PDB with explicit CRYST1 box
        box_pdb = """CRYST1   80.000   80.000   80.000  90.00  90.00  90.00 P 1           1
ATOM      1  N   ALA A   1       2.000   2.000   2.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.500   2.500   2.500  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.500   2.500  1.00  0.00           C
ATOM      4  O   ALA A   1       3.500   2.500   2.500  1.00  0.00           O
ATOM      5  N   GLY A   5      78.000  78.000  78.000  1.00  0.00           N
ATOM      6  CA  GLY A   5      78.500  78.500  78.500  1.00  0.00           C
ATOM      7  C   GLY A   5      79.000  78.500  78.500  1.00  0.00           C
ATOM      8  O   GLY A   5      79.500  78.500  78.500  1.00  0.00           O
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(box_pdb)
            tmp_path = f.name

        try:
            universe = mda.Universe(tmp_path)
            box = universe.dimensions
            ca_atoms = universe.select_atoms("name CA")
            expected_pbc_dist = float(calc_bonds(
                ca_atoms[0].position[np.newaxis, :],
                ca_atoms[1].position[np.newaxis, :],
                box=box,
            )[0])
            expected_nopbc_dist = float(np.linalg.norm(ca_atoms[0].position - ca_atoms[1].position))

            # Non-PBC distance across 76 Å span in 80 Å box
            self.assertGreater(expected_nopbc_dist, 70.0)
            # Minimum-image distance across PBC boundary is ~6.928 Å
            self.assertLess(expected_pbc_dist, 10.0)

            plugin_pbc = ProteinCADistanceContactPlugin(use_pbc=True, minimum_sequence_separation=1)
            res_pbc = plugin_pbc.extract(tmp_path, tmp_path, max_frames=1)
            self.assertEqual(res_pbc.assessment.status, ApplicabilityStatus.APPLICABLE)

            pbc_dist_col = [c for c in res_pbc.frame_features.columns if c.startswith("CA_DISTANCE_angstrom__")][0]
            reported_pbc_dist = res_pbc.frame_features.loc[0, pbc_dist_col]
            self.assertAlmostEqual(reported_pbc_dist, expected_pbc_dist, places=5)

            # Check without PBC
            plugin_nopbc = ProteinCADistanceContactPlugin(use_pbc=False, minimum_sequence_separation=1)
            res_nopbc = plugin_nopbc.extract(tmp_path, tmp_path, max_frames=1)
            nopbc_dist_col = [c for c in res_nopbc.frame_features.columns if c.startswith("CA_DISTANCE_angstrom__")][0]
            reported_nopbc_dist = res_nopbc.frame_features.loc[0, nopbc_dist_col]
            self.assertAlmostEqual(reported_nopbc_dist, expected_nopbc_dist, places=5)
        finally:
            Path(tmp_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
