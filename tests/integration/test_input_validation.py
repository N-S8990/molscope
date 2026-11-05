import tempfile
import unittest
from pathlib import Path

from molscope.ingestion import InputIntegrityStatus, inspect_md_inputs, topology_identity


TRAJECTORY = """MODEL        1
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
TER
ATOM      3  C1  UNK B   1       4.000   0.000   0.000  1.00  0.00           C
ENDMDL
MODEL        2
ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
TER
ATOM      3  C1  UNK B   1       3.500   0.000   0.000  1.00  0.00           C
ENDMDL
END
"""

MISMATCHED_TOPOLOGY = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
END
"""


class IngestionTests(unittest.TestCase):
    def test_identity_preserves_source_atom_and_residue_mapping(self):
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "trajectory.pdb"
            trajectory.write_text(TRAJECTORY, encoding="utf-8")
            identity = topology_identity(trajectory)
        self.assertEqual(identity.atom_count, 3)
        self.assertEqual(identity.residue_count, 2)
        self.assertEqual(identity.atoms[2].residue_index, identity.residues[1].index)
        self.assertEqual(identity.residues[0].chain_id, "A")

    def test_complete_integrity_scan_reports_finite_compatible_trajectory(self):
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "trajectory.pdb"
            trajectory.write_text(TRAJECTORY, encoding="utf-8")
            report = inspect_md_inputs(trajectory, trajectory)
        self.assertEqual(report.status, InputIntegrityStatus.VALID)
        self.assertEqual(report.trajectory.frame_count, 2)
        self.assertTrue(report.trajectory.complete_scan)
        self.assertTrue(report.trajectory.finite_coordinates)
        self.assertEqual(report.trajectory.trajectory_atom_count, 3)

    def test_incompatible_topology_and_trajectory_are_invalid(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            trajectory = directory / "trajectory.pdb"
            topology = directory / "topology.pdb"
            trajectory.write_text(TRAJECTORY, encoding="utf-8")
            topology.write_text(MISMATCHED_TOPOLOGY, encoding="utf-8")
            report = inspect_md_inputs(topology, trajectory)
        self.assertEqual(report.status, InputIntegrityStatus.INVALID)
        self.assertTrue(report.errors)

    def test_limited_scan_is_explicitly_labeled(self):
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "trajectory.pdb"
            trajectory.write_text(TRAJECTORY, encoding="utf-8")
            report = inspect_md_inputs(trajectory, trajectory, max_frames=1)
        self.assertFalse(report.trajectory.complete_scan)
        self.assertIn("limited frame subset", " ".join(report.warnings))


if __name__ == "__main__":
    unittest.main()
