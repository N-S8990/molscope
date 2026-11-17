import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import MDAnalysis as mda

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

    def test_unreadable_later_frame_returns_invalid_report(self):
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "trajectory.pdb"
            trajectory.write_text(TRAJECTORY, encoding="utf-8")
            topology_universe = mda.Universe(str(trajectory))
            coordinate_universe = mda.Universe(str(trajectory), str(trajectory))

            class FailingReader:
                n_atoms = 3
                dt = 1.0

                def __len__(self):
                    return 2

                def __getitem__(self, frame):
                    if frame == 1:
                        raise OSError("unreadable frame")
                    return coordinate_universe.trajectory[frame]

            fake_universe = SimpleNamespace(atoms=coordinate_universe.atoms,
                                            trajectory=FailingReader())
            with patch("molscope.io.validation.mda.Universe",
                       side_effect=[topology_universe, fake_universe]):
                report = inspect_md_inputs(trajectory, trajectory)
        self.assertEqual(report.status, InputIntegrityStatus.INVALID)
        self.assertEqual(report.trajectory.frames_checked, 1)
        self.assertFalse(report.trajectory.complete_scan)
        self.assertIn("frame 1", " ".join(report.errors))

    def test_wrapped_coordinates_emit_loud_imaging_fix_instructions(self):
        wrapped_pdb = (
            "CRYST1   60.000   60.000   60.000  90.00  90.00  90.00 P 1           1\n"
            "ATOM      1  CA  ALA A   1       1.000   1.000   1.000  1.00  0.00           C\n"
            "ATOM      2  C   ALA A   1      59.500   1.000   1.000  1.00  0.00           C\n"
            "CONECT    1    2\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            trajectory = Path(temporary) / "wrapped.pdb"
            trajectory.write_text(wrapped_pdb, encoding="utf-8")
            # Default: valid with loud warnings
            report_warn = inspect_md_inputs(trajectory, trajectory, require_imaged=False)
            self.assertEqual(report_warn.status, InputIntegrityStatus.VALID_WITH_WARNINGS)
            self.assertTrue(report_warn.trajectory.wrapped_coordinates_detected)
            warn_str = " ".join(report_warn.warnings)
            self.assertIn("apply_imaging_transformations", warn_str)
            self.assertIn("gmx trjconv", warn_str)
            self.assertIn("mdconvert", warn_str)

            # Strict: invalid with fatal error
            report_err = inspect_md_inputs(trajectory, trajectory, require_imaged=True)
            self.assertEqual(report_err.status, InputIntegrityStatus.INVALID)
            err_str = " ".join(report_err.errors)
            self.assertIn("PBC", err_str)
            self.assertIn("apply_imaging_transformations", err_str)


if __name__ == "__main__":
    unittest.main()
