"""Unit tests for physical correctness, PBC wrapping checks, and transparent feature propagation."""

import tempfile
from pathlib import Path
import unittest

import MDAnalysis as mda
import numpy as np
import pandas as pd

from molscope.analysis.discovery.states import rank_state_explanatory_features
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.quality.assessment import PrioritizationConfig, assess_feature_quality
from molscope.io.validation import (
    InputIntegrityStatus,
    detect_pbc_wrapping_artifacts,
    inspect_md_inputs,
)
from molscope.profiling.system import profile_system


def make_synthetic_universe(
    *,
    wrapped: bool = False,
    n_atoms: int = 2,
    box_size: float = 50.0,
    has_bonds: bool = True,
) -> mda.Universe:
    """Create an in-memory MDAnalysis Universe for PBC testing."""
    u = mda.Universe.empty(n_atoms, n_residues=1, atom_resindex=[0] * n_atoms, trajectory=True)
    u.add_TopologyAttr("name", ["CA", "C"][:n_atoms] if n_atoms <= 2 else [f"A{i}" for i in range(n_atoms)])
    u.add_TopologyAttr("resname", ["ALA"] * 1)
    u.add_TopologyAttr("resid", [1])
    u.add_TopologyAttr("chainID", ["A"] * n_atoms)
    u.add_TopologyAttr("segid", ["SEG"])
    if has_bonds and n_atoms >= 2:
        u.add_bonds([(0, 1)])
    u.dimensions = np.array([box_size, box_size, box_size, 90.0, 90.0, 90.0])
    if wrapped:
        # Atom 0 at (1.0, 1.0, 1.0), Atom 1 wrapped across x-boundary to (box_size - 0.5, 1.0, 1.0)
        # Raw distance = box_size - 1.5 = 48.5 A; PBC distance = 1.5 A
        positions = np.zeros((n_atoms, 3))
        positions[0] = [1.0, 1.0, 1.0]
        positions[1] = [box_size - 0.5, 1.0, 1.0]
        u.atoms.positions = positions
    else:
        positions = np.zeros((n_atoms, 3))
        positions[0] = [1.0, 1.0, 1.0]
        if n_atoms > 1:
            positions[1] = [2.5, 1.0, 1.0]
        u.atoms.positions = positions
    return u


def make_pdb_file(path: Path, *, wrapped: bool = False, box_size: float = 50.0) -> None:
    """Write a minimal PDB file with CRYST1 and coordinates."""
    pos1 = (1.0, 1.0, 1.0)
    pos2 = (box_size - 0.5, 1.0, 1.0) if wrapped else (2.5, 1.0, 1.0)
    content = (
        f"CRYST1{box_size:9.3f}{box_size:9.3f}{box_size:9.3f} 90.00  90.00  90.00 P 1           1\n"
        f"ATOM      1  CA  ALA A   1    {pos1[0]:8.3f}{pos1[1]:8.3f}{pos1[2]:8.3f}  1.00  0.00           C\n"
        f"ATOM      2  C   ALA A   1    {pos2[0]:8.3f}{pos2[1]:8.3f}{pos2[2]:8.3f}  1.00  0.00           C\n"
        "CONECT    1    2\n"
        "END\n"
    )
    path.write_text(content, encoding="utf-8")


class PhysicalCorrectnessPBCTests(unittest.TestCase):
    def test_detect_pbc_wrapping_artifacts_identifies_wrapped_covalent_bonds(self):
        u = make_synthetic_universe(wrapped=True, box_size=50.0, has_bonds=True)
        res = detect_pbc_wrapping_artifacts(u, frame_index=0)
        self.assertTrue(res["has_pbc"])
        self.assertTrue(res["wrapped_coordinates_detected"])
        self.assertTrue(res["broken_bonds_detected"])
        self.assertEqual(res["broken_bond_count"], 1)
        self.assertTrue(res["warnings"])
        warning_msg = res["warnings"][0]
        self.assertIn("PBC", warning_msg)
        self.assertIn("coordinate wrapping", warning_msg)
        # Verify loud instructions on recommended imaging fixes
        self.assertIn("apply_imaging_transformations", warning_msg)
        self.assertIn("gmx trjconv", warning_msg)
        self.assertIn("mdconvert", warning_msg)

    def test_detect_pbc_wrapping_artifacts_clean_on_imaged_universe(self):
        u = make_synthetic_universe(wrapped=False, box_size=50.0, has_bonds=True)
        res = detect_pbc_wrapping_artifacts(u, frame_index=0)
        self.assertTrue(res["has_pbc"])
        self.assertFalse(res["wrapped_coordinates_detected"])
        self.assertFalse(res["broken_bonds_detected"])
        self.assertEqual(res["broken_bond_count"], 0)
        self.assertEqual(len(res["warnings"]), 0)

    def test_detect_pbc_wrapping_artifacts_identifies_wrapped_protein_backbone_without_bonds(self):
        # 2 CA atoms in consecutive residues without CONECT/bonds
        u = mda.Universe.empty(2, n_residues=2, atom_resindex=[0, 1], trajectory=True)
        u.add_TopologyAttr("name", ["CA", "CA"])
        u.add_TopologyAttr("resname", ["ALA", "VAL"])
        u.add_TopologyAttr("resid", [1, 2])
        u.add_TopologyAttr("chainID", ["A", "A"])
        u.add_TopologyAttr("segid", ["SEG"])
        u.dimensions = np.array([50.0, 50.0, 50.0, 90.0, 90.0, 90.0])
        u.atoms.positions = np.array([[1.0, 1.0, 1.0], [49.0, 1.0, 1.0]])
        res = detect_pbc_wrapping_artifacts(u)
        self.assertTrue(res["wrapped_coordinates_detected"])
        self.assertGreaterEqual(res["broken_bond_count"], 1)
        self.assertTrue(any(issue["kind"] == "wrapped_protein_backbone" for issue in res["issues"]))

    def test_inspect_md_inputs_warns_on_wrapped_coordinates_by_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "wrapped.pdb"
            make_pdb_file(pdb_path, wrapped=True, box_size=50.0)
            report = inspect_md_inputs(pdb_path, pdb_path, require_imaged=False)

        self.assertEqual(report.status, InputIntegrityStatus.VALID_WITH_WARNINGS)
        self.assertIsNotNone(report.trajectory)
        self.assertTrue(report.trajectory.wrapped_coordinates_detected)
        self.assertTrue(report.trajectory.broken_bonds_detected)
        self.assertGreaterEqual(report.trajectory.broken_bond_count, 1)
        # Warning must contain recommended fixes
        warning_text = " ".join(report.warnings)
        self.assertIn("apply_imaging_transformations", warning_text)
        self.assertIn("gmx trjconv", warning_text)
        self.assertIn("mdconvert", warning_text)

    def test_inspect_md_inputs_errors_on_wrapped_coordinates_when_required(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "wrapped.pdb"
            make_pdb_file(pdb_path, wrapped=True, box_size=50.0)
            report = inspect_md_inputs(pdb_path, pdb_path, require_imaged=True)

        self.assertEqual(report.status, InputIntegrityStatus.INVALID)
        self.assertTrue(report.errors)
        error_text = " ".join(report.errors)
        self.assertIn("PBC", error_text)
        self.assertIn("apply_imaging_transformations", error_text)
        self.assertIn("gmx trjconv", error_text)

    def test_profile_system_flags_wrapped_coordinates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "wrapped.pdb"
            make_pdb_file(pdb_path, wrapped=True, box_size=50.0)
            profile = profile_system(pdb_path, pdb_path)

        self.assertTrue(profile.has_periodic_cell)
        self.assertTrue(profile.wrapped_coordinates_detected)
        warning_text = " ".join(profile.warnings)
        self.assertIn("PBC", warning_text)
        self.assertIn("Recommended imaging fixes", warning_text)

    def test_mismatched_topologies_reported_as_invalid(self):
        top_pdb = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "ATOM      2  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        traj_pdb = (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "top.pdb"
            traj_path = Path(tmpdir) / "traj.pdb"
            top_path.write_text(top_pdb, encoding="utf-8")
            traj_path.write_text(traj_pdb, encoding="utf-8")
            report = inspect_md_inputs(top_path, traj_path)

        self.assertEqual(report.status, InputIntegrityStatus.INVALID)
        self.assertTrue(any("differ" in e.lower() or "atoms" in e.lower() for e in report.errors))

    def test_non_finite_coordinates_reported_as_invalid(self):
        nan_pdb = (
            "ATOM      1  CA  ALA A   1         NaN     NaN     NaN  1.00  0.00           C\n"
            "END\n"
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            pdb_path = Path(tmpdir) / "nan.pdb"
            pdb_path.write_text(nan_pdb, encoding="utf-8")
            report = inspect_md_inputs(pdb_path, pdb_path)

        self.assertEqual(report.status, InputIntegrityStatus.INVALID)
        self.assertTrue(any("non-finite coordinates" in e.lower() for e in report.errors))


class TransparentFeaturePropagationTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame({
            "trajectory_id": ["t1"] * 10 + ["t2"] * 10,
            "replica_id": ["r1"] * 10 + ["r2"] * 10,
            "ensemble_id": ["e1"] * 20,
        })
        wave = np.tile([0.1, 0.9, 0.2, 0.8, 0.3], 4)
        partial = np.copy(wave)
        partial[:5] = np.nan # valid fraction = 15/20 = 0.75 (< 0.95)

        self.features = pd.DataFrame({
            "valid_feature": wave,
            "partial_feature": partial,
            "trajectory_feature": np.repeat([1.0, 2.0], 10),
            "unavailable_feature": np.full(20, np.nan),
        })

    def test_feature_quality_reports_explicit_status_and_reasons_without_zero_filling(self):
        quality = assess_feature_quality(
            self.features,
            self.manifest,
            PrioritizationConfig(min_valid_fraction=0.95),
        ).set_index("Feature")

        # 1. Unavailable feature (all NaNs)
        self.assertEqual(quality.loc["unavailable_feature", "Valid_Fraction"], 0.0)
        self.assertEqual(quality.loc["unavailable_feature", "Eligibility_Status"], "unavailable")
        self.assertIn("valid_fraction=0.0", quality.loc["unavailable_feature", "Ineligibility_Reason"])
        self.assertIn("no silent imputation", quality.loc["unavailable_feature", "Ineligibility_Reason"])
        self.assertFalse(quality.loc["unavailable_feature", "Pareto_Eligible"])

        # 2. Partially missing feature (valid fraction 0.75 < 0.95)
        self.assertEqual(quality.loc["partial_feature", "Valid_Fraction"], 0.75)
        self.assertEqual(quality.loc["partial_feature", "Eligibility_Status"], "insufficient_valid_frames")
        self.assertIn("below threshold", quality.loc["partial_feature", "Ineligibility_Reason"])
        self.assertIn("without imputation", quality.loc["partial_feature", "Ineligibility_Reason"])
        self.assertFalse(quality.loc["partial_feature", "Pareto_Eligible"])

        # 3. Trajectory-level feature (constant per trajectory)
        self.assertEqual(quality.loc["trajectory_feature", "Eligibility_Status"], "trajectory_level")
        self.assertFalse(quality.loc["trajectory_feature", "Pareto_Eligible"])

        # 4. Valid frame feature
        self.assertEqual(quality.loc["valid_feature", "Eligibility_Status"], "eligible")
        self.assertTrue(quality.loc["valid_feature", "Pareto_Eligible"])

    def test_state_explanatory_ranking_propagates_inapplicable_features_with_explicit_status(self):
        assignments = pd.DataFrame({"State": [0] * 10 + [1] * 10})
        # Default: allow_inapplicable=True
        ranking = rank_state_explanatory_features(self.features, assignments)
        self.assertIn("Ranking_Status", ranking.columns)
        self.assertIn("Status_Reason", ranking.columns)

        indexed = ranking.set_index("Feature")
        # Valid feature is ranked
        self.assertEqual(indexed.loc["valid_feature", "Ranking_Status"], "ranked")
        self.assertTrue(np.isfinite(indexed.loc["valid_feature", "Rank"]))
        self.assertTrue(np.isfinite(indexed.loc["valid_feature", "State_Explained_Variance_Eta_Squared"]))

        # Unavailable feature is included with explicit inapplicable status
        self.assertEqual(indexed.loc["unavailable_feature", "Ranking_Status"], "inapplicable")
        self.assertTrue(np.isnan(indexed.loc["unavailable_feature", "Rank"]))
        self.assertTrue(np.isnan(indexed.loc["unavailable_feature", "State_Explained_Variance_Eta_Squared"]))
        self.assertIn("without silent imputation", indexed.loc["unavailable_feature", "Status_Reason"])

        # Partial feature is included with explicit missing_values status
        self.assertEqual(indexed.loc["partial_feature", "Ranking_Status"], "missing_values")
        self.assertTrue(np.isnan(indexed.loc["partial_feature", "Rank"]))
        self.assertIn("without silent imputation", indexed.loc["partial_feature", "Status_Reason"])

        # Strict mode raises ValueError
        with self.assertRaises(ValueError):
            rank_state_explanatory_features(self.features, assignments, allow_inapplicable=False)

    def test_sam_mfp_prioritization_records_status_and_reasons(self):
        quality = assess_feature_quality(self.features, self.manifest)
        salience = pd.DataFrame({
            "Feature": ["valid_feature", "trajectory_feature"],
            "PCA_Salience": [0.8, 0.2],
        })
        candidates, notes = prioritize_sam_mfp(quality, salience, SAMMFPConfig())
        self.assertIn("SAM_MFP_Status", candidates.columns)
        self.assertIn("SAM_MFP_Reason", candidates.columns)

        indexed = candidates.set_index("Feature")
        # Inapplicable / unavailable features are marked ineligible with transparent reason
        self.assertEqual(indexed.loc["unavailable_feature", "SAM_MFP_Status"], "ineligible")
        self.assertIn("unavailable", indexed.loc["unavailable_feature", "SAM_MFP_Reason"])


if __name__ == "__main__":
    unittest.main()
