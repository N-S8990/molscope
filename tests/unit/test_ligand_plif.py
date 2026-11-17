"""Unit tests for Protein-Ligand Interaction Fingerprints (PLIF) plugin."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pytest

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.plugins import PLUGIN_REGISTRY, get_plugin
from molscope.features.ligand.plif import ProteinLigandInteractionFingerprintPlugin
from molscope.profiling.system import ComponentProfile, SystemProfile


FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "protein_ligand_contacts.pdb"


class TestProteinLigandInteractionFingerprintPlugin(unittest.TestCase):
    def test_plugin_registration(self):
        self.assertIn("ligand.plif", PLUGIN_REGISTRY)
        plugin = get_plugin("ligand.plif")
        self.assertIsInstance(plugin, ProteinLigandInteractionFingerprintPlugin)
        self.assertEqual(plugin.plugin_id, "ligand.plif")
        self.assertEqual(plugin.version, "1")
        self.assertEqual(plugin.hydrophobic_cutoff_angstrom, 4.5)
        self.assertEqual(plugin.hbond_cutoff_angstrom, 3.5)
        self.assertEqual(plugin.aromatic_cutoff_angstrom, 5.0)
        self.assertEqual(plugin.salt_bridge_cutoff_angstrom, 4.5)
        self.assertIsNone(plugin.ligand_selection)
        self.assertEqual(plugin.protein_selection, "protein")

    def test_instantiation_with_custom_parameters(self):
        plugin = get_plugin(
            "ligand.plif",
            parameters={
                "ligand_selection": "resname DRG",
                "hydrophobic_cutoff_angstrom": 5.0,
                "hbond_cutoff_angstrom": 3.0,
                "aromatic_cutoff_angstrom": 4.5,
                "salt_bridge_cutoff_angstrom": 4.0,
                "protein_selection": "protein and name CA",
            },
        )
        self.assertEqual(plugin.ligand_selection, "resname DRG")
        self.assertEqual(plugin.hydrophobic_cutoff_angstrom, 5.0)
        self.assertEqual(plugin.hbond_cutoff_angstrom, 3.0)
        self.assertEqual(plugin.aromatic_cutoff_angstrom, 4.5)
        self.assertEqual(plugin.salt_bridge_cutoff_angstrom, 4.0)
        self.assertEqual(plugin.protein_selection, "protein and name CA")

    def test_parameter_validation_errors(self):
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(hydrophobic_cutoff_angstrom=-1.0)
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(hydrophobic_cutoff_angstrom=0.0)
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(hydrophobic_cutoff_angstrom=float("nan"))
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(hydrophobic_cutoff_angstrom=True)
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(hbond_cutoff_angstrom=-3.5)
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(aromatic_cutoff_angstrom="5.0")
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(salt_bridge_cutoff_angstrom=0)
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(ligand_selection="")
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(ligand_selection="   ")
        with self.assertRaises(ValueError):
            ProteinLigandInteractionFingerprintPlugin(protein_selection="")

    def test_assess_applicability(self):
        applicable_profile = SystemProfile(
            topology_path="complex.pdb",
            atom_count=50,
            residue_count=5,
            components=(
                ComponentProfile("chain:A:protein", "protein", 40, 4, ("ALA", "ASP", "GLY", "LYS")),
                ComponentProfile("chain:B:small_molecule_or_polymer", "small_molecule_or_polymer", 10, 1, ("LIG",)),
            ),
            capabilities=frozenset({"protein", "small_molecule_or_polymer", "coordinates", "topology"}),
            has_periodic_cell=False,
            has_bonds=False,
        )
        plugin = ProteinLigandInteractionFingerprintPlugin()
        assessment = plugin.assess(applicable_profile)
        self.assertEqual(assessment.status, ApplicabilityStatus.APPLICABLE)

        # Missing protein capability
        no_protein_profile = SystemProfile(
            topology_path="ligand_only.pdb",
            atom_count=10,
            residue_count=1,
            components=(ComponentProfile("chain:B:small_molecule_or_polymer", "small_molecule_or_polymer", 10, 1, ("LIG",)),),
            capabilities=frozenset({"small_molecule_or_polymer", "coordinates", "topology"}),
            has_periodic_cell=False,
            has_bonds=False,
        )
        assessment = plugin.assess(no_protein_profile)
        self.assertEqual(assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertIn("protein", assessment.missing_capabilities)

        # Missing small_molecule_or_polymer when using default ligand selection
        protein_only_profile = SystemProfile(
            topology_path="protein_only.pdb",
            atom_count=40,
            residue_count=4,
            components=(ComponentProfile("chain:A:protein", "protein", 40, 4, ("ALA", "ASP", "GLY", "LYS")),),
            capabilities=frozenset({"protein", "coordinates", "topology"}),
            has_periodic_cell=False,
            has_bonds=False,
        )
        assessment = plugin.assess(protein_only_profile)
        self.assertEqual(assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertIn("small_molecule_or_polymer", assessment.missing_capabilities)

        # Custom ligand selection does not strictly require small_molecule_or_polymer
        custom_plugin = ProteinLigandInteractionFingerprintPlugin(ligand_selection="resname INH")
        assessment = custom_plugin.assess(protein_only_profile)
        self.assertEqual(assessment.status, ApplicabilityStatus.APPLICABLE)

    def test_extract_against_fixture_protein_ligand_contacts(self):
        plugin = ProteinLigandInteractionFingerprintPlugin()
        result = plugin.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=1)

        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertIsNotNone(result.provenance)
        self.assertEqual(result.provenance.analyzer_id, "ligand.plif")

        frames = result.frame_features
        self.assertEqual(len(frames), 1)

        # Global columns exist
        expected_globals = [
            "frame_index",
            "frame_order",
            "time_ps",
            "PLIF_hydrophobic_count",
            "PLIF_hydrophobic_min_distance",
            "PLIF_hbond_count",
            "PLIF_hbond_min_distance",
            "PLIF_aromatic_count",
            "PLIF_aromatic_min_distance",
            "PLIF_salt_bridge_count",
            "PLIF_salt_bridge_min_distance",
            "PLIF_total_count",
            "PLIF_min_distance",
        ]
        for col in expected_globals:
            self.assertIn(col, frames.columns, f"Missing column: {col}")

        # Per-residue indicator columns exist
        for res in ("ALA_1", "ASP_2", "GLY_3", "LYS_4"):
            for itype in ("contact", "hydrophobic", "hbond", "aromatic", "salt_bridge"):
                col = f"PLIF_{itype}__{res}"
                self.assertIn(col, frames.columns, f"Missing residue column: {col}")

        # ALA 1 CB (1.0, 1.5, 0.0) is within 2.915 Å of LIG C1 (3.5, 0.0, 0.0)
        self.assertEqual(frames.loc[0, "PLIF_hydrophobic__ALA_1"], 1)
        self.assertEqual(frames.loc[0, "PLIF_contact__ALA_1"], 1)
        self.assertGreaterEqual(frames.loc[0, "PLIF_hydrophobic_count"], 1)
        self.assertAlmostEqual(frames.loc[0, "PLIF_hydrophobic_min_distance"], np.sqrt(2.5**2 + 1.5**2), places=3)

        # ASP 2, GLY 3, LYS 4 are beyond cutoffs
        self.assertEqual(frames.loc[0, "PLIF_contact__ASP_2"], 0)
        self.assertEqual(frames.loc[0, "PLIF_contact__GLY_3"], 0)
        self.assertEqual(frames.loc[0, "PLIF_contact__LYS_4"], 0)

        # Fixture ligand only has C1 and C2 (no heteroatoms, no rings, no charges)
        self.assertEqual(frames.loc[0, "PLIF_hbond_count"], 0)
        self.assertTrue(np.isnan(frames.loc[0, "PLIF_hbond_min_distance"]))
        self.assertEqual(frames.loc[0, "PLIF_aromatic_count"], 0)
        self.assertTrue(np.isnan(frames.loc[0, "PLIF_aromatic_min_distance"]))
        self.assertEqual(frames.loc[0, "PLIF_salt_bridge_count"], 0)
        self.assertTrue(np.isnan(frames.loc[0, "PLIF_salt_bridge_min_distance"]))

        # Trajectory features
        traj = result.trajectory_features
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj.loc[0, "PLIF_any_contact_occupancy"], 1.0)
        self.assertEqual(traj.loc[0, "PLIF_contact_occupancy__ALA_1"], 1.0)
        self.assertEqual(traj.loc[0, "PLIF_contact_occupancy__ASP_2"], 0.0)
        self.assertEqual(traj.loc[0, "PLIF_hydrophobic_occupancy__ALA_1"], 1.0)

    def test_extract_with_custom_selection_and_cutoffs(self):
        # Selecting only C2 (at 30, 0, 0) which is far from everything
        plugin_c2 = ProteinLigandInteractionFingerprintPlugin(ligand_selection="resname LIG and name C2")
        result_c2 = plugin_c2.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=1)
        self.assertEqual(result_c2.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(result_c2.frame_features.loc[0, "PLIF_total_count"], 0)
        self.assertEqual(result_c2.frame_features.loc[0, "PLIF_contact__ALA_1"], 0)

        # Tight cutoff 2.0 Å: distance to ALA CB is 2.915 Å so contact is lost
        plugin_tight = ProteinLigandInteractionFingerprintPlugin(hydrophobic_cutoff_angstrom=2.0)
        result_tight = plugin_tight.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=1)
        self.assertEqual(result_tight.frame_features.loc[0, "PLIF_hydrophobic_count"], 0)
        self.assertEqual(result_tight.frame_features.loc[0, "PLIF_hydrophobic__ALA_1"], 0)

    def test_extract_invalid_stride_or_max_frames(self):
        plugin = ProteinLigandInteractionFingerprintPlugin()
        with self.assertRaises(ValueError):
            plugin.extract(FIXTURE_PDB, FIXTURE_PDB, frame_stride=0)
        with self.assertRaises(ValueError):
            plugin.extract(FIXTURE_PDB, FIXTURE_PDB, frame_stride=-1)
        with self.assertRaises(ValueError):
            plugin.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=0)
        with self.assertRaises(ValueError):
            plugin.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=-2)

    def test_extract_empty_ligand_selection(self):
        plugin = ProteinLigandInteractionFingerprintPlugin(ligand_selection="resname NONEXISTENT")
        result = plugin.extract(FIXTURE_PDB, FIXTURE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertTrue(result.frame_features.empty)

    def test_comprehensive_interactions_synthetic_complex(self):
        """Test PLIF with a synthetic complex containing all 4 interaction types:
        - Aromatic stacking (PHE ring <-> Ligand benzene ring)
        - Hydrogen bonds (Ligand donor -> Protein backbone O, Protein donor -> Ligand acceptor)
        - Salt bridge (ASP OD1/OD2 <-> Ligand amine N+; LYS NZ <-> Ligand carboxylate O-)
        - Hydrophobic contact (ALA CB <-> Ligand aliphatic C)
        """
        pdb_content = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       1.000   1.500   0.000  1.00  0.00           C
ATOM      6  N   PHE A   2       0.000  10.000   0.000  1.00  0.00           N
ATOM      7  CA  PHE A   2       1.000  10.000   0.000  1.00  0.00           C
ATOM      8  C   PHE A   2       2.000  10.000   0.000  1.00  0.00           C
ATOM      9  O   PHE A   2       3.000  10.000   0.000  1.00  0.00           O
ATOM     10  CB  PHE A   2       1.000  11.500   0.000  1.00  0.00           C
ATOM     11  CG  PHE A   2       1.000  13.000   0.000  1.00  0.00           C
ATOM     12  CD1 PHE A   2       0.000  13.700   0.000  1.00  0.00           C
ATOM     13  CD2 PHE A   2       2.000  13.700   0.000  1.00  0.00           C
ATOM     14  CE1 PHE A   2       0.000  15.100   0.000  1.00  0.00           C
ATOM     15  CE2 PHE A   2       2.000  15.100   0.000  1.00  0.00           C
ATOM     16  CZ  PHE A   2       1.000  15.800   0.000  1.00  0.00           C
ATOM     17  N   ASP A   3       0.000  20.000   0.000  1.00  0.00           N
ATOM     18  CA  ASP A   3       1.000  20.000   0.000  1.00  0.00           C
ATOM     19  C   ASP A   3       2.000  20.000   0.000  1.00  0.00           C
ATOM     20  O   ASP A   3       3.000  20.000   0.000  1.00  0.00           O
ATOM     21  CB  ASP A   3       1.000  21.500   0.000  1.00  0.00           C
ATOM     22  CG  ASP A   3       1.000  23.000   0.000  1.00  0.00           C
ATOM     23  OD1 ASP A   3       0.000  23.700   0.000  1.00  0.00           O
ATOM     24  OD2 ASP A   3       2.000  23.700   0.000  1.00  0.00           O
HETATM   25  C1  LIG B   1       1.000   3.500   0.000  1.00  0.00           C
HETATM   26  AR1 LIG B   1       1.000  13.000   3.500  1.00  0.00           C
HETATM   27  AR2 LIG B   1       0.000  13.700   3.500  1.00  0.00           C
HETATM   28  AR3 LIG B   1       2.000  13.700   3.500  1.00  0.00           C
HETATM   29  AR4 LIG B   1       0.000  15.100   3.500  1.00  0.00           C
HETATM   30  AR5 LIG B   1       2.000  15.100   3.500  1.00  0.00           C
HETATM   31  AR6 LIG B   1       1.000  15.800   3.500  1.00  0.00           C
HETATM   32  NL1 LIG B   1       0.000  23.700   2.800  1.00  0.00           N
HETATM   33  C2  LIG B   1       0.000  23.700   4.250  1.00  0.00           C
END
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir) / "multi_interaction.pdb"
            temp_path.write_text(pdb_content, encoding="utf-8")

            plugin = ProteinLigandInteractionFingerprintPlugin()
            result = plugin.extract(temp_path, temp_path, max_frames=1)

            self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
            frames = result.frame_features
            self.assertEqual(len(frames), 1)

            # 1. Hydrophobic contact: ALA 1 CB (1, 1.5, 0) <-> C1 (1, 3.5, 0): dist = 2.0 Å <= 4.5 Å
            self.assertEqual(frames.loc[0, "PLIF_hydrophobic__ALA_1"], 1)
            self.assertEqual(frames.loc[0, "PLIF_contact__ALA_1"], 1)

            # 2. Aromatic stacking: PHE ring at Z=0, Ligand benzene ring at Z=3.5: center dist = 3.5 Å <= 5.0 Å
            self.assertEqual(frames.loc[0, "PLIF_aromatic__PHE_2"], 1)
            self.assertEqual(frames.loc[0, "PLIF_contact__PHE_2"], 1)
            self.assertGreaterEqual(frames.loc[0, "PLIF_aromatic_count"], 1)
            self.assertAlmostEqual(frames.loc[0, "PLIF_aromatic_min_distance"], 3.5, places=2)

            # 3. Salt bridge & H-bond: ASP OD1 at (0, 23.7, 0) <-> NL1 at (0, 23.7, 2.8): dist = 2.8 Å <= 3.5 Å (H-bond) and <= 4.5 Å (Salt bridge)
            self.assertEqual(frames.loc[0, "PLIF_salt_bridge__ASP_3"], 1)
            self.assertEqual(frames.loc[0, "PLIF_hbond__ASP_3"], 1)
            self.assertEqual(frames.loc[0, "PLIF_contact__ASP_3"], 1)
            self.assertGreaterEqual(frames.loc[0, "PLIF_salt_bridge_count"], 1)
            self.assertGreaterEqual(frames.loc[0, "PLIF_hbond_count"], 1)
            self.assertAlmostEqual(frames.loc[0, "PLIF_salt_bridge_min_distance"], 2.8, places=2)
            self.assertAlmostEqual(frames.loc[0, "PLIF_hbond_min_distance"], 2.8, places=2)

            # Total interaction count
            self.assertGreaterEqual(frames.loc[0, "PLIF_total_count"], 3)
            self.assertFalse(np.isnan(frames.loc[0, "PLIF_min_distance"]))


if __name__ == "__main__":
    unittest.main()
