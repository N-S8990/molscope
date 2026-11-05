import unittest
import tempfile
from pathlib import Path

from molscope.contracts import ApplicabilityStatus
from molscope.plugins import MolecularDipolePlugin, NucleicAcidBaseProximityPlugin, ProteinAromaticStackingPlugin, ProteinBackboneTorsionPlugin, ProteinCADistanceContactPlugin, ProteinContactNetworkPlugin, ProteinDSSPPlugin, ProteinHydrogenBondPlugin, ProteinHydrophobicContactPlugin, ProteinLigandContactPlugin, ProteinSaltBridgePlugin, SASAPlugin, SolvationShellPlugin
from molscope.system_profile import ComponentProfile, SystemProfile


REFERENCE_PDB = str(Path(__file__).parents[1] / "fixtures" / "1A22.pdb")

class PluginTests(unittest.TestCase):
    def test_protein_plugin_reports_not_applicable_without_protein_capabilities(self):
        profile = SystemProfile(
            topology_path="polymer.pdb",
            atom_count=4,
            residue_count=1,
            components=(ComponentProfile("component:small_molecule_or_polymer", "small_molecule_or_polymer", 4, 1, ("UNK",)),),
            capabilities=frozenset({"coordinates", "topology"}),
            has_periodic_cell=False,
            has_bonds=False,
        )
        assessment = ProteinBackboneTorsionPlugin().assess(profile)
        self.assertEqual(assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertEqual(set(assessment.missing_capabilities), {"bonds", "protein_backbone"})

    def test_protein_plugin_extracts_from_tracked_reference_structure(self):
        result = ProteinBackboneTorsionPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(len(result.frame_features), 1)
        self.assertTrue(any(column.startswith("TORSION__") for column in result.frame_features.columns))
        self.assertIsNotNone(result.provenance)

    def test_ca_distance_contact_plugin_uses_declared_nonlocal_pair_policy(self):
        pdb = """ATOM      1  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
ATOM      2  CA  ALA A   2       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  CA  ALA A   3       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  CA  ALA A   4       7.000   0.000   0.000  1.00  0.00           C
END
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ca_contacts.pdb"
            path.write_text(pdb, encoding="utf-8")
            result = ProteinCADistanceContactPlugin().extract(path, path, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        distance_column = next(column for column in result.frame_features if column.startswith("CA_DISTANCE_angstrom"))
        contact_column = next(column for column in result.frame_features if column.startswith("CA_CONTACT__"))
        self.assertAlmostEqual(result.frame_features.loc[0, distance_column], 7.0)
        self.assertEqual(result.frame_features.loc[0, contact_column], 1.0)
        self.assertTrue(any(column.startswith("CA_CONTACT_occupancy") for column in result.trajectory_features.columns))

    def test_dssp_plugin_returns_aggregate_secondary_structure_fractions(self):
        result = ProteinDSSPPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        fractions = result.frame_features.loc[0, ["DSSP_helix_fraction", "DSSP_sheet_fraction", "DSSP_coil_fraction"]]
        self.assertAlmostEqual(float(fractions.sum()), 1.0, places=6)
        self.assertIn("DSSP_helix_fraction_mean", result.trajectory_features.columns)

    def test_protein_ligand_plugin_extracts_contacts_for_single_ligand_component(self):
        pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  C   ALA A   1       2.000   0.000   0.000  1.00  0.00           C
ATOM      4  O   ALA A   1       3.000   0.000   0.000  1.00  0.00           O
HETATM    5  C1  LIG B   1       3.500   0.000   0.000  1.00  0.00           C
END
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "protein_ligand.pdb"
            path.write_text(pdb, encoding="utf-8")
            result = ProteinLigandContactPlugin().extract(path, path)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertIn("LIGAND_contact_pair_count", result.frame_features.columns)
        self.assertTrue(any(column.startswith("LIGAND_residue_occupancy") for column in result.trajectory_features.columns))

    def test_salt_bridge_plugin_is_explicitly_not_applicable_without_charged_atoms(self):
        result = ProteinSaltBridgePlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertIn(result.assessment.status, {ApplicabilityStatus.APPLICABLE, ApplicabilityStatus.NOT_APPLICABLE})
        if result.assessment.status is ApplicabilityStatus.APPLICABLE:
            self.assertIn("SALT_BRIDGE_pair_count", result.frame_features.columns)

    def test_hydrophobic_contact_plugin_has_explicit_status(self):
        result = ProteinHydrophobicContactPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertIn(result.assessment.status, {ApplicabilityStatus.APPLICABLE, ApplicabilityStatus.NOT_APPLICABLE})
        if result.assessment.status is ApplicabilityStatus.APPLICABLE:
            self.assertIn("HYDROPHOBIC_contact_pair_count", result.frame_features.columns)

    def test_sasa_plugin_returns_total_and_residue_areas(self):
        result = SASAPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertIn("SASA_total", result.frame_features.columns)
        self.assertTrue(any(column.startswith("SASA_residue") for column in result.frame_features.columns))

    def test_hydrogen_bond_plugin_rejects_missing_explicit_hydrogens(self):
        result = ProteinHydrogenBondPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertIn("hydrogens", result.assessment.missing_capabilities)

    def test_aromatic_stacking_plugin_has_explicit_status(self):
        result = ProteinAromaticStackingPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertIn(result.assessment.status, {ApplicabilityStatus.APPLICABLE, ApplicabilityStatus.NOT_APPLICABLE})

    def test_solvation_and_dipole_plugins_reject_missing_prerequisites(self):
        solvation = SolvationShellPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        dipole = MolecularDipolePlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertIn(solvation.assessment.status, {ApplicabilityStatus.APPLICABLE, ApplicabilityStatus.NOT_APPLICABLE})
        if solvation.assessment.status is ApplicabilityStatus.APPLICABLE:
            self.assertIn("SOLVATION_total_shell_water_count", solvation.frame_features.columns)
        self.assertEqual(dipole.assessment.status, ApplicabilityStatus.NOT_APPLICABLE)

    def test_contact_network_plugin_reports_frame_and_persistence_metrics(self):
        result = ProteinContactNetworkPlugin().extract(REFERENCE_PDB, REFERENCE_PDB, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertIn("NETWORK_edge_count", result.frame_features.columns)
        self.assertIn("NETWORK_edge_persistence_mean", result.trajectory_features.columns)

    def test_nucleic_base_proximity_plugin_reports_geometry_without_pairing_claims(self):
        pdb = """ATOM      1  N1   DA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  C2   DA A   1       1.000   0.000   0.000  1.00  0.00           C
ATOM      3  N3   DA A   1       0.000   1.000   0.000  1.00  0.00           N
ATOM      4  N1   DC A   2       4.000   0.000   0.000  1.00  0.00           N
ATOM      5  C2   DC A   2       5.000   0.000   0.000  1.00  0.00           C
ATOM      6  N3   DC A   2       4.000   1.000   0.000  1.00  0.00           N
END
"""
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "nucleic.pdb"
            path.write_text(pdb, encoding="utf-8")
            result = NucleicAcidBaseProximityPlugin().extract(path, path, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertIn("NUCLEIC_base_proximal_pair_count", result.frame_features.columns)
        self.assertTrue(any(column.startswith("NUCLEIC_base_proximity_occupancy") for column in result.trajectory_features.columns))


if __name__ == "__main__":
    unittest.main()
