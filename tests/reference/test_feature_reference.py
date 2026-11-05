"""Cross-system reference checks for capability-gated plugin wrappers.

These checks use the trusted backend/API directly, rather than the plugin's
reported values.  They are deliberately limited to matched non-PBC policies;
they do not validate imaging or periodic-boundary behaviour.
"""

from pathlib import Path
import unittest

import freesasa
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
import numpy as np

from molscope.contracts import ApplicabilityStatus
from molscope.plugins import (
    NucleicAcidBaseProximityPlugin,
    ProteinLigandContactPlugin,
    ProteinSaltBridgePlugin,
    SASAPlugin,
    SolvationShellPlugin,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"


class ReferenceValidationTests(unittest.TestCase):
    def test_protein_ligand_and_salt_bridge_match_mdanalysis_distance_reference(self):
        path = FIXTURES / "protein_ligand_contacts.pdb"
        universe = mda.Universe(path)

        ligand = universe.select_atoms("resname LIG")
        protein = universe.select_atoms("protein")
        expected_ligand_pairs = int(np.count_nonzero(
            distance_array(protein.positions, ligand.positions) <= ProteinLigandContactPlugin.contact_cutoff_angstrom
        ))
        ligand_result = ProteinLigandContactPlugin().extract(path, path, max_frames=1)
        self.assertEqual(ligand_result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(ligand_result.frame_features.loc[0, "LIGAND_contact_pair_count"], expected_ligand_pairs)

        acidic = universe.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
        basic = universe.select_atoms("resname LYS ARG and name NZ NH1 NH2 NE")
        expected_bridges = int(np.count_nonzero(
            distance_array(acidic.positions, basic.positions) <= ProteinSaltBridgePlugin.cutoff_angstrom
        ))
        bridge_result = ProteinSaltBridgePlugin().extract(path, path, max_frames=1)
        self.assertEqual(bridge_result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(bridge_result.frame_features.loc[0, "SALT_BRIDGE_pair_count"], expected_bridges)

    def test_solvation_shell_matches_mdanalysis_distance_reference(self):
        path = FIXTURES / "solvated_protein.pdb"
        universe = mda.Universe(path)
        molecular = universe.select_atoms("not resname HOH")
        water_oxygen = universe.select_atoms("resname HOH and name O")
        distances = distance_array(molecular.positions, water_oxygen.positions)
        expected_water_count = int(np.count_nonzero(np.any(distances <= SolvationShellPlugin.cutoff_angstrom, axis=0)))

        result = SolvationShellPlugin().extract(path, path, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(result.frame_features.loc[0, "SOLVATION_total_shell_water_count"], expected_water_count)

    def test_nucleic_base_centers_match_direct_geometry_reference(self):
        path = FIXTURES / "nucleic_base_centers.pdb"
        universe = mda.Universe(path)
        centers = [residue.atoms.positions.mean(axis=0) for residue in universe.select_atoms("nucleic").residues]
        expected_distances = [float(np.linalg.norm(left - right)) for index, left in enumerate(centers) for right in centers[index + 1:]]
        expected_count = sum(distance <= NucleicAcidBaseProximityPlugin.maximum_center_distance_angstrom for distance in expected_distances)

        result = NucleicAcidBaseProximityPlugin().extract(path, path, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertEqual(result.frame_features.loc[0, "NUCLEIC_base_proximal_pair_count"], expected_count)
        self.assertAlmostEqual(result.frame_features.loc[0, "NUCLEIC_min_base_center_distance"], min(expected_distances))

    def test_sasa_wrapper_matches_direct_freesasa_reference(self):
        path = FIXTURES / "protein_ligand_contacts.pdb"
        plugin = SASAPlugin()
        universe = mda.Universe(path)
        parameters = freesasa.Parameters({"probe-radius": plugin.probe_radius_angstrom, "n-points": plugin.n_points})
        structure = freesasa.Structure()
        for atom in universe.atoms:
            structure.addAtom(atom.name.strip(), atom.resname.strip(), str(atom.resid), (atom.chainID or atom.segid or " ").strip()[:1] or " ", *map(float, atom.position))
        expected_total = freesasa.calc(structure, parameters).totalArea()

        result = plugin.extract(path, path, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)
        self.assertAlmostEqual(result.frame_features.loc[0, "SASA_total"], expected_total, places=6)


if __name__ == "__main__":
    unittest.main()
