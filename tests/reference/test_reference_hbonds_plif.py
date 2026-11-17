"""Reference validation tests for hydrogen bonds and protein-ligand contact cutoffs.

Validates MolScope's:
- ProteinLigandInteractionFingerprintPlugin (PLIF) against direct MDAnalysis geometric criteria.
- ProteinLigandContactPlugin against MDAnalysis distance_array.
- ProteinHydrogenBondPlugin against explicit donor-H...acceptor vector geometry and angular criteria.
- ProteinSaltBridgePlugin against MDAnalysis acidic/basic atom distance array.
- ProteinHydrophobicContactPlugin against sequence-separated carbon-carbon contact criteria.
"""

from pathlib import Path
import tempfile
import unittest

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.lib.distances import distance_array
import numpy as np

from molscope.core.contracts import ApplicabilityStatus
from molscope.features.ligand.plugins import ProteinLigandContactPlugin
from molscope.features.ligand.plif import ProteinLigandInteractionFingerprintPlugin, _find_nonpolar_carbons
from molscope.features.protein.plugins import (
    ProteinHydrogenBondPlugin,
    ProteinHydrophobicContactPlugin,
    ProteinSaltBridgePlugin,
)


FIXTURES = Path(__file__).parents[1] / "fixtures"
PDB_PLIF = FIXTURES / "protein_ligand_contacts.pdb"
PDB_1A22 = FIXTURES / "1A22.pdb"


class ReferenceHBondsPLIFValidationTests(unittest.TestCase):
    """Validation test suite for hydrogen bonds and PLIF contact criteria."""

    def test_plif_interactions_match_direct_mdanalysis_geometry(self):
        """Verify PLIF interaction counts and distances match direct MDAnalysis geometric selections."""
        plugin = ProteinLigandInteractionFingerprintPlugin(
            hydrophobic_cutoff_angstrom=4.5,
            hbond_cutoff_angstrom=3.5,
            aromatic_cutoff_angstrom=5.0,
            salt_bridge_cutoff_angstrom=4.5,
        )
        result = plugin.extract(PDB_PLIF, PDB_PLIF, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_PLIF))
        protein = universe.select_atoms("protein")
        ligand = universe.select_atoms("resname LIG")

        # 1. Hydrophobic reference: nonpolar carbons (carbons without heteroatom bonds) within 4.5 Å
        prot_nonpolar = _find_nonpolar_carbons(protein, protein)
        lig_nonpolar = _find_nonpolar_carbons(ligand, ligand)
        if len(prot_nonpolar) and len(lig_nonpolar):
            h_pairs, h_dists = capped_distance(
                prot_nonpolar.positions, lig_nonpolar.positions, max_cutoff=4.5, return_distances=True
            )
            expected_hydrophobic = len(h_pairs)
            expected_min_h = float(np.min(h_dists)) if len(h_dists) else np.nan
        else:
            expected_hydrophobic = 0
            expected_min_h = np.nan

        reported_hydrophobic = result.frame_features.loc[0, "PLIF_hydrophobic_count"]
        self.assertEqual(reported_hydrophobic, expected_hydrophobic)
        if expected_hydrophobic > 0:
            self.assertAlmostEqual(
                result.frame_features.loc[0, "PLIF_hydrophobic_min_distance"], expected_min_h, places=5
            )

        # 2. Hydrogen bond reference: protein donors/acceptors with ligand acceptors/donors within 3.5 Å
        prot_donors = protein.select_atoms("name N NE NH1 NH2 NZ ND1 NE2 ND2 SG OG OG1 OH")
        prot_acceptors = protein.select_atoms("name O OXT OD1 OD2 OE1 OE2 ND1 NE2 OG OG1 OH SG SD")
        lig_acceptors = ligand.select_atoms("name O* N* S*")
        lig_donors = ligand.select_atoms("name N* O*")

        hb_count = 0
        hb_dists = []
        if len(prot_donors) and len(lig_acceptors):
            p1, d1 = capped_distance(prot_donors.positions, lig_acceptors.positions, max_cutoff=3.5, return_distances=True)
            hb_count += len(p1)
            hb_dists.extend(d1.tolist())
        if len(lig_donors) and len(prot_acceptors):
            p2, d2 = capped_distance(prot_acceptors.positions, lig_donors.positions, max_cutoff=3.5, return_distances=True)
            hb_count += len(p2)
            hb_dists.extend(d2.tolist())

        reported_hb = result.frame_features.loc[0, "PLIF_hbond_count"]
        self.assertEqual(reported_hb, hb_count)
        if hb_count > 0:
            expected_min_hb = float(np.min(hb_dists))
            self.assertAlmostEqual(
                result.frame_features.loc[0, "PLIF_hbond_min_distance"], expected_min_hb, places=5
            )

        # 3. Total PLIF interaction count is exactly the sum of interaction categories
        total_count = (
            result.frame_features.loc[0, "PLIF_hydrophobic_count"]
            + result.frame_features.loc[0, "PLIF_hbond_count"]
            + result.frame_features.loc[0, "PLIF_aromatic_count"]
            + result.frame_features.loc[0, "PLIF_salt_bridge_count"]
        )
        self.assertEqual(result.frame_features.loc[0, "PLIF_total_count"], total_count)

    def test_protein_ligand_contacts_match_mdanalysis_distance_array(self):
        """Verify ProteinLigandContactPlugin against direct all-atom distance array."""
        cutoff = 4.5
        plugin = ProteinLigandContactPlugin()
        result = plugin.extract(PDB_PLIF, PDB_PLIF, max_frames=1)
        self.assertEqual(result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_PLIF))
        protein = universe.select_atoms("protein")
        ligand = universe.select_atoms("resname LIG")

        dists = distance_array(protein.positions, ligand.positions)
        expected_pair_count = int(np.count_nonzero(dists <= cutoff))
        expected_min_distance = float(np.min(dists))

        reported_pair_count = result.frame_features.loc[0, "LIGAND_contact_pair_count"]
        reported_min_dist = result.frame_features.loc[0, "LIGAND_min_contact_distance"]

        self.assertEqual(reported_pair_count, expected_pair_count)
        self.assertAlmostEqual(reported_min_dist, expected_min_distance, places=5)

        # Check residue contact fractions
        for residue in protein.residues:
            res_dists = distance_array(residue.atoms.positions, ligand.positions)
            expected_frac = float(np.count_nonzero(res_dists <= cutoff) / (len(residue.atoms) * len(ligand)))
            col = f"LIGAND_residue_contact_fraction__{residue.resname}_{residue.resid}"
            self.assertIn(col, result.frame_features.columns)
            self.assertAlmostEqual(result.frame_features.loc[0, col], expected_frac, places=6)

    def test_protein_hydrogen_bond_directional_criteria(self):
        """Validate ProteinHydrogenBondPlugin against vector geometry and strict cosine limits."""
        test_pdb = """ATOM      1  N   ALA A   1       0.000   0.000   0.000  1.00  0.00           N
ATOM      2  H   ALA A   1       1.000   0.000   0.000  1.00  0.00           H
ATOM      3  CA  ALA A   1       0.000   1.500   0.000  1.00  0.00           C
ATOM      4  O   GLY A   2       2.800   0.315   0.000  1.00  0.00           O
ATOM      5  N   ALA A   3       0.000  10.000   0.000  1.00  0.00           N
ATOM      6  H   ALA A   3       1.000  10.000   0.000  1.00  0.00           H
ATOM      7  CA  ALA A   3       0.000  11.500   0.000  1.00  0.00           C
ATOM      8  O   GLY A   4       1.000  12.800   0.000  1.00  0.00           O
ATOM      9  N   ALA A   5       0.000  20.000   0.000  1.00  0.00           N
ATOM     10  H   ALA A   5       1.000  20.000   0.000  1.00  0.00           H
ATOM     11  CA  ALA A   5       0.000  21.500   0.000  1.00  0.00           C
ATOM     12  O   GLY A   6       4.200  20.000   0.000  1.00  0.00           O
CONECT    1    2    3
CONECT    2    1
CONECT    3    1
CONECT    5    6    7
CONECT    6    5
CONECT    7    5
CONECT    9   10   11
CONECT   10    9
CONECT   11    9
END
"""
        with tempfile.NamedTemporaryFile("w", suffix=".pdb", delete=False) as f:
            f.write(test_pdb)
            tmp_path = f.name

        try:
            plugin = ProteinHydrogenBondPlugin()
            res = plugin.extract(tmp_path, tmp_path, max_frames=1)
            self.assertEqual(res.assessment.status, ApplicabilityStatus.APPLICABLE)

            reported_count = res.frame_features.loc[0, "HBOND_count"]
            self.assertEqual(reported_count, 1, "Only pair 1 should satisfy distance and angle constraints")

            reported_dist = res.frame_features.loc[0, "HBOND_min_donor_acceptor_distance"]
            expected_dist = float(np.linalg.norm(np.array([2.800, 0.315, 0.0]) - np.array([0.0, 0.0, 0.0])))
            self.assertAlmostEqual(reported_dist, expected_dist, places=4)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_protein_salt_bridge_and_hydrophobic_contact_geometry(self):
        """Verify ProteinSaltBridgePlugin and ProteinHydrophobicContactPlugin cutoffs vs MDAnalysis."""
        # 1. Salt bridges on protein_ligand_contacts.pdb
        sb_plugin = ProteinSaltBridgePlugin()
        sb_result = sb_plugin.extract(PDB_PLIF, PDB_PLIF, max_frames=1)
        self.assertEqual(sb_result.assessment.status, ApplicabilityStatus.APPLICABLE)

        universe = mda.Universe(str(PDB_PLIF))
        acidic = universe.select_atoms("resname ASP GLU and name OD1 OD2 OE1 OE2")
        basic = universe.select_atoms("resname LYS ARG and name NZ NH1 NH2 NE")
        sb_dists = distance_array(acidic.positions, basic.positions)
        expected_sb_count = int(np.count_nonzero(sb_dists <= 4.0))

        self.assertEqual(sb_result.frame_features.loc[0, "SALT_BRIDGE_pair_count"], expected_sb_count)

        # 2. Hydrophobic contacts on 1A22.pdb
        hp_plugin = ProteinHydrophobicContactPlugin()
        hp_result = hp_plugin.extract(PDB_1A22, PDB_1A22, max_frames=1)
        self.assertEqual(hp_result.assessment.status, ApplicabilityStatus.APPLICABLE)

        u1a22 = mda.Universe(str(PDB_1A22))
        hydrophobic = u1a22.select_atoms("protein and resname ALA VAL ILE LEU MET PHE TRP TYR PRO and element C")
        pairs, dists = capped_distance(hydrophobic.positions, hydrophobic.positions, max_cutoff=4.5, return_distances=True)

        atom_resix = np.asarray([a.residue.ix for a in hydrophobic])
        atom_resids = np.asarray([a.residue.resid for a in hydrophobic])
        atom_chains = np.asarray([a.chainID.strip() for a in hydrophobic])

        kept = []
        for idx, (l, r) in enumerate(pairs):
            if l >= r or atom_resix[l] == atom_resix[r]:
                continue
            if atom_chains[l] == atom_chains[r] and abs(atom_resids[l] - atom_resids[r]) < 3:
                continue
            kept.append(float(dists[idx]))

        self.assertEqual(hp_result.frame_features.loc[0, "HYDROPHOBIC_contact_pair_count"], len(kept))
        self.assertAlmostEqual(
            hp_result.frame_features.loc[0, "HYDROPHOBIC_min_contact_distance"], min(kept), places=5
        )


if __name__ == "__main__":
    unittest.main()
