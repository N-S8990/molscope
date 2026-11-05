import unittest

from molscope.analyzers import AnalyzerCatalog
from molscope.contracts import ApplicabilityStatus
from molscope.system_profile import ComponentProfile, SystemProfile


def profile_with(*capabilities: str) -> SystemProfile:
    return SystemProfile(
        topology_path="synthetic.pdb",
        atom_count=10,
        residue_count=2,
        components=(ComponentProfile("component:protein", "protein", 10, 2, ("ALA",)),),
        capabilities=frozenset({"coordinates", *capabilities}),
        has_periodic_cell="periodic_cell" in capabilities,
        has_bonds="bonds" in capabilities,
    )


class ApplicabilityTests(unittest.TestCase):
    def test_protein_torsion_requires_bonds(self):
        profile = profile_with("protein", "protein_backbone")
        assessment = next(item for item in AnalyzerCatalog().assess(profile) if item.feature_id == "protein.backbone_torsions")
        self.assertEqual(assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertEqual(assessment.missing_capabilities, ("bonds",))

    def test_protein_torsion_applies_with_bonds(self):
        profile = profile_with("protein", "protein_backbone", "bonds")
        assessment = next(item for item in AnalyzerCatalog().assess(profile) if item.feature_id == "protein.backbone_torsions")
        self.assertEqual(assessment.status, ApplicabilityStatus.APPLICABLE)

    def test_membrane_analysis_is_not_zero_filled_for_protein(self):
        profile = profile_with("protein", "protein_backbone", "bonds")
        assessment = next(item for item in AnalyzerCatalog().assess(profile) if item.feature_id == "membrane.order")
        self.assertEqual(assessment.status, ApplicabilityStatus.NOT_APPLICABLE)
        self.assertIn("lipid", assessment.missing_capabilities)


if __name__ == "__main__":
    unittest.main()
