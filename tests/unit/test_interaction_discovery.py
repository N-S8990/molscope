import unittest

from molscope.analyzers import AnalyzerCatalog
from molscope.contracts import ApplicabilityStatus
from molscope.system_profile import ComponentProfile, SystemProfile


class InteractionDiscoveryTests(unittest.TestCase):
    def test_hydrogen_bond_discovery_reports_missing_charges(self):
        profile = SystemProfile(
            topology_path="uncharged.pdb", atom_count=12, residue_count=2,
            components=(
                ComponentProfile("chain:A:protein", "protein", 6, 1, ("ALA",)),
                ComponentProfile("chain:B:protein", "protein", 6, 1, ("SER",)),
            ),
            capabilities=frozenset({"coordinates", "bonds", "multiple_molecular_components", "protein"}),
            has_periodic_cell=False, has_bonds=True,
        )
        assessment = next(item for item in AnalyzerCatalog().assess(profile) if item.feature_id == "interactions.hydrogen_bonds")
        self.assertEqual(assessment.status, ApplicabilityStatus.MISSING_INPUT)
        self.assertEqual(assessment.missing_inputs, ("partial_charges",))


if __name__ == "__main__":
    unittest.main()
