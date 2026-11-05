import unittest

from molscope.registries import DEFAULT_FEATURE_REGISTRY, FeatureStatus
from molscope.system_profile import ComponentProfile, SystemProfile


def profile(*capabilities: str) -> SystemProfile:
    return SystemProfile(
        topology_path="synthetic.pdb", atom_count=3, residue_count=1,
        components=(ComponentProfile("component", "small_molecule_or_polymer", 3, 1, ("UNK",)),),
        capabilities=frozenset({"coordinates", *capabilities}),
        has_periodic_cell=False, has_bonds=False,
    )


class RegistryTests(unittest.TestCase):
    def test_every_platform_plugin_has_a_scientific_registry_definition(self):
        implementations = {definition.implementation for definition in DEFAULT_FEATURE_REGISTRY.values()}
        expected = {
            "molscope.plugins.ProteinBackboneTorsionPlugin",
            "molscope.plugins.ProteinCADistanceContactPlugin",
            "molscope.plugins.ProteinDSSPPlugin",
            "molscope.plugins.ProteinLigandContactPlugin",
            "molscope.plugins.ProteinSaltBridgePlugin",
            "molscope.plugins.ProteinHydrophobicContactPlugin",
            "molscope.plugins.SASAPlugin",
            "molscope.plugins.ProteinHydrogenBondPlugin",
            "molscope.plugins.ProteinAromaticStackingPlugin",
            "molscope.plugins.SolvationShellPlugin",
            "molscope.plugins.MolecularDipolePlugin",
            "molscope.plugins.ProteinContactNetworkPlugin",
        }
        self.assertTrue(expected.issubset(implementations))
        for definition in DEFAULT_FEATURE_REGISTRY.values():
            self.assertIn(definition.method_id, {method.method_id for method in DEFAULT_FEATURE_REGISTRY.methods.values()})

    def test_matched_policy_reference_validation_is_explicit_and_scoped(self):
        expected = {
            "interactions.protein_ligand_residue_contact_fraction",
            "interactions.protein_salt_bridge_pair_count",
            "structural.sasa",
            "solvation.water_shells",
            "nucleic_acid.base_proximities",
        }
        observed = {
            definition.feature_id
            for definition in DEFAULT_FEATURE_REGISTRY.values()
            if definition.validation_status == "matched_policy_reference_fixture"
        }
        self.assertEqual(observed, expected)

    def test_registered_feature_requires_trajectory_input(self):
        result = DEFAULT_FEATURE_REGISTRY.assess(
            "geometry.global_radius_of_gyration", profile(), available_inputs={"topology"},
        )
        self.assertEqual(result.status, FeatureStatus.MISSING_INPUT)
        self.assertEqual(result.missing_inputs, ("trajectory",))

    def test_component_feature_is_not_applicable_to_single_component_system(self):
        result = DEFAULT_FEATURE_REGISTRY.assess(
            "geometry.component_contact_fraction", profile(), available_inputs={"topology", "trajectory"},
        )
        self.assertEqual(result.status, FeatureStatus.NOT_APPLICABLE)
        self.assertIn("multiple_molecular_components", result.missing_capabilities)

    def test_user_and_cost_skips_are_explicit_states(self):
        user = DEFAULT_FEATURE_REGISTRY.assess(
            "geometry.global_radius_of_gyration", profile(), available_inputs=set(), skipped_by_user=True,
        )
        cost = DEFAULT_FEATURE_REGISTRY.assess(
            "geometry.global_radius_of_gyration", profile(), available_inputs=set(), skipped_by_cost_policy=True,
        )
        self.assertEqual(user.status, FeatureStatus.SKIPPED_BY_USER)
        self.assertEqual(cost.status, FeatureStatus.SKIPPED_BY_COST_POLICY)

    def test_protein_ligand_contact_feature_requires_protein_and_ligand_capabilities(self):
        profile = SystemProfile(
            topology_path="single_component.pdb", atom_count=4, residue_count=1, components=(),
            capabilities=frozenset({"coordinates", "topology"}), has_periodic_cell=False, has_bonds=False,
        )
        availability = DEFAULT_FEATURE_REGISTRY.assess(
            "interactions.protein_ligand_residue_contact_fraction", profile, available_inputs={"topology", "trajectory"},
        )
        self.assertEqual(availability.status, FeatureStatus.NOT_APPLICABLE)
        self.assertIn("protein", availability.missing_capabilities)

    def test_sasa_registry_requires_its_optional_backend_input(self):
        availability = DEFAULT_FEATURE_REGISTRY.assess(
            "structural.sasa", profile(), available_inputs={"topology", "trajectory"},
        )
        self.assertEqual(availability.status, FeatureStatus.MISSING_INPUT)
        self.assertEqual(availability.missing_inputs, ("freesasa_backend",))


if __name__ == "__main__":
    unittest.main()
