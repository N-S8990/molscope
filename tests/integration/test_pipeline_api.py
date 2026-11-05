import tempfile
import unittest
import json
from pathlib import Path

import pandas as pd

from molscope.discovery_pipeline import DiscoveryPipelineConfig
from molscope.pipeline import MDContactMLPipeline
from molscope.manifest import TrajectoryInput
from molscope.generic_features import GenericFeatureConfig
from molscope.storage import FeatureCache, extraction_cache_key


REFERENCE_PDB = str(Path(__file__).parents[1] / "fixtures" / "1A22.pdb")

class PipelineAPITests(unittest.TestCase):
    def test_runs_and_writes_a_feature_matrix_workflow(self):
        features = pd.DataFrame({"signal": [-1, -0.5, 1, 1.5] * 3, "noise": [0, 1, 0, 1] * 3})
        manifest = pd.DataFrame([
            {"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e", "frame_order": frame}
            for trajectory in range(3) for frame in range(4)
        ])
        pipeline = MDContactMLPipeline(DiscoveryPipelineConfig(pca_components=1, state_count=2)).load_feature_matrix(features, manifest).run()
        self.assertIn("Feature", pipeline.table("sam_mfp_candidates").columns)
        with tempfile.TemporaryDirectory() as temporary:
            pipeline.write_results(temporary)
            self.assertTrue((Path(temporary) / "research_report.md").is_file())
            self.assertTrue((Path(temporary) / "sam_mfp_tradeoff.png").is_file())
            self.assertTrue((Path(temporary) / "state_explanatory_feature_ranking.png").is_file())

    def test_loads_trajectories_and_builds_identity_manifest(self):
        inputs = [
            TrajectoryInput(REFERENCE_PDB, REFERENCE_PDB, f"t{index}", f"r{index}", "e", metadata={"condition": index % 2})
            for index in range(2)
        ]
        pipeline = MDContactMLPipeline().load_trajectories(inputs, GenericFeatureConfig(max_frames=1))
        self.assertEqual(len(pipeline.features), 2)
        self.assertEqual(set(pipeline.manifest["trajectory_id"]), {"t0", "t1"})
        self.assertIn("metadata_condition", pipeline.manifest.columns)

    def test_loads_selected_plugin_features_with_generic_features(self):
        inputs = [TrajectoryInput(REFERENCE_PDB, REFERENCE_PDB, f"t{index}", f"r{index}", "e") for index in range(2)]
        pipeline = MDContactMLPipeline().load_trajectories(
            inputs, GenericFeatureConfig(max_frames=1), plugins=["protein.salt_bridges"],
        )
        self.assertIn("SALT_BRIDGE_pair_count", pipeline.features.columns)

    def test_load_project_applies_declared_analysis_and_exposes_scalar_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = {
                "project_id": "configured-study",
                "analysis": {"generic_feature_parameters": {"max_frames": 1}},
                "systems": [{
                    "system_id": "system", "topology": str(Path(REFERENCE_PDB).resolve()),
                    "ensembles": [{"ensemble_id": "condition-a", "metadata": {"condition": "a"}, "replicas": [
                        {"replica_id": "replica-1", "trajectory": str(Path(REFERENCE_PDB).resolve())},
                    ]}],
                }],
            }
            project_path = root / "project.json"
            project_path.write_text(json.dumps(payload), encoding="utf-8")
            pipeline = MDContactMLPipeline().load_project(project_path)
        self.assertEqual(len(pipeline.features), 1)
        self.assertEqual(pipeline.manifest.loc[0, "metadata_condition"], "a")
        self.assertEqual(pipeline.manifest.loc[0, "metadata_ensemble_condition"], "a")

    def test_attaches_exactly_aligned_external_energy_terms(self):
        features = pd.DataFrame({"feature": [1.0, 2.0]})
        manifest = pd.DataFrame({"trajectory_id": ["t1", "t1"], "replica_id": ["r1", "r1"], "ensemble_id": ["e", "e"], "frame_order": [0, 1], "time_ps": [0.0, 2.0]})
        energies = pd.DataFrame({"trajectory_id": ["t1", "t1"], "time_ps": [0.0, 2.0], "potential_kj_mol": [-1.0, -2.0]})
        pipeline = MDContactMLPipeline().load_feature_matrix(features, manifest).add_external_energies(energies)
        self.assertEqual(pipeline.features["potential_kj_mol"].tolist(), [-1.0, -2.0])

    def test_saves_and_restores_reproducible_session_inputs(self):
        features = pd.DataFrame({"feature": [1.0, 2.0, 3.0]})
        manifest = pd.DataFrame({"trajectory_id": ["t1"] * 3, "replica_id": ["r1"] * 3, "ensemble_id": ["e"] * 3, "frame_order": [0, 1, 2]})
        with tempfile.TemporaryDirectory() as temporary:
            original = MDContactMLPipeline(DiscoveryPipelineConfig(pca_components=1, state_count=2)).load_feature_matrix(features, manifest)
            original.save_session(temporary)
            restored = MDContactMLPipeline.load_session(temporary)
        self.assertEqual(restored.config.pca_components, 1)
        self.assertTrue(restored.features.equals(features))
        self.assertTrue(restored.manifest.equals(manifest))

    def test_session_uses_documented_persistent_table_storage(self):
        features = pd.DataFrame({"feature": [1.0, 2.0]})
        manifest = pd.DataFrame({"trajectory_id": ["t1", "t1"], "replica_id": ["r1", "r1"], "ensemble_id": ["e", "e"], "frame_order": [0, 1]})
        with tempfile.TemporaryDirectory() as temporary:
            MDContactMLPipeline().load_feature_matrix(features, manifest).save_session(temporary)
            storage = json.loads((Path(temporary) / "session_storage.json").read_text(encoding="utf-8"))
            self.assertIn(Path(storage["features"]).suffix, {".parquet", ".csv"})
            self.assertTrue((Path(temporary) / storage["features"]).is_file())

    def test_trajectory_extraction_cache_is_content_addressed_and_reused(self):
        config = GenericFeatureConfig(max_frames=1)
        source = Path(REFERENCE_PDB)
        key = extraction_cache_key(source, source, generic_config=config, plugins={})
        with tempfile.TemporaryDirectory() as temporary:
            inputs = [TrajectoryInput(source, source, "t0", "r0", "e")]
            first = MDContactMLPipeline().load_trajectories(inputs, config, cache_dir=temporary)
            cache = FeatureCache(temporary)
            cached = cache.get(key)
            self.assertIsNotNone(cached)
            second = MDContactMLPipeline().load_trajectories(inputs, config, cache_dir=temporary)
        self.assertTrue(first.features.equals(second.features))


if __name__ == "__main__":
    unittest.main()
