import unittest

import numpy as np
import pandas as pd

from molscope.discovery_pipeline import DiscoveryPipelineConfig, run_target_free_discovery_pipeline


class DiscoveryPipelineTests(unittest.TestCase):
    def test_runs_label_independent_pipeline_end_to_end(self):
        rows, manifest = [], []
        for trajectory in range(3):
            for frame in range(12):
                rows.append({"signal": (-1 if frame < 6 else 1) + trajectory * 0.01, "noise": np.sin(frame + trajectory)})
                manifest.append({"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e", "frame_order": frame})
        outputs = run_target_free_discovery_pipeline(pd.DataFrame(rows), pd.DataFrame(manifest), DiscoveryPipelineConfig(pca_components=1, state_count=2))
        self.assertIn("sam_mfp_candidates", outputs)
        self.assertIn("state_assignments", outputs)
        ranking = outputs["state_explanatory_feature_ranking"]
        self.assertEqual(ranking.iloc[0]["Interpretation"], "descriptive_candidate_state_separation_not_predictive_or_causal")
        self.assertIn("signal", set(ranking["Feature"]))
        self.assertIn("Target-free prioritized", outputs["research_report_markdown"])

    def test_adds_grouped_supervised_ml_when_a_valid_target_is_supplied(self):
        rows, manifest = [], []
        for trajectory in range(4):
            target = trajectory // 2
            for frame in range(10):
                rows.append({"signal": target * 3 + frame * 0.01, "noise": np.sin(frame + trajectory)})
                manifest.append({"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e", "frame_order": frame, "condition": target})
        outputs = run_target_free_discovery_pipeline(
            pd.DataFrame(rows), pd.DataFrame(manifest),
            DiscoveryPipelineConfig(pca_components=1, state_count=2, supervised_splits=2, supervised_bootstrap_samples=5),
            target_column="condition",
        )
        self.assertIn("supervised_model_summary", outputs)
        self.assertIn("supervised_predictive_feature_ranking", outputs)
        self.assertIn("Predictive ML", outputs["research_report_markdown"])
        self.assertIn("Predictive feature ranking", outputs["research_report_markdown"])


if __name__ == "__main__":
    unittest.main()
