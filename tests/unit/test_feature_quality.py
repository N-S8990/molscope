import unittest

import numpy as np
import pandas as pd

from molscope.prioritization import PrioritizationConfig, assess_feature_quality


class FeatureQualityTests(unittest.TestCase):
    def setUp(self):
        self.manifest = pd.DataFrame({
            "trajectory_id": ["a1"] * 8 + ["a2"] * 8 + ["b1"] * 8 + ["b2"] * 8,
            "replica_id": ["r1"] * 8 + ["r2"] * 8 + ["r1"] * 8 + ["r2"] * 8,
            "ensemble_id": ["a"] * 16 + ["b"] * 16,
        })
        wave = np.tile(np.array([0.0, 1.0, 0.0, 1.0, 0.2, 0.8, 0.2, 0.8]), 4)
        self.features = pd.DataFrame({
            "frame_feature": wave,
            "trajectory_feature": np.repeat([1.0, 2.0, 1.0, 2.0], 8),
            "unavailable_feature": np.nan,
        })

    def test_trajectory_level_values_are_not_treated_as_frame_features(self):
        quality = assess_feature_quality(self.features, self.manifest).set_index("Feature")
        self.assertEqual(quality.loc["trajectory_feature", "Resolution"], "trajectory")
        self.assertFalse(quality.loc["trajectory_feature", "Pareto_Eligible"])

    def test_unavailable_values_are_explicitly_unavailable(self):
        quality = assess_feature_quality(self.features, self.manifest).set_index("Feature")
        self.assertEqual(quality.loc["unavailable_feature", "Resolution"], "unavailable")
        self.assertEqual(quality.loc["unavailable_feature", "Valid_Fraction"], 0.0)

    def test_frame_feature_receives_quality_diagnostics(self):
        quality = assess_feature_quality(self.features, self.manifest).set_index("Feature")
        self.assertEqual(quality.loc["frame_feature", "Resolution"], "frame")
        self.assertTrue(np.isfinite(quality.loc["frame_feature", "ESS_Ratio"]))
        self.assertTrue(np.isfinite(quality.loc["frame_feature", "Replica_Consistency"]))

    def test_block_uncertainty_and_measured_cost_are_reported_explicitly(self):
        costs = pd.DataFrame({"Feature": ["frame_feature"], "Elapsed_Seconds": [1.25]})
        quality = assess_feature_quality(
            self.features, self.manifest, PrioritizationConfig(block_size_frames=4), feature_costs=costs,
        ).set_index("Feature")
        self.assertTrue(np.isfinite(quality.loc["frame_feature", "Median_Block_SE"]))
        self.assertEqual(quality.loc["frame_feature", "Cost_Status"], "measured")
        self.assertEqual(quality.loc["frame_feature", "Measured_Cost_Seconds"], 1.25)
        self.assertEqual(quality.loc["trajectory_feature", "Cost_Status"], "not_provided")

    def test_pareto_membership_frequency_uses_trajectory_resampling(self):
        quality = assess_feature_quality(
            self.features, self.manifest,
            PrioritizationConfig(block_size_frames=4, pareto_bootstrap_samples=5, random_state=7),
        ).set_index("Feature")
        frequency = quality.loc["frame_feature", "Pareto_Membership_Frequency"]
        self.assertGreaterEqual(frequency, 0.0)
        self.assertLessEqual(frequency, 1.0)


if __name__ == "__main__":
    unittest.main()
