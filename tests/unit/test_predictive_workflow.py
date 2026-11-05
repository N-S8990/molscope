import unittest

import numpy as np
import pandas as pd

from molscope.ml_workflow import MLWorkflowConfig, run_grouped_ml_workflow


class MLWorkflowTests(unittest.TestCase):
    def setUp(self):
        rows = []
        feature_rows = []
        for label, prefix in [(0, "control"), (1, "perturbed")]:
            for replica in range(3):
                trajectory = f"{prefix}-{replica}"
                for frame in range(12):
                    rows.append({"trajectory_id": trajectory, "condition": label})
                    feature_rows.append({"signal": label + (frame % 2) * 0.03, "noise": (frame + replica) % 3})
        self.manifest = pd.DataFrame(rows)
        self.features = pd.DataFrame(feature_rows)

    def test_grouped_lofo_reports_fold_stability(self):
        outputs = run_grouped_ml_workflow(
            self.features,
            self.manifest,
            "condition",
            MLWorkflowConfig(n_splits=3, rf_estimators=30, bootstrap_samples=20),
            feature_groups={"signal": "signal_group", "noise": "noise_group"},
        )
        importance = outputs["lofo_importance"].set_index("Feature_Group")
        ranking = outputs["predictive_feature_ranking"].set_index("Feature_Group")
        self.assertEqual(len(outputs["outer_fold_metrics"]), 6)
        self.assertIn("Trajectory_Mean_Accuracy_Drop_Mean", importance.columns)
        self.assertGreaterEqual(importance.loc["signal_group", "Positive_Drop_Frequency"], importance.loc["noise_group", "Positive_Drop_Frequency"])
        self.assertEqual(ranking.loc["signal_group", "Ranking_Evidence"], "trajectory_grouped_LOFO_random_forest")
        self.assertEqual(ranking.loc["signal_group", "Interpretation"], "predictive_association_not_causality")

    def test_unequal_trajectory_lengths_preserve_classes_and_groups(self):
        indices = []
        for index, trajectory in enumerate(self.manifest["trajectory_id"].unique()):
            rows = self.manifest.index[self.manifest["trajectory_id"] == trajectory].tolist()
            indices.extend(rows * (20 if index == 0 else 1))
        manifest = self.manifest.loc[indices].reset_index(drop=True)
        features = self.features.loc[indices].reset_index(drop=True)
        outputs = run_grouped_ml_workflow(
            features, manifest, "condition",
            MLWorkflowConfig(rf_estimators=5, bootstrap_samples=5), run_lofo=False,
        )
        assignments = outputs["fold_assignments"]
        self.assertEqual(len(assignments), 6)
        self.assertTrue(assignments["trajectory_id"].is_unique)
        self.assertTrue((assignments.groupby("Test_Fold")["Target"].nunique() == 2).all())
        self.assertTrue(outputs["outer_fold_metrics"]["ROC_AUC"].notna().all())

    def test_missing_trajectory_identity_is_rejected(self):
        manifest = self.manifest.copy()
        manifest.loc[0, "trajectory_id"] = None
        with self.assertRaisesRegex(ValueError, "trajectory_id contains missing"):
            run_grouped_ml_workflow(self.features, manifest, "condition")

    def test_duplicate_feature_names_are_rejected(self):
        features = self.features.copy()
        features.columns = ["signal", "signal"]
        with self.assertRaisesRegex(ValueError, "Feature names must be unique"):
            run_grouped_ml_workflow(features, self.manifest, "condition")


if __name__ == "__main__":
    unittest.main()
