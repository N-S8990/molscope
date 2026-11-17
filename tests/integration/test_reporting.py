import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from molscope.reporting import ReportInputs, build_research_report, render_report_figures
from molscope.reporting.research import _largest_absolute_accuracy_changes, write_result_guide


class ReportingTests(unittest.TestCase):
    def test_predictive_figures_select_both_directions_of_accuracy_change(self):
        ranking = pd.DataFrame({
            "Feature_Group": [f"positive_{index}" for index in range(16)] + ["negative_large"],
            "Trajectory_Mean_Accuracy_Drop_Mean": [0.01 + index * 0.005 for index in range(16)] + [-0.25],
        })
        selected = _largest_absolute_accuracy_changes(ranking)
        self.assertEqual(len(selected), 15)
        self.assertIn("negative_large", selected["Feature_Group"].tolist())
        self.assertNotIn("positive_0", selected["Feature_Group"].tolist())

    def test_report_separates_target_free_and_predictive_evidence(self):
        markdown, summary = build_research_report(ReportInputs(
            quality=pd.DataFrame({"Resolution": ["frame", "unavailable"]}),
            sam_mfp=pd.DataFrame({"Feature": ["contact"], "SAM_MFP_Pareto_Front": [True], "PCA_Salience": [0.8], "Representation_Stability": [0.9], "Replica_Consistency": [0.7], "Uniqueness": [0.4]}),
            state_diagnostics=pd.DataFrame({"State_Count": [2], "Trajectory_Count": [3], "Silhouette": [0.5], "Restart_ARI_Median": [0.8], "State_Population_JSD_Median": [0.1]}),
            supervised_summary=pd.DataFrame({"Model": ["random_forest"], "Trajectory_Mean_Accuracy_Mean": [0.7]}),
            predictive_ranking=pd.DataFrame({"Rank": [1], "Feature_Group": ["contact"], "Trajectory_Mean_Accuracy_Drop_Mean": [0.2], "Positive_Drop_Frequency": [1.0]}),
            state_explanatory_ranking=pd.DataFrame({"Rank": [1], "Feature": ["distance"], "State_Explained_Variance_Eta_Squared": [0.7], "Max_State_Mean_Difference": [2.0]}),
        ))
        self.assertIn("Target-free prioritized", markdown)
        self.assertIn("Predictive ML", markdown)
        self.assertIn("Predictive feature ranking", markdown)
        self.assertIn("Candidate-state explanatory ranking", markdown)
        self.assertIn("not scientific causality", markdown)
        self.assertEqual(summary["sam_mfp_pareto_candidate_count"], 1)

    def test_writes_prioritization_and_state_population_pngs(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = render_report_figures(
                pd.DataFrame({"PCA_Salience": [0.5, 0.8], "Replica_Consistency": [0.7, 0.9], "SAM_MFP_Pareto_Front": [False, True]}),
                pd.DataFrame({"trajectory_id": ["t1", "t1", "t2", "t2"], "State": [0, 1, 0, 1], "Population": [0.7, 0.3, 0.4, 0.6]}),
                temporary,
            )
            self.assertEqual(set(paths), {"sam_mfp_tradeoff.png", "state_populations.png"})
            self.assertTrue((Path(temporary) / "sam_mfp_tradeoff.png").is_file())

    def test_writes_sam_mfp_tradeoff_with_ess_ratio_fallback_when_replica_consistency_is_all_nan(self):
        with tempfile.TemporaryDirectory() as temporary:
            sam_mfp = pd.DataFrame({
                "Feature": ["f1", "f2", "f3"],
                "PCA_Salience": [0.1, 0.4, 0.8],
                "Replica_Consistency": [np.nan, np.nan, np.nan],
                "ESS_Ratio": [0.2, 0.5, 0.9],
                "SAM_MFP_Pareto_Front": [False, True, True],
                "SAM_MFP_Eligible": [True, True, True],
            })
            paths = render_report_figures(sam_mfp, None, temporary)
            self.assertIn("sam_mfp_tradeoff.png", paths)
            plot_file = Path(temporary) / "sam_mfp_tradeoff.png"
            self.assertTrue(plot_file.is_file())
            self.assertGreater(plot_file.stat().st_size, 1000)

    def test_writes_quality_and_representation_figures_when_tables_are_supplied(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = render_report_figures(
                None, None, temporary,
                quality=pd.DataFrame({
                    "Valid_Fraction": [1.0, 0.8], "Replica_Consistency": [0.9, 0.5], "Pareto_Front": [True, False],
                }),
                representation_salience=pd.DataFrame({"Feature": ["contact", "distance"], "PCA_Salience": [0.8, 0.4]}),
            )
            self.assertEqual(set(paths), {"feature_quality_tradeoff.png", "representation_salience.png"})
            self.assertTrue((Path(temporary) / "representation_salience.png").is_file())

    def test_writes_descriptive_predictive_and_state_ranking_figures(self):
        with tempfile.TemporaryDirectory() as temporary:
            paths = render_report_figures(
                None, None, temporary,
                predictive_ranking=pd.DataFrame({
                    "Feature_Group": ["contact", "distance"],
                    "Trajectory_Mean_Accuracy_Drop_Mean": [0.2, -0.1],
                }),
                state_explanatory_ranking=pd.DataFrame({
                    "Feature": ["contact", "distance"],
                    "State_Explained_Variance_Eta_Squared": [0.7, 0.4],
                }),
            )
            self.assertEqual(set(paths), {"predictive_feature_ranking.png", "state_explanatory_feature_ranking.png"})
            self.assertTrue((Path(temporary) / "predictive_feature_ranking.png").is_file())

    def test_writes_density_correlation_accuracy_and_reader_guide(self):
        with tempfile.TemporaryDirectory() as temporary:
            features = pd.DataFrame({"a": [0, 1, 2, 3], "b": [3, 2, 1, 0], "c": [0, 1, 0, 1]})
            salience = pd.DataFrame({"Feature": ["a", "b", "c"], "PCA_Salience": [0.8, 0.7, 0.2]})
            coordinates = pd.DataFrame({"PC1": [0, 1, 2, 3], "PC2": [0, 1, 0, 1], "PC3": [1, 0, 1, 0]})
            predictive = pd.DataFrame({"Feature_Group": ["a"], "Trajectory_Mean_Accuracy_Drop_Mean": [0.2]})
            summary = pd.DataFrame({"Model": ["random_forest"], "Trajectory_Mean_Accuracy_Mean": [0.8]})
            paths = render_report_figures(None, None, temporary, features=features,
                                          representation_salience=salience,
                                          representation_coordinates=coordinates,
                                          predictive_ranking=predictive,
                                          supervised_model_summary=summary)
            expected = {"feature_correlation_map.png", "pca_density_2d.png",
                        "pca_density_3d.png", "predictive_accuracy_gain_drop.png"}
            self.assertTrue(expected.issubset(paths))
            write_result_guide(temporary, {"sam_mfp_candidates": pd.DataFrame({
                "Feature": ["a", "b"], "SAM_MFP_Pareto_Front": [True, False],
            }), "supervised_predictive_feature_ranking": predictive}, paths)
            self.assertTrue((Path(temporary) / "feature_guide.csv").is_file())
            feature_guide = pd.read_csv(Path(temporary) / "feature_guide.csv")
            self.assertIn("Evidence_Summary", feature_guide.columns)
            self.assertIn("Suggested_Next_Check", feature_guide.columns)
            guide = (Path(temporary) / "START_HERE.md").read_text()
            self.assertIn("pca_density_3d.png", guide)
            self.assertIn("not physical x/y/z", guide)


if __name__ == "__main__":
    unittest.main()
