import unittest
import tempfile
from pathlib import Path

import pandas as pd

from molscope.reporting import ReportInputs, build_research_report, render_report_figures


class ReportingTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
