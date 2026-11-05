import unittest

import pandas as pd

from molscope.sam_mfp import prioritize_sam_mfp


class SAMMFPTests(unittest.TestCase):
    def test_returns_pareto_candidates_without_a_scalar_importance_score(self):
        quality = pd.DataFrame({
            "Feature": ["stable_informative", "unstable"], "Resolution": ["frame", "frame"],
            "Valid_Fraction": [1.0, 1.0], "ESS_Ratio": [0.9, 0.8],
            "Replica_Consistency": [0.9, 0.2], "Occupancy_Entropy": [0.8, 0.7], "Uniqueness": [0.8, 0.7],
        })
        discovery = pd.DataFrame({
            "Feature": ["stable_informative", "unstable"], "PCA_Salience": [0.8, 0.4],
            "LOTO_Rank_Std": [0.2, 20.0], "LOTO_Rank_Std_Normalized": [0.02, 0.9], "LOTO_Folds": [3, 3],
        })
        result, notes = prioritize_sam_mfp(quality, discovery)
        indexed = result.set_index("Feature")
        self.assertTrue(indexed.loc["stable_informative", "SAM_MFP_Pareto_Front"])
        self.assertFalse(indexed.loc["unstable", "SAM_MFP_Pareto_Front"])
        self.assertNotIn("Importance_Score", result.columns)
        self.assertTrue(any("not a scalar" in note for note in notes))


if __name__ == "__main__":
    unittest.main()
