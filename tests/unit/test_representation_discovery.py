import unittest

import numpy as np
import pandas as pd

from molscope.discovery import RepresentationDiscoveryConfig, discover_representation_features


class RepresentationDiscoveryTests(unittest.TestCase):
    def test_reports_target_free_salience_and_trajectory_stability(self):
        values = []
        identities = []
        rng = np.random.default_rng(4)
        for trajectory in range(4):
            signal = np.linspace(-2, 2, 12) + trajectory * 0.05
            values.extend(
                {"signal": item, "signal_partner": item + rng.normal(0, 0.02), "small_noise": rng.normal()}
                for item in signal
            )
            identities.extend({"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e"} for _ in signal)
        outputs = discover_representation_features(
            pd.DataFrame(values), pd.DataFrame(identities), RepresentationDiscoveryConfig(n_components=1),
        )
        salience = outputs["representation_feature_salience"].set_index("Feature")
        self.assertGreater(salience.loc["signal", "PCA_Salience"], salience.loc["small_noise", "PCA_Salience"])
        self.assertEqual(salience.loc["signal", "LOTO_Folds"], 4)
        self.assertGreaterEqual(salience.loc["signal", "LOTO_Rank_Std_Normalized"], 0.0)
        self.assertLessEqual(salience.loc["signal", "LOTO_Rank_Std_Normalized"], 1.0)
        self.assertEqual(len(outputs["representation_coordinates"]), 48)


if __name__ == "__main__":
    unittest.main()
