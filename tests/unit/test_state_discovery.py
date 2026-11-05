import unittest

import numpy as np
import pandas as pd

from molscope.states import StateDiscoveryConfig, discover_conformational_states, evaluate_state_count_sensitivity


class StateDiscoveryTests(unittest.TestCase):
    def test_reports_assignments_and_per_trajectory_populations(self):
        coordinates, identities = [], []
        for trajectory in range(3):
            for value in np.r_[-2 + np.linspace(0, 0.2, 6), 2 + np.linspace(0, 0.2, 6)]:
                identity = {"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e", "frame_order": len(coordinates)}
                coordinates.append({"PC1": value, "PC2": value * 0.05, **identity})
                identities.append(identity)
        outputs = discover_conformational_states(pd.DataFrame(coordinates), pd.DataFrame(identities), StateDiscoveryConfig(n_states=2, n_restarts=3))
        self.assertEqual(len(outputs["state_assignments"]), 36)
        self.assertEqual(len(outputs["state_populations"]), 6)
        self.assertAlmostEqual(outputs["state_populations"].groupby("trajectory_id")["Population"].sum().iloc[0], 1.0)
        self.assertGreaterEqual(outputs["state_diagnostics"].loc[0, "Initialization_ARI_Median"], 0.0)
        transitions = outputs["aggregate_state_transitions"]
        self.assertEqual(len(transitions), 4)
        self.assertTrue((transitions["Transition_Count"] >= 0).all())

    def test_state_count_sensitivity_reports_multiple_candidates(self):
        coordinates = pd.DataFrame([
            {"trajectory_id": f"t{trajectory}", "replica_id": f"r{trajectory}", "ensemble_id": "e", "frame_order": frame, "PC1": (-1 if frame < 4 else 1) + trajectory * 0.1}
            for trajectory in range(3) for frame in range(8)
        ])
        manifest = coordinates[["trajectory_id", "replica_id", "ensemble_id", "frame_order"]].copy()
        sensitivity = evaluate_state_count_sensitivity(coordinates, manifest, (2, 3), n_restarts=3)
        self.assertEqual(sensitivity["State_Count"].tolist(), [2, 3])


if __name__ == "__main__":
    unittest.main()
