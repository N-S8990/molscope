import unittest

import pandas as pd

from molscope.energetics import align_energy_to_frames, validate_energy_table


class EnergeticsTests(unittest.TestCase):
    def test_exact_alignment_preserves_supplied_energy_values(self):
        manifest = pd.DataFrame({"trajectory_id": ["t1", "t1"], "time_ps": [0.0, 2.0]})
        energies = pd.DataFrame({"trajectory_id": ["t1", "t1"], "time_ps": [0.0, 2.0], "potential_kj_mol": [-10.0, -12.0]})
        aligned = align_energy_to_frames(manifest, energies)
        self.assertEqual(aligned["potential_kj_mol"].tolist(), [-10.0, -12.0])

    def test_missing_energy_rows_are_not_imputed(self):
        manifest = pd.DataFrame({"trajectory_id": ["t1", "t1"], "time_ps": [0.0, 2.0]})
        energies = pd.DataFrame({"trajectory_id": ["t1"], "time_ps": [0.0], "potential_kj_mol": [-10.0]})
        with self.assertRaisesRegex(ValueError, "not imputed"):
            align_energy_to_frames(manifest, energies)

    def test_nonfinite_energy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_energy_table(pd.DataFrame({"trajectory_id": ["t1"], "time_ps": [0.0], "potential_kj_mol": [float("nan")]}))
