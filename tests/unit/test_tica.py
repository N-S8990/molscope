"""Unit tests for Time-lagged Independent Component Analysis (TICA)."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molscope.analysis.discovery.states import discover_conformational_states
from molscope.analysis.discovery.tica import TICAConfig, TICAResult, fit_tica
from molscope.api import Study

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def _generate_synthetic_kinetic_system(
    n_frames: int = 4000,
    p_trans: float = 0.05,
    seed: int = 42,
    n_trajectories: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic trajectories with a known slow two-state kinetic process."""
    rng = np.random.default_rng(seed)
    frames_per_traj = n_frames // n_trajectories

    all_features = []
    all_manifest = []

    for traj_idx in range(n_trajectories):
        state = 0
        states = []
        for _ in range(frames_per_traj):
            if state == 0 and rng.random() < p_trans:
                state = 1
            elif state == 1 and rng.random() < p_trans:
                state = 0
            states.append(state)

        states = np.asarray(states, dtype=float)
        # Feature 1: slow kinetic coordinate (tracks 2-state process with small noise)
        f_slow = (states * 4.0 - 2.0) + rng.normal(0, 0.1, size=frames_per_traj)
        # Feature 2: fast independent white noise with large variance
        f_fast = rng.normal(0, 5.0, size=frames_per_traj)
        # Feature 3: intermediate noise
        f_mid = rng.normal(0, 1.0, size=frames_per_traj)

        traj_features = pd.DataFrame({
            "slow_coordinate": f_slow,
            "fast_noise": f_fast,
            "mid_noise": f_mid,
        })
        traj_manifest = pd.DataFrame({
            "trajectory_id": f"traj_{traj_idx:02d}",
            "replica_id": f"rep_{traj_idx:02d}",
            "ensemble_id": "ens_default",
            "frame_index": list(range(frames_per_traj)),
            "frame_order": list(range(frames_per_traj)),
            "time_ps": [float(i * 10.0) for i in range(frames_per_traj)],
        })

        all_features.append(traj_features)
        all_manifest.append(traj_manifest)

    features_df = pd.concat(all_features, ignore_index=True)
    manifest_df = pd.concat(all_manifest, ignore_index=True)
    return features_df, manifest_df


class TestTICAConfig:
    def test_default_config(self):
        config = TICAConfig()
        assert config.lag_time == 1
        assert config.n_components == 2
        assert config.shrinkage == 1e-4
        assert config.dt == 1.0

    def test_config_validation(self):
        with pytest.raises(ValueError, match="lag_time must be a positive integer"):
            TICAConfig(lag_time=0)
        with pytest.raises(ValueError, match="lag_time must be a positive integer"):
            TICAConfig(lag_time=-1)
        with pytest.raises(ValueError, match="n_components must be a positive integer"):
            TICAConfig(n_components=0)
        with pytest.raises(ValueError, match="shrinkage must be non-negative"):
            TICAConfig(shrinkage=-0.1)
        with pytest.raises(ValueError, match="dt must be positive"):
            TICAConfig(dt=0.0)

    def test_config_to_dict(self):
        config = TICAConfig(lag_time=5, n_components=3, shrinkage=1e-3, dt=2.5)
        d = config.to_dict()
        assert d == {"lag_time": 5, "n_components": 3, "shrinkage": 1e-3, "dt": 2.5}


class TestTICAKinetics:
    def test_identifies_slow_kinetic_process_over_high_variance_noise(self):
        """TICA should discover the slow kinetic transition as TIC 1, whereas PCA

        would mistakenly focus on the fast noise due to larger variance.
        """
        features, manifest = _generate_synthetic_kinetic_system(n_frames=5000, p_trans=0.05, seed=42)
        config = TICAConfig(lag_time=1, n_components=2, shrinkage=1e-4, dt=1.0)
        result = fit_tica(features, manifest, config)

        # 1. Eigenvalues should be sorted descending
        assert result.eigenvalues[0] > result.eigenvalues[1]
        # Theoretical autocorrelation for 2-state jump with p_trans=0.05 is ~ 1 - 2*0.05 = 0.90
        assert result.eigenvalues[0] > 0.80
        # Fast noise has near-zero autocorrelation
        assert abs(result.eigenvalues[1]) < 0.15

        # 2. Implied timescales: slowest process should have timescale ~ -1 / ln(0.9) ~ 9.5
        assert result.implied_timescales[0] > 5.0
        assert result.implied_timescales[1] < 1.0 or np.isnan(result.implied_timescales[1])

        # 3. Kinetic Salience: slow_coordinate must be ranked #1
        loadings = result.loadings.set_index("Feature")
        assert loadings.loc["slow_coordinate", "Rank"] == 1
        assert loadings.loc["slow_coordinate", "Kinetic_Salience"] > loadings.loc["fast_noise", "Kinetic_Salience"]

        # 4. Projected coordinates
        assert "TIC_1" in result.coordinates.columns
        assert "TIC_2" in result.coordinates.columns
        assert len(result.coordinates) == len(features)

    def test_multi_trajectory_isolation_strictly_prevents_cross_contamination(self):
        """Pairs (x_t, x_{t+tau}) must strictly remain WITHIN individual continuous

        trajectories. Frames at the boundary between replicas must never be paired.
        """
        # Create 2 independent replicas where replica 1 is in state +10 and replica 2 is in state -10
        N = 100
        traj1_feat = pd.DataFrame({
            "coord": np.ones(N) * 10.0 + np.random.normal(0, 0.01, N),
            "noise": np.random.normal(0, 0.1, N),
        })
        traj2_feat = pd.DataFrame({
            "coord": np.ones(N) * -10.0 + np.random.normal(0, 0.01, N),
            "noise": np.random.normal(0, 0.1, N),
        })
        features = pd.concat([traj1_feat, traj2_feat], ignore_index=True)

        manifest = pd.DataFrame({
            "trajectory_id": ["traj_1"] * N + ["traj_2"] * N,
            "replica_id": ["rep_1"] * N + ["rep_2"] * N,
            "ensemble_id": ["ens_0"] * (2 * N),
            "frame_order": list(range(N)) + list(range(N)),
            "frame_index": list(range(N)) + list(range(N)),
        })

        config = TICAConfig(lag_time=1, n_components=1)
        result = fit_tica(features, manifest, config)

        # In both trajectories, coord is nearly static at +10 and -10.
        # Within each trajectory: x_{t+1} - x_t ~ 0 (autocorrelation is ~ 1.0).
        # If cross-contamination occurred at the boundary (frame 99 -> 100),
        # an artificial transition from +10 to -10 (-20 step) would dilute the covariance.
        assert result.c_tau is not None
        # Verify both eigenvalues and coordinates
        assert result.eigenvalues[0] > 0.90
        assert len(result.coordinates) == 2 * N

    def test_short_trajectories_handled_without_leakage(self):
        """A trajectory shorter than lag_time provides 0 pairs and must not crash."""
        features = pd.DataFrame({
            "f1": np.linspace(0, 10, 20),
            "f2": np.random.normal(0, 1, 20),
        })
        manifest = pd.DataFrame({
            "trajectory_id": ["long_traj"] * 18 + ["short_traj"] * 2,
            "replica_id": ["r1"] * 18 + ["r2"] * 2,
            "ensemble_id": ["e1"] * 20,
            "frame_order": list(range(18)) + list(range(2)),
        })
        # lag_time=3: short_traj has length 2 <= 3, contributes 0 pairs
        config = TICAConfig(lag_time=3, n_components=1)
        result = fit_tica(features, manifest, config)
        assert len(result.coordinates) == 20
        assert "TIC_1" in result.coordinates.columns

    def test_lag_time_too_large_raises_value_error(self):
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        manifest = pd.DataFrame({
            "trajectory_id": ["t1", "t1", "t1"],
            "replica_id": ["r1", "r1", "r1"],
            "ensemble_id": ["e1", "e1", "e1"],
        })
        config = TICAConfig(lag_time=5)
        with pytest.raises(ValueError, match="is too large"):
            fit_tica(features, manifest, config)

    def test_frame_ordering_and_interleaving_invariance(self):
        """Interleaved rows in manifest/features must be properly reconstructed

        according to frame_order, yielding identical covariance matrices.
        """
        features, manifest = _generate_synthetic_kinetic_system(n_frames=200, n_trajectories=2, seed=123)
        config = TICAConfig(lag_time=2, n_components=2)

        # Baseline sequential fit
        res_sequential = fit_tica(features, manifest, config)

        # Shuffle rows
        perm = np.random.default_rng(99).permutation(len(features))
        shuffled_features = features.iloc[perm].reset_index(drop=True)
        shuffled_manifest = manifest.iloc[perm].reset_index(drop=True)

        res_shuffled = fit_tica(shuffled_features, shuffled_manifest, config)

        # Eigenvalues and covariances must match within numerical tolerance
        np.testing.assert_allclose(res_sequential.eigenvalues, res_shuffled.eigenvalues, rtol=1e-5)
        np.testing.assert_allclose(res_sequential.c0, res_shuffled.c0, rtol=1e-5)
        np.testing.assert_allclose(res_sequential.c_tau, res_shuffled.c_tau, rtol=1e-5)

    def test_implied_timescales_dt_scaling(self):
        """Implied timescale t_i = - (tau * dt) / ln |lambda_i| must scale linearly with dt."""
        features, manifest = _generate_synthetic_kinetic_system(n_frames=1000, seed=42)
        res1 = fit_tica(features, manifest, TICAConfig(lag_time=1, dt=1.0))
        res2 = fit_tica(features, manifest, TICAConfig(lag_time=1, dt=5.0))

        valid = (res1.eigenvalues > 0) & (res1.eigenvalues < 1.0)
        assert np.any(valid)
        np.testing.assert_allclose(
            res2.implied_timescales[valid],
            res1.implied_timescales[valid] * 5.0,
            rtol=1e-6,
        )

    def test_shrinkage_regularization_handles_collinear_features(self):
        """Collinear features create a singular instantaneous covariance C_0.

        Shrinkage regularization must ensure positive-definiteness and stable solving.
        """
        base = np.random.normal(0, 1, 500)
        features = pd.DataFrame({
            "col1": base,
            "col2": base * 2.0,  # Perfectly collinear
            "col3": np.random.normal(0, 1, 500),
        })
        manifest = pd.DataFrame({
            "trajectory_id": ["t1"] * 500,
            "replica_id": ["r1"] * 500,
            "ensemble_id": ["e1"] * 500,
            "frame_order": list(range(500)),
        })
        # With default shrinkage 1e-4, should fit stably without error
        result = fit_tica(features, manifest, TICAConfig(shrinkage=1e-4, n_components=2))
        assert len(result.eigenvalues) == 2
        assert np.all(np.isfinite(result.eigenvalues))

    def test_out_of_sample_transform(self):
        """TICAResult.transform projects new feature frames consistently."""
        features, manifest = _generate_synthetic_kinetic_system(n_frames=400, seed=42)
        train_feat = features.iloc[:300]
        train_man = manifest.iloc[:300]
        test_feat = features.iloc[300:].reset_index(drop=True)

        result = fit_tica(train_feat, train_man, TICAConfig(n_components=2))
        test_proj = result.transform(test_feat)

        assert isinstance(test_proj, pd.DataFrame)
        assert list(test_proj.columns) == ["TIC_1", "TIC_2"]
        assert len(test_proj) == 100

        # Missing feature in transform should raise KeyError
        with pytest.raises(KeyError, match="missing required columns"):
            result.transform(test_feat.drop(columns=["slow_coordinate"]))


class TestTICAValidationAndErrors:
    def test_non_finite_features_raise(self):
        features = pd.DataFrame({"f1": [1.0, np.nan, 3.0], "f2": [4.0, 5.0, 6.0]})
        manifest = pd.DataFrame({
            "trajectory_id": ["t1"] * 3,
            "replica_id": ["r1"] * 3,
            "ensemble_id": ["e1"] * 3,
        })
        with pytest.raises(ValueError, match="finite feature values"):
            fit_tica(features, manifest)

    def test_all_constant_features_raise(self):
        features = pd.DataFrame({"f1": [2.0, 2.0, 2.0], "f2": [5.0, 5.0, 5.0]})
        manifest = pd.DataFrame({
            "trajectory_id": ["t1"] * 3,
            "replica_id": ["r1"] * 3,
            "ensemble_id": ["e1"] * 3,
        })
        with pytest.raises(ValueError, match="at least one non-constant feature"):
            fit_tica(features, manifest)

    def test_drops_constant_feature_keeps_active(self):
        features = pd.DataFrame({
            "constant": [1.0] * 100,
            "active": np.random.normal(0, 1, 100),
        })
        manifest = pd.DataFrame({
            "trajectory_id": ["t1"] * 100,
            "replica_id": ["r1"] * 100,
            "ensemble_id": ["e1"] * 100,
            "frame_order": list(range(100)),
        })
        result = fit_tica(features, manifest, TICAConfig(n_components=1))
        assert "active" in result.feature_names
        assert "constant" not in result.feature_names
        assert len(result.loadings) == 1

    def test_duplicate_frame_order_raises(self):
        features = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
        manifest = pd.DataFrame({
            "trajectory_id": ["t1", "t1", "t1"],
            "replica_id": ["r1", "r1", "r1"],
            "ensemble_id": ["e1", "e1", "e1"],
            "frame_order": [0, 0, 1],  # Duplicate 0
        })
        with pytest.raises(ValueError, match="Duplicate frame_order"):
            fit_tica(features, manifest)

    def test_manifest_mismatch_raises(self):
        features = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0]})
        manifest = pd.DataFrame({
            "trajectory_id": ["t1"],
            "replica_id": ["r1"],
            "ensemble_id": ["e1"],
        })
        with pytest.raises(ValueError, match="Feature matrix has 2 rows but manifest has 1 rows"):
            fit_tica(features, manifest)


class TestTICAResultInterface:
    def test_dict_and_property_access(self):
        features, manifest = _generate_synthetic_kinetic_system(n_frames=200, seed=1)
        result = fit_tica(features, manifest, TICAConfig(n_components=2))

        # Property access
        assert isinstance(result.coordinates, pd.DataFrame)
        assert isinstance(result.eigenvalues, np.ndarray)
        assert isinstance(result.implied_timescales, np.ndarray)
        assert isinstance(result.loadings, pd.DataFrame)
        assert isinstance(result.eigenvectors, np.ndarray)
        assert len(result.eigenvalues) == 2
        assert len(result.implied_timescales) == 2
        assert result.eigenvectors.shape == (3, 2)

        # Dict access
        assert result["coordinates"] is result.coordinates
        assert result["eigenvalues"] is result.eigenvalues
        assert result["loadings"] is result.loadings
        assert result.get("nonexistent", "fallback") == "fallback"

        # Components summary
        comps = result.components
        assert list(comps.columns) == ["Component", "Eigenvalue", "Implied_Timescale"]
        assert len(comps) == 2

    def test_integration_with_state_discovery(self):
        """State discovery should seamlessly accept TICA coordinates."""
        features, manifest = _generate_synthetic_kinetic_system(n_frames=500, seed=42)
        tica_res = fit_tica(features, manifest, TICAConfig(n_components=2))

        states_res = discover_conformational_states(tica_res.coordinates, manifest)
        assert "state_assignments" in states_res
        assert "state_populations" in states_res
        assert "State" in states_res["state_assignments"].columns


class TestStudyAPIIntegration:
    def test_study_run_tica(self, tmp_path):
        import MDAnalysis as mda

        # Create a small multi-frame trajectory using solvated_protein.pdb
        fixture = Path(__file__).parents[1] / "fixtures" / "solvated_protein.pdb"
        u = mda.Universe(str(fixture))
        multi_pdb = tmp_path / "multi_traj.pdb"
        with mda.Writer(str(multi_pdb), len(u.atoms)) as w:
            for _ in range(15):
                u.atoms.positions += np.random.normal(0, 0.05, size=u.atoms.positions.shape)
                w.write(u)

        study = Study(project_id="test_tica_study", out_dir=tmp_path / "study_out")
        study.add_system(
            system_id="sys1",
            topology=fixture,
            trajectories=[multi_pdb],
        )

        # Run TICA directly (will run extract_only automatically)
        tica_res = study.run_tica(lag_time=1, components=2)

        assert isinstance(tica_res, TICAResult)
        assert tica_res is study.get_tica_result()
        assert "TIC_1" in tica_res.coordinates.columns
        assert len(tica_res.eigenvalues) <= 2
        assert not tica_res.loadings.empty
        assert len(tica_res.coordinates) == 15

    def test_study_run_tica_without_systems_raises(self):
        study = Study(project_id="empty_study")
        with pytest.raises(RuntimeError, match="Run the study or configure systems"):
            study.run_tica()
