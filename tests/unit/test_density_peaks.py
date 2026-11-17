"""Unit tests for Density Peak Advanced (DPA) Clustering."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import cdist

from molscope.analysis.discovery.contact_kpca import ContactKPCAConfig, fit_contact_kpca
from molscope.analysis.discovery.density_peaks import (
    DPAConfig,
    DPAResult,
    _compute_transitions,
    _estimate_cutoff_distance,
    _extract_coordinate_matrix,
    cluster_dpa,
)
from molscope.analysis.discovery.tica import TICAConfig, fit_tica
from molscope.api import Study

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def _generate_synthetic_multibasin_system(
    n_per_basin: int = 60,
    separation: float = 8.0,
    seed: int = 42,
    n_trajectories: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic 2D coordinates with 3 clearly separated Gaussian conformational basins."""
    rng = np.random.default_rng(seed)

    # Basin 0: centered at (-separation, 0)
    basin_0 = rng.normal(loc=[-separation, 0.0], scale=0.6, size=(n_per_basin, 2))
    # Basin 1: centered at (separation, 0)
    basin_1 = rng.normal(loc=[separation, 0.0], scale=0.6, size=(n_per_basin, 2))
    # Basin 2: centered at (0, separation * 1.2)
    basin_2 = rng.normal(loc=[0.0, separation * 1.2], scale=0.6, size=(n_per_basin, 2))

    coords = np.vstack([basin_0, basin_1, basin_2])
    n_total = len(coords)

    coords_df = pd.DataFrame(coords, columns=["PC1", "PC2"])

    # Trajectory manifest
    frames_per_traj = n_total // n_trajectories
    traj_ids = []
    for t in range(n_trajectories):
        traj_ids.extend([f"traj_{t:02d}"] * frames_per_traj)
    if len(traj_ids) < n_total:
        traj_ids.extend([f"traj_{n_trajectories - 1:02d}"] * (n_total - len(traj_ids)))

    manifest_df = pd.DataFrame({
        "trajectory_id": traj_ids,
        "replica_id": [f"rep_{i % n_trajectories:02d}" for i in range(n_total)],
        "ensemble_id": "ens_default",
        "frame_index": list(range(n_total)),
        "frame_order": list(range(n_total)),
    })

    return coords_df, manifest_df


class TestDPAConfig:
    def test_default_config(self):
        config = DPAConfig()
        assert config.cutoff_distance is None
        assert config.percentile == 2.0
        assert config.z_score == 2.0
        assert config.n_clusters is None
        assert config.min_clusters == 1
        assert config.metric == "euclidean"

    def test_config_validation(self):
        with pytest.raises(ValueError, match="cutoff_distance must be positive"):
            DPAConfig(cutoff_distance=0.0)
        with pytest.raises(ValueError, match="percentile must be in"):
            DPAConfig(percentile=0.0)
        with pytest.raises(ValueError, match="percentile must be in"):
            DPAConfig(percentile=100.0)
        with pytest.raises(ValueError, match="z_score must be non-negative"):
            DPAConfig(z_score=-1.0)
        with pytest.raises(ValueError, match="n_clusters must be >= 1"):
            DPAConfig(n_clusters=0)
        with pytest.raises(ValueError, match="min_clusters must be >= 1"):
            DPAConfig(min_clusters=0)

    def test_config_to_dict(self):
        config = DPAConfig(cutoff_distance=1.5, percentile=3.0, z_score=2.5, n_clusters=3)
        d = config.to_dict()
        assert d["cutoff_distance"] == 1.5
        assert d["percentile"] == 3.0
        assert d["z_score"] == 2.5
        assert d["n_clusters"] == 3


class TestDPAMathematicsAndMechanics:
    def test_local_density_peaks_at_basin_centers(self):
        """Points near the center of a dense cluster must have higher rho than peripheral points."""
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=50, seed=42)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=3))

        # Check that centers identified correspond to peak densities
        assert len(res.cluster_centers) == 3
        for center_idx in res.cluster_centers:
            center_rho = res.rho[center_idx]
            # Center rho should be higher than the average rho of the dataset
            assert center_rho > np.mean(res.rho)

    def test_higher_density_distance_delta_definition(self):
        """For global density maximum, delta = max(d_ij). For all others, delta = min_{j: rho_j > rho_i} d_ij."""
        coords = np.array([
            [0.0, 0.0],  # Peak point
            [0.1, 0.0],  # Close to peak
            [0.2, 0.0],  # Further from peak
            [5.0, 0.0],  # Distant point
        ])
        d_mat = cdist(coords, coords)
        res = cluster_dpa(coords, config=DPAConfig(cutoff_distance=1.0, min_clusters=1))

        peak_idx = int(np.argmax(res.rho))
        # Peak delta must equal maximum distance from peak to any other point
        assert np.isclose(res.delta[peak_idx], np.max(d_mat[peak_idx]))

        # For every non-peak point, delta must strictly equal min distance to a higher density frame
        for i in range(len(coords)):
            if i != peak_idx:
                higher_mask = res.rho > res.rho[i]
                if np.any(higher_mask):
                    expected_delta = np.min(d_mat[i, higher_mask])
                    assert np.isclose(res.delta[i], expected_delta)

    def test_gamma_decision_metric_and_z_score_outlier_detection(self):
        """Cluster centers must have anomalously high gamma = rho * delta."""
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=60, separation=10.0, seed=42)
        res = cluster_dpa(coords_df, config=DPAConfig(z_score=1.5))

        # 3 basins should be detected via z_score >= 1.5
        assert res.n_clusters == 3
        # Centers must have high Z-scores
        for c in res.cluster_centers:
            assert res.z_scores[c] >= 1.5

    def test_cluster_assignment_propagation(self):
        """Non-center frames should propagate to the nearest higher-density neighbor cluster."""
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=40, separation=8.0, seed=1)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=3))

        # Basin 0: frames 0..39
        # Basin 1: frames 40..79
        # Basin 2: frames 80..119
        labels_basin_0 = res.assignments["Cluster"].iloc[0:40].to_numpy()
        labels_basin_1 = res.assignments["Cluster"].iloc[40:80].to_numpy()
        labels_basin_2 = res.assignments["Cluster"].iloc[80:120].to_numpy()

        # All frames within each well-separated basin should belong to a single unique cluster
        assert len(np.unique(labels_basin_0)) == 1
        assert len(np.unique(labels_basin_1)) == 1
        assert len(np.unique(labels_basin_2)) == 1

        # The 3 clusters must be mutually distinct
        cluster_set = {labels_basin_0[0], labels_basin_1[0], labels_basin_2[0]}
        assert len(cluster_set) == 3


class TestDPACoordinateTypes:
    def test_clusters_pca_coordinates(self):
        coords_df, manifest = _generate_synthetic_multibasin_system(n_per_basin=40, seed=42)
        res = cluster_dpa(coords_df, manifest=manifest, coordinate_prefix="PCA")

        assert "PC1" in res.coordinate_columns
        assert "PC2" in res.coordinate_columns
        assert res.n_clusters >= 2
        assert "trajectory_id" in res.assignments.columns

    def test_clusters_ckpca_coordinates(self):
        # Generate contact features and run CKPCA
        rng = np.random.default_rng(42)
        n_frames = 90
        # 3 states in contact space
        state_a = rng.normal(4.0, 0.3, size=(30, 8))
        state_b = rng.normal(8.0, 0.4, size=(30, 8))
        state_c = rng.normal(12.0, 0.3, size=(30, 8))
        contacts = pd.DataFrame(np.vstack([state_a, state_b, state_c]), columns=[f"contact_{i}" for i in range(8)])

        ckpca_res = fit_contact_kpca(contacts, config=ContactKPCAConfig(n_components=3))
        dpa_res = cluster_dpa(ckpca_res.coordinates, coordinate_prefix="CKPCA", config=DPAConfig(n_clusters=3))

        assert "CKPCA_1" in dpa_res.coordinate_columns
        assert dpa_res.n_clusters == 3
        # Check that the 3 contact states are partitioned accurately
        labels = dpa_res.assignments["Cluster"].to_numpy()
        assert len(np.unique(labels[:30])) == 1
        assert len(np.unique(labels[30:60])) == 1
        assert len(np.unique(labels[60:90])) == 1
        assert len(set(labels)) == 3

    def test_clusters_tica_coordinates(self):
        # Generate synthetic coordinates with TIC naming
        coords_df = pd.DataFrame({
            "TIC_1": np.concatenate([np.random.normal(-3, 0.3, 40), np.random.normal(3, 0.3, 40)]),
            "TIC_2": np.random.normal(0, 0.5, 80),
        })
        dpa_res = cluster_dpa(coords_df, coordinate_prefix="TICA", config=DPAConfig(n_clusters=2))

        assert "TIC_1" in dpa_res.coordinate_columns
        assert dpa_res.n_clusters == 2
        assert len(dpa_res.assignments) == 80


class TestDPADecisionGraphAndTransitions:
    def test_decision_graph_table_structure(self):
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=30, seed=42)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=3))

        dg = res.decision_graph
        expected_cols = [
            "Frame_Index",
            "rho",
            "delta",
            "gamma",
            "Z_Score",
            "Is_Center",
            "Cluster",
            "Cluster_Label",
            "Nearest_Higher_Density_Neighbor",
            "Cluster_Role",
        ]
        assert list(dg.columns) == expected_cols
        assert dg["Is_Center"].sum() == 3
        assert set(dg["Cluster_Role"].unique()).issubset({"core", "halo"})

    def test_core_and_halo_classification(self):
        """Cluster centers must always be classified as 'core'."""
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=50, seed=42)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=3))

        centers = res.cluster_centers
        roles = res.assignments.loc[centers, "Cluster_Role"].to_list()
        assert all(role == "core" for role in roles)

    def test_per_trajectory_transitions_and_populations(self):
        coords_df, manifest = _generate_synthetic_multibasin_system(n_per_basin=40, seed=42, n_trajectories=2)
        res = cluster_dpa(coords_df, manifest=manifest, config=DPAConfig(n_clusters=3))

        pops = res.populations
        assert not pops.empty
        assert "trajectory_id" in pops.columns
        assert "Population" in pops.columns

        # Sum of populations per trajectory must equal 1.0
        for traj_id, subset in pops.groupby("trajectory_id"):
            assert np.isclose(subset["Population"].sum(), 1.0)

        # Transition tables
        agg_trans = res.aggregate_transitions
        assert not agg_trans.empty
        assert "Transition_Count" in agg_trans.columns
        assert "Conditional_Probability" in agg_trans.columns

    def test_out_of_sample_predict(self):
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=50, separation=6.0, seed=42)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=3))

        # Query points located directly at the basin centroids
        center_coords = res.training_coordinates[res.cluster_centers]
        predicted_centers = res.predict(center_coords)
        assert np.array_equal(predicted_centers, np.arange(3))

        # New query point near center 0
        new_pt = center_coords[0:1] + 0.1
        assert res.predict(new_pt)[0] == 0


class TestDPAValidationAndEdgeCases:
    def test_manifest_length_mismatch_raises(self):
        coords_df, manifest = _generate_synthetic_multibasin_system(n_per_basin=30)
        with pytest.raises(ValueError):
            cluster_dpa(coords_df, manifest=manifest.iloc[:10])

    def test_non_finite_values_raise(self):
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=30)
        coords_df.iloc[0, 0] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            cluster_dpa(coords_df)

    def test_empty_coordinates_raise(self):
        with pytest.raises(ValueError, match="empty"):
            cluster_dpa(pd.DataFrame())

    def test_high_z_score_fallback_to_min_clusters(self):
        """When z_score is too high for any frame to qualify, fallback to min_clusters."""
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=30)
        res = cluster_dpa(coords_df, config=DPAConfig(z_score=100.0, min_clusters=2))
        assert res.n_clusters == 2

    def test_dict_access_and_keys(self):
        coords_df, _ = _generate_synthetic_multibasin_system(n_per_basin=30)
        res = cluster_dpa(coords_df, config=DPAConfig(n_clusters=2))

        assert res["assignments"] is res.assignments
        assert res["cluster_centers"] is res.cluster_centers
        assert res["rho"] is res.rho
        assert res.get("nonexistent", "fallback") == "fallback"

        with pytest.raises(KeyError):
            _ = res["unknown_key"]


class TestDPAStudyIntegration:
    def test_study_cluster_dpa_on_pca(self, tmp_path):
        import MDAnalysis as mda

        # Create multi-frame PDB trajectory
        fixture = Path(__file__).parents[1] / "fixtures" / "solvated_protein.pdb"
        u = mda.Universe(str(fixture))
        multi_pdb = tmp_path / "multi_traj.pdb"
        with mda.Writer(str(multi_pdb), len(u.atoms)) as w:
            for _ in range(15):
                u.atoms.positions += np.random.normal(0, 0.05, size=u.atoms.positions.shape)
                w.write(u)

        study = Study(project_id="test_dpa_study", out_dir=tmp_path / "study_out")
        study.add_system(system_id="sys1", topology=fixture, trajectories=[multi_pdb])
        study.configure_features(
            ["protein.ca_distances_contacts"],
            plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:5", "max_pairs": 15}},
        )

        dpa_res = study.cluster_dpa(coordinate_name="PCA", z_score=1.0)
        assert isinstance(dpa_res, DPAResult)
        assert dpa_res is study.get_dpa_result()
        assert "Cluster" in dpa_res.assignments.columns
        assert len(dpa_res.assignments) == 15
        assert dpa_res.n_clusters >= 1

    def test_study_cluster_dpa_on_ckpca(self, tmp_path):
        import MDAnalysis as mda

        fixture = Path(__file__).parents[1] / "fixtures" / "solvated_protein.pdb"
        u = mda.Universe(str(fixture))
        multi_pdb = tmp_path / "multi_traj.pdb"
        with mda.Writer(str(multi_pdb), len(u.atoms)) as w:
            for _ in range(15):
                u.atoms.positions += np.random.normal(0, 0.05, size=u.atoms.positions.shape)
                w.write(u)

        study = Study(project_id="test_dpa_ckpca_study", out_dir=tmp_path / "study_out")
        study.add_system(system_id="sys1", topology=fixture, trajectories=[multi_pdb])
        study.configure_features(
            ["protein.ca_distances_contacts"],
            plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:5", "max_pairs": 15}},
        )

        # Run DPA on CKPCA (will automatically run CKPCA if not yet run)
        dpa_res = study.cluster_dpa(coordinate_name="CKPCA", z_score=1.0)
        assert isinstance(dpa_res, DPAResult)
        assert "CKPCA_1" in dpa_res.coordinate_columns
        assert study.get_contact_kpca_result() is not None

    def test_study_cluster_dpa_without_systems_raises(self):
        study = Study(project_id="empty_study")
        with pytest.raises(RuntimeError, match="Run the study or configure systems"):
            study.cluster_dpa(coordinate_name="PCA")

    def test_study_get_dpa_before_run_raises(self):
        study = Study(project_id="unrun_study")
        with pytest.raises(RuntimeError, match="Run DPA clustering with study.cluster_dpa"):
            study.get_dpa_result()
