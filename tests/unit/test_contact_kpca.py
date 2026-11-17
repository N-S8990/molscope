"""Unit tests for Contact Kernel PCA (CKPCA) conformational manifold learning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

from molscope.analysis.discovery.contact_kpca import (
    ContactKPCAConfig,
    ContactKPCAResult,
    _compute_feature_salience,
    _estimate_gamma,
    _prepare_contact_matrix,
    fit_contact_kpca,
)
from molscope.api import Study

FIXTURE_PDB = Path(__file__).parents[1] / "fixtures" / "1A22.pdb"


def _generate_synthetic_contact_system(
    n_frames: int = 120,
    n_contacts: int = 15,
    seed: int = 42,
    binary: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic continuous or binary contact features with 2 distinct conformational basins."""
    rng = np.random.default_rng(seed)
    n_half = n_frames // 2

    # Basin A (frames 0..n_half-1): compact contacts (short distances / active contacts)
    # Basin B (frames n_half..n_frames-1): extended contacts (long distances / inactive contacts)
    if binary:
        # State A has contacts 0..4 formed (1), contacts 5..9 absent (0)
        # State B has contacts 0..4 absent (0), contacts 5..9 formed (1)
        # contacts 10..14 are random noise
        basin_a = rng.binomial(1, p=0.85, size=(n_half, 5))
        basin_b = rng.binomial(1, p=0.15, size=(n_frames - n_half, 5))
        c_0_4 = np.vstack([basin_a, basin_b])

        basin_a_2 = rng.binomial(1, p=0.15, size=(n_half, 5))
        basin_b_2 = rng.binomial(1, p=0.85, size=(n_frames - n_half, 5))
        c_5_9 = np.vstack([basin_a_2, basin_b_2])

        noise = rng.binomial(1, p=0.5, size=(n_frames, n_contacts - 10))
        data = np.hstack([c_0_4, c_5_9, noise]).astype(float)
    else:
        # Continuous contact distances (in Angstroms)
        basin_a_dist = rng.normal(loc=4.5, scale=0.4, size=(n_half, 5))
        basin_b_dist = rng.normal(loc=11.0, scale=0.6, size=(n_frames - n_half, 5))
        c_0_4 = np.vstack([basin_a_dist, basin_b_dist])

        basin_a_dist2 = rng.normal(loc=12.0, scale=0.5, size=(n_half, 5))
        basin_b_dist2 = rng.normal(loc=4.2, scale=0.4, size=(n_frames - n_half, 5))
        c_5_9 = np.vstack([basin_a_dist2, basin_b_dist2])

        noise = rng.normal(loc=8.0, scale=1.5, size=(n_frames, n_contacts - 10))
        data = np.hstack([c_0_4, c_5_9, noise])

    feature_cols = [f"contact_pair_{i:02d}" for i in range(n_contacts)]
    features_df = pd.DataFrame(data, columns=feature_cols)

    manifest_df = pd.DataFrame({
        "trajectory_id": ["traj_01"] * n_half + ["traj_02"] * (n_frames - n_half),
        "replica_id": ["rep_01"] * n_half + ["rep_02"] * (n_frames - n_half),
        "ensemble_id": "ens_default",
        "frame_index": list(range(n_frames)),
        "frame_order": list(range(n_frames)),
    })

    return features_df, manifest_df


class TestContactKPCAConfig:
    def test_default_config(self):
        config = ContactKPCAConfig()
        assert config.n_components == 4
        assert config.gamma is None
        assert config.kernel == "rbf"
        assert config.center is True
        assert config.drop_constant is True

    def test_config_validation(self):
        with pytest.raises(ValueError, match="n_components must be a positive integer"):
            ContactKPCAConfig(n_components=0)
        with pytest.raises(ValueError, match="gamma must be positive"):
            ContactKPCAConfig(gamma=-0.5)
        with pytest.raises(ValueError, match="gamma heuristic must be"):
            ContactKPCAConfig(gamma="invalid_heuristic")
        with pytest.raises(TypeError, match="gamma must be"):
            ContactKPCAConfig(gamma=[1, 2, 3])
        with pytest.raises(ValueError, match="Unsupported kernel"):
            ContactKPCAConfig(kernel="polynomial")

    def test_to_dict(self):
        config = ContactKPCAConfig(n_components=3, gamma=0.05, kernel="exponential")
        d = config.to_dict()
        assert d["n_components"] == 3
        assert d["gamma"] == 0.05
        assert d["kernel"] == "exponential"


class TestContactKPCAMathematics:
    def test_median_heuristic_estimation(self):
        rng = np.random.default_rng(123)
        x = rng.normal(0, 1, size=(50, 6))
        sq_dists = pdist(x, metric="sqeuclidean")
        expected_gamma = 1.0 / (2.0 * float(np.median(sq_dists)))

        gamma_computed = _estimate_gamma(x, gamma=None, kernel="rbf")
        assert np.isclose(gamma_computed, expected_gamma)

        gamma_median = _estimate_gamma(x, gamma="median", kernel="rbf")
        assert np.isclose(gamma_median, expected_gamma)

    def test_scale_heuristic_estimation(self):
        rng = np.random.default_rng(123)
        x = rng.normal(0, 2.5, size=(40, 5))
        expected_gamma = 1.0 / (x.shape[1] * float(np.var(x)))

        gamma_scale = _estimate_gamma(x, gamma="scale", kernel="rbf")
        assert np.isclose(gamma_scale, expected_gamma)

    def test_user_defined_gamma(self):
        x = np.ones((10, 4))
        assert _estimate_gamma(x, gamma=0.42) == 0.42

    def test_kernel_centering_row_and_column_sums(self):
        """Centered kernel matrix K_c must have zero row and column sums."""
        features, manifest = _generate_synthetic_contact_system(n_frames=60, seed=42)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=3, gamma=0.01))

        # Reconstruct centered kernel K_c: K_c = eigenvectors * lambda * eigenvectors.T
        # Row sums of K_c should be zero within numerical precision:
        # Sum_j K_c(i, j) = 0 for all i
        k = res.training_features.shape[0]
        sq_dists = squareform(pdist(res.training_features, metric="sqeuclidean"))
        k_mat = np.exp(-res.gamma * sq_dists)
        r_means = np.mean(k_mat, axis=1)
        g_mean = float(np.mean(k_mat))
        k_c = k_mat - r_means[:, None] - r_means[None, :] + g_mean

        row_sums = np.sum(k_c, axis=1)
        col_sums = np.sum(k_c, axis=0)
        assert np.allclose(row_sums, 0.0, atol=1e-10)
        assert np.allclose(col_sums, 0.0, atol=1e-10)

    def test_eigenvalue_sorting_and_orthogonality(self):
        """Eigenvalues must be descending and eigenvectors must be orthonormal."""
        features, manifest = _generate_synthetic_contact_system(n_frames=80, seed=42)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=4))

        # Descending positive eigenvalues
        assert (np.diff(res.eigenvalues) <= 0).all()
        assert (res.eigenvalues > 0).all()

        # Eigenvectors orthonormal: V^T V = I
        v = res.eigenvectors
        vtv = v.T @ v
        assert np.allclose(vtv, np.eye(res.n_components), atol=1e-10)

    def test_out_of_sample_projection_matches_training_coordinates(self):
        """Projecting the training set via transform() must match fitted coordinates."""
        features, manifest = _generate_synthetic_contact_system(n_frames=50, seed=99)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=3, gamma=0.02))

        projected = res.transform(features)
        coord_cols = [f"CKPCA_{i + 1}" for i in range(3)]
        training_coords = res.coordinates[coord_cols].to_numpy()

        assert np.allclose(projected.to_numpy(), training_coords, atol=1e-10)

    def test_transform_unseen_data(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=60, seed=1)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=2))

        # Unseen new test frames
        new_features, _ = _generate_synthetic_contact_system(n_frames=10, seed=2)
        projected = res.transform(new_features)

        assert projected.shape == (10, 2)
        assert list(projected.columns) == ["CKPCA_1", "CKPCA_2"]
        assert np.isfinite(projected.to_numpy()).all()


class TestContactKPCAConformationalBasins:
    def test_separates_two_distinct_conformational_basins_continuous(self):
        """CKPCA_1 should cleanly separate the two conformational basins for distance contacts."""
        features, manifest = _generate_synthetic_contact_system(n_frames=100, binary=False, seed=42)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=2))

        ckpca_1 = res.coordinates["CKPCA_1"].to_numpy()
        basin_a = ckpca_1[:50]
        basin_b = ckpca_1[50:]

        # Mean separation between basin A and basin B should be clear
        mean_diff = abs(float(np.mean(basin_a)) - float(np.mean(basin_b)))
        pooled_std = float(np.sqrt(0.5 * (np.var(basin_a) + np.var(basin_b))))
        # Effect size (Cohen's d) > 3.0 indicates unambiguous separation
        assert mean_diff / pooled_std > 3.0

    def test_binary_contact_matrix(self):
        """CKPCA handles binary (0/1) contact presence matrices effectively."""
        features, manifest = _generate_synthetic_contact_system(n_frames=100, binary=True, seed=42)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=3, kernel="exponential"))

        assert res.n_components == 3
        assert len(res.coordinates) == 100
        assert "CKPCA_1" in res.coordinates.columns
        assert "CKPCA_2" in res.coordinates.columns
        assert res.kernel == "exponential"

        # Check basin separation
        ckpca_1 = res.coordinates["CKPCA_1"].to_numpy()
        mean_a = float(np.mean(ckpca_1[:50]))
        mean_b = float(np.mean(ckpca_1[50:]))
        assert abs(mean_a - mean_b) > 0.5


class TestContactKPCASalienceAndOutputs:
    def test_salience_ranks_discriminative_contacts_highest(self):
        """Contacts 0..9 that distinguish the basins must have higher salience than noise contacts 10..14."""
        features, manifest = _generate_synthetic_contact_system(n_frames=120, n_contacts=15, seed=42)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=2))

        salience_df = res.salience.set_index("Feature")
        discriminative_contacts = [f"contact_pair_{i:02d}" for i in range(10)]
        noise_contacts = [f"contact_pair_{i:02d}" for i in range(10, 15)]

        mean_disc_salience = salience_df.loc[discriminative_contacts, "CKPCA_Salience"].mean()
        mean_noise_salience = salience_df.loc[noise_contacts, "CKPCA_Salience"].mean()

        assert mean_disc_salience > mean_noise_salience

    def test_result_structure_and_dict_access(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=40, seed=7)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=3))

        # Dict access and keys
        assert res["coordinates"] is res.coordinates
        assert res["eigenvalues"] is res.eigenvalues
        assert res["salience"] is res.salience
        assert res["components"] is not None
        assert res.get("nonexistent", "fallback") == "fallback"

        with pytest.raises(KeyError):
            _ = res["invalid_key"]

        # Components table
        comps = res.components
        assert list(comps.columns) == ["Component", "Eigenvalue", "Explained_Variance_Ratio", "Cumulative_Variance_Ratio"]
        assert len(comps) == 3
        assert np.isclose(comps["Cumulative_Variance_Ratio"].iloc[-1], comps["Explained_Variance_Ratio"].sum())

    def test_manifest_metadata_preservation(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=40, seed=7)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=2))

        assert "trajectory_id" in res.coordinates.columns
        assert "replica_id" in res.coordinates.columns
        assert "frame_order" in res.coordinates.columns
        assert "CKPCA_1" in res.coordinates.columns
        assert "CKPCA_2" in res.coordinates.columns


class TestContactKPCAValidationAndEdgeCases:
    def test_manifest_length_mismatch_raises(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=40)
        with pytest.raises(ValueError):
            fit_contact_kpca(features, manifest.iloc[:20])

    def test_non_finite_values_raise(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=40)
        bad_features = features.copy()
        bad_features.iloc[5, 2] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            fit_contact_kpca(bad_features, manifest)

    def test_empty_features_raise(self):
        with pytest.raises(ValueError, match="empty"):
            fit_contact_kpca(pd.DataFrame())

    def test_constant_features_filtering(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=40)
        # Add a constant contact column
        features["constant_contact"] = 5.0
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=2, drop_constant=True))

        assert "constant_contact" not in res.feature_names
        assert res.n_components == 2

    def test_all_constant_features_raise(self):
        features = pd.DataFrame(np.ones((20, 5)))
        with pytest.raises(ValueError, match="zero variance"):
            fit_contact_kpca(features, config=ContactKPCAConfig(drop_constant=True))

    def test_single_component_request(self):
        features, manifest = _generate_synthetic_contact_system(n_frames=30)
        res = fit_contact_kpca(features, manifest, ContactKPCAConfig(n_components=1))
        assert res.n_components == 1
        assert "CKPCA_1" in res.coordinates.columns
        assert "CKPCA_2" not in res.coordinates.columns


class TestContactKPCAStudyIntegration:
    def test_study_run_contact_kpca_with_synthetic_study(self, tmp_path):
        import MDAnalysis as mda

        # Create multi-frame PDB trajectory
        fixture = Path(__file__).parents[1] / "fixtures" / "solvated_protein.pdb"
        u = mda.Universe(str(fixture))
        multi_pdb = tmp_path / "multi_traj.pdb"
        with mda.Writer(str(multi_pdb), len(u.atoms)) as w:
            for _ in range(12):
                u.atoms.positions += np.random.normal(0, 0.05, size=u.atoms.positions.shape)
                w.write(u)

        study = Study(project_id="test_ckpca_study", out_dir=tmp_path / "study_out")
        study.add_system(system_id="sys1", topology=fixture, trajectories=[multi_pdb])
        study.configure_features(
            ["protein.ca_distances_contacts"],
            plugin_parameters={"protein.ca_distances_contacts": {"selection": "resid 1:5", "max_pairs": 15}},
        )

        res = study.run_contact_kpca(n_components=2)
        assert isinstance(res, ContactKPCAResult)
        assert res is study.get_contact_kpca_result()
        assert "CKPCA_1" in res.coordinates.columns
        assert "CKPCA_2" in res.coordinates.columns
        assert len(res.coordinates) == 12

    def test_study_run_contact_kpca_without_systems_raises(self):
        study = Study(project_id="empty_study")
        with pytest.raises(RuntimeError, match="Run the study or configure systems"):
            study.run_contact_kpca()

    def test_study_get_contact_kpca_before_run_raises(self):
        study = Study(project_id="unrun_study")
        with pytest.raises(RuntimeError, match="Run Contact Kernel PCA with study.run_contact_kpca"):
            study.get_contact_kpca_result()
