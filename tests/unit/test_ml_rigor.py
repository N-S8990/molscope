"""Unit tests for statistical ML rigor: stability, validity, convergence, and leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow
from molscope.analysis.quality.statistical_rigor import (
    check_simulation_convergence,
    compute_clustering_validity_indices,
    evaluate_bootstrap_cluster_stability,
    evaluate_seed_stability,
)


def _generate_synthetic_clusters(
    n_samples: int = 300,
    n_features: int = 5,
    n_clusters: int = 3,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    samples_per_cluster = n_samples // n_clusters
    X_list = []
    y_list = []
    for k in range(n_clusters):
        center = np.zeros(n_features)
        center[k % n_features] = 10.0 * (k + 1)
        cluster_samples = rng.normal(loc=center, scale=1.0, size=(samples_per_cluster, n_features))
        X_list.append(cluster_samples)
        y_list.append(np.full(samples_per_cluster, k))
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    return X, y


class TestSeedStability:
    def test_seed_stability_well_separated_clusters(self):
        X, _ = _generate_synthetic_clusters(n_samples=300, n_clusters=3, seed=42)
        res = evaluate_seed_stability(X, n_clusters=3, seeds=(42, 100, 2024))
        assert "mean_ari" in res
        assert res["mean_ari"] > 0.90
        assert len(res["pairwise_comparisons"]) == 3
        assert (res["pairwise_comparisons"]["Adjusted_Rand_Index"] > 0.85).all()

    def test_seed_stability_dataframe_input(self):
        X, _ = _generate_synthetic_clusters(n_samples=200, n_clusters=2, seed=42)
        df = pd.DataFrame(X, columns=[f"PC{i+1}" for i in range(X.shape[1])])
        res = evaluate_seed_stability(df, n_clusters=2, seeds=(1, 2))
        assert res["mean_ari"] > 0.80

    def test_seed_stability_requires_multiple_seeds(self):
        X, _ = _generate_synthetic_clusters(n_samples=100, n_clusters=2)
        with pytest.raises(ValueError, match="At least two seeds"):
            evaluate_seed_stability(X, n_clusters=2, seeds=(42,))


class TestBootstrapClusterStability:
    def test_bootstrap_stability_tight_confidence_interval(self):
        X, _ = _generate_synthetic_clusters(n_samples=300, n_clusters=3, seed=42)
        res = evaluate_bootstrap_cluster_stability(X, n_clusters=3, n_bootstraps=15, random_state=42)
        assert res["mean_ari"] > 0.85
        assert 0.0 <= res["ci_lower"] <= res["ci_upper"] <= 1.0
        assert len(res["bootstrap_results"]) == 15

    def test_bootstrap_requires_minimum_bootstraps(self):
        X, _ = _generate_synthetic_clusters(n_samples=100, n_clusters=2)
        with pytest.raises(ValueError, match="n_bootstraps must be at least 2"):
            evaluate_bootstrap_cluster_stability(X, n_clusters=2, n_bootstraps=1)


class TestClusteringValidityIndices:
    def test_validity_indices_well_separated_clusters(self):
        X, true_labels = _generate_synthetic_clusters(n_samples=300, n_clusters=3, seed=42)
        indices = compute_clustering_validity_indices(X, true_labels)
        # Well-separated clusters should have high silhouette (> 0.5), high Calinski-Harabasz, low Davies-Bouldin (< 1.0)
        assert indices["silhouette_score"] > 0.50
        assert indices["calinski_harabasz_score"] > 100.0
        assert indices["davies_bouldin_index"] < 1.0
        assert indices["n_clusters"] == 3
        assert indices["n_samples"] == 300

    def test_validity_indices_mismatched_lengths_raise(self):
        X, true_labels = _generate_synthetic_clusters(n_samples=100, n_clusters=2)
        with pytest.raises(ValueError, match="Length of coordinates"):
            compute_clustering_validity_indices(X, true_labels[:-5])

    def test_validity_indices_single_cluster_raises(self):
        X, true_labels = _generate_synthetic_clusters(n_samples=100, n_clusters=2)
        with pytest.raises(ValueError, match="n_clusters"):
            compute_clustering_validity_indices(X, np.zeros(len(X), dtype=int))


class TestSimulationConvergence:
    def test_stationary_simulation_converged(self):
        rng = np.random.default_rng(42)
        # Stationary white noise / OU process
        t = np.linspace(0, 500, 500)
        pc1 = rng.normal(0, 1.0, 500)
        pc2 = rng.normal(0, 1.0, 500)
        df = pd.DataFrame({"time_ps": t, "PC1": pc1, "PC2": pc2})
        diag = check_simulation_convergence(df, min_effective_samples=15.0)
        assert diag["sampling_adequate"]
        assert diag["is_converged"]
        assert diag["effective_sample_size"] > 15.0
        assert not diag["flags"]

    def test_drifting_simulation_flags_unconverged(self):
        # Linear drift across trajectory (unconverged non-equilibrium)
        t = np.linspace(0, 500, 500)
        pc1 = np.linspace(0, 15, 500) + np.random.default_rng(42).normal(0, 0.2, 500)
        df = pd.DataFrame({"time_ps": t, "PC1": pc1})
        diag = check_simulation_convergence(df, drift_z_threshold=3.0)
        assert not diag["is_converged"]
        assert any("drift" in f.lower() for f in diag["flags"])


class TestGroupedCVLeakageAndBaselines:
    def test_grouped_cv_has_zero_trajectory_leakage(self):
        rows = []
        feature_rows = []
        for label, prefix in [(0, "ctrl"), (1, "pert")]:
            for rep in range(4):
                traj_id = f"{prefix}_{rep}"
                for frame in range(10):
                    rows.append({"trajectory_id": traj_id, "label": label})
                    feature_rows.append({"f1": label * 2.0 + frame * 0.01, "f2": rep * 0.5})
        manifest = pd.DataFrame(rows)
        features = pd.DataFrame(feature_rows)

        outputs = run_grouped_ml_workflow(
            features,
            manifest,
            "label",
            MLWorkflowConfig(n_splits=2, include_baseline_dummy=True, rf_estimators=10, bootstrap_samples=10),
        )
        fold_assignments = outputs["fold_assignments"]
        # Verify no trajectory appears in multiple test folds
        test_counts = fold_assignments.groupby("trajectory_id")["Test_Fold"].nunique()
        assert (test_counts == 1).all()

        # Verify dummy baseline is included in models
        out_preds = outputs["out_of_fold_predictions"]
        models_evaluated = set(out_preds["Model"].unique())
        assert "dummy" in models_evaluated
        assert "logistic_regression" in models_evaluated
        assert "random_forest" in models_evaluated
