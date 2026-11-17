"""Unit tests for statistical ML rigor: stability, validity, convergence, and leakage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from molscope.analysis.predictive.workflow import (
    MLWorkflowConfig,
    purged_temporal_block_splits,
    run_grouped_ml_workflow,
)
from molscope.analysis.quality.statistical_rigor import (
    check_simulation_convergence,
    compute_clustering_validity_indices,
    evaluate_bootstrap_cluster_stability,
    evaluate_feature_ranking_stability,
    evaluate_group_bootstrap_uncertainty,
    evaluate_permutation_null_control,
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

        # Explicitly verify ZERO frame leakage between train and test across all folds
        out_preds = outputs["out_of_fold_predictions"]
        for fold in [1, 2]:
            fold_test_rows = set(out_preds[out_preds["Fold"] == fold]["Frame_Row"].unique())
            other_test_rows = set(out_preds[out_preds["Fold"] != fold]["Frame_Row"].unique())
            # Test frames for this fold must be completely disjoint from test frames of other folds
            assert fold_test_rows.isdisjoint(other_test_rows)
            assert len(fold_test_rows & other_test_rows) == 0

            # Verify trajectory partition: train trajectories and test trajectories have zero overlap
            test_trajs = set(manifest.iloc[list(fold_test_rows)]["trajectory_id"].unique())
            train_trajs = set(manifest.iloc[list(other_test_rows)]["trajectory_id"].unique())
            assert test_trajs.isdisjoint(train_trajs)
            assert len(test_trajs & train_trajs) == 0

        # Verify dummy baseline is included in models
        models_evaluated = set(out_preds["Model"].unique())
        assert "dummy" in models_evaluated
        assert "logistic_regression" in models_evaluated
        assert "random_forest" in models_evaluated


class TestPurgedTemporalBlockCrossValidation:
    def test_purged_block_splits_strict_zero_leakage_and_embargo(self):
        # 120 frames in a single trajectory with 2 alternating/switching states
        n_frames = 120
        y = np.array([0 if (i // 20) % 2 == 0 else 1 for i in range(n_frames)])
        metadata = pd.DataFrame({
            "trajectory_id": ["traj_1"] * n_frames,
            "frame_index": np.arange(n_frames),
        })

        embargo = 3
        splits = purged_temporal_block_splits(
            metadata, y, n_splits=3, random_state=42, chunk_size=20, embargo=embargo
        )
        assert len(splits) == 3

        for fold_idx, (train_frames, test_frames) in enumerate(splits):
            assert len(train_frames) > 0
            assert len(test_frames) > 0

            # 1. STRICT ZERO FRAME LEAKAGE: train and test indices must be 100% disjoint
            train_set = set(train_frames)
            test_set = set(test_frames)
            assert train_set.isdisjoint(test_set), f"Fold {fold_idx} has frame leakage!"
            assert len(train_set & test_set) == 0

            # 2. STRICT EMBARGO MARGIN: no train frame may be within +/- embargo of any test frame
            for tf in test_frames:
                for offset in range(-embargo, embargo + 1):
                    neighbor = tf + offset
                    assert neighbor not in train_set, (
                        f"Fold {fold_idx}: neighbor {neighbor} within embargo distance {offset} "
                        f"of test frame {tf} is present in train frames!"
                    )

    def test_within_trajectory_workflow_purged_block_cv_execution(self):
        # Conformational state transitions within trajectories (cannot group by trajectory)
        rows = []
        feature_rows = []
        for traj_idx in range(2):
            for frame in range(60):
                # State 0 for first 30 frames, State 1 for next 30 frames
                state = 0 if frame < 30 else 1
                rows.append({"trajectory_id": f"traj_{traj_idx}", "frame_index": frame, "state": state})
                feature_rows.append({"coord_1": state * 3.0 + np.sin(frame * 0.1), "noise": frame % 5})

        manifest = pd.DataFrame(rows)
        features = pd.DataFrame(feature_rows)

        config = MLWorkflowConfig(
            n_splits=3,
            rf_estimators=10,
            bootstrap_samples=10,
            block_chunk_size=15,
            block_embargo=2,
            include_baseline_dummy=True,
        )
        outputs = run_grouped_ml_workflow(features, manifest, "state", config=config, run_lofo=False)

        assert "outer_fold_metrics" in outputs
        assert len(outputs["outer_fold_metrics"]) > 0
        assignments = outputs["fold_assignments"]
        assert len(assignments) == len(manifest)

        # Check zero frame leakage across fold assignments
        out_preds = outputs["out_of_fold_predictions"]
        for fold in [1, 2, 3]:
            fold_frames = set(out_preds[out_preds["Fold"] == fold]["Frame_Row"].unique())
            other_frames = set(out_preds[out_preds["Fold"] != fold]["Frame_Row"].unique())
            assert fold_frames.isdisjoint(other_frames)
            assert len(fold_frames & other_frames) == 0


class TestRankingStability:
    def test_ranking_stability_across_seeds_and_replicas(self):
        rows = []
        feat_rows = []
        rng = np.random.default_rng(42)
        for label, prefix in [(0, "ctrl"), (1, "pert")]:
            for rep in range(4):
                traj_id = f"{prefix}_{rep}"
                for frame in range(12):
                    rows.append({"trajectory_id": traj_id, "condition": label})
                    # Feature 1 has decisive signal, Feature 2 has modest signal, Feature 3 is pure noise
                    feat_rows.append({
                        "decisive_signal": label * 5.0 + rng.normal(0, 0.2),
                        "modest_signal": label * 1.5 + rng.normal(0, 0.5),
                        "pure_noise": rng.normal(0, 1.0),
                    })
        manifest = pd.DataFrame(rows)
        features = pd.DataFrame(feat_rows)

        out_seed1 = run_grouped_ml_workflow(
            features, manifest, "condition",
            MLWorkflowConfig(n_splits=2, random_state=42, rf_estimators=30, bootstrap_samples=10),
            run_lofo=True,
        )
        out_seed2 = run_grouped_ml_workflow(
            features, manifest, "condition",
            MLWorkflowConfig(n_splits=2, random_state=2024, rf_estimators=30, bootstrap_samples=10),
            run_lofo=True,
        )

        rankings = [out_seed1["predictive_feature_ranking"], out_seed2["predictive_feature_ranking"]]
        res = evaluate_feature_ranking_stability(rankings, top_k=1)

        assert res["n_rankings"] == 2
        assert res["mean_spearman_rho"] >= 0.50
        assert res["mean_kendall_tau"] >= 0.33
        assert res["is_stable"] is True
        summary = res["feature_stability_summary"].set_index("Feature")
        # decisive_signal should have the lowest (best) mean rank (rank 1)
        assert summary.loc["decisive_signal", "Mean_Rank"] == 1.0
        assert summary.loc["decisive_signal", "Top_1_Frequency"] == 1.0

    def test_ranking_stability_requires_at_least_two_rankings(self):
        with pytest.raises(ValueError, match="At least two rankings"):
            evaluate_feature_ranking_stability([["feat1", "feat2"]])

    def test_ranking_stability_with_string_lists(self):
        rank1 = ["f1", "f2", "f3", "f4", "f5"]
        rank2 = ["f1", "f2", "f3", "f5", "f4"]
        res = evaluate_feature_ranking_stability([rank1, rank2], top_k=3)
        assert res["mean_spearman_rho"] >= 0.85
        assert res["mean_kendall_tau"] >= 0.75
        assert res["mean_top_k_jaccard"] == 1.0


class TestGroupBootstrapUncertainty:
    def test_group_bootstrap_uncertainty_resamples_trajectories(self):
        rows = []
        feat_rows = []
        for label, prefix in [(0, "ctrl"), (1, "pert")]:
            for rep in range(3):
                traj_id = f"{prefix}_{rep}"
                for frame in range(10):
                    rows.append({"trajectory_id": traj_id, "label": label})
                    feat_rows.append({"f1": label * 2.0 + frame * 0.05, "f2": frame % 3})
        manifest = pd.DataFrame(rows)
        features = pd.DataFrame(feat_rows)

        config = MLWorkflowConfig(n_splits=2, rf_estimators=10, bootstrap_samples=5)
        res = evaluate_group_bootstrap_uncertainty(
            features, manifest, "label", config=config, n_bootstraps=5, random_state=42
        )
        assert res["n_successful"] >= 3
        summary = res["bootstrap_summary"].set_index("Metric")
        assert "Balanced_Accuracy" in summary.index
        assert "Frame_Accuracy" in summary.index
        bal_row = summary.loc["Balanced_Accuracy"]
        assert 0.0 <= bal_row["CI_Lower"] <= bal_row["Mean"] <= bal_row["CI_Upper"] <= 1.0
        assert np.isfinite(bal_row["Std_Error"])

    def test_group_bootstrap_requires_minimum_bootstraps_and_groups(self):
        df = pd.DataFrame({"trajectory_id": ["t1"] * 10, "y": [0] * 5 + [1] * 5})
        feat = pd.DataFrame({"x": np.arange(10)})
        with pytest.raises(ValueError, match="at least 2 independent groups"):
            evaluate_group_bootstrap_uncertainty(feat, df, "y", n_bootstraps=5)


class TestPermutationNegativeControl:
    def test_permutation_null_scores_near_chance(self):
        # 8 trajectories (4 control, 4 perturbed)
        rows = []
        feat_rows = []
        rng = np.random.default_rng(42)
        for label, prefix in [(0, "ctrl"), (1, "pert")]:
            for rep in range(4):
                traj_id = f"{prefix}_{rep}"
                for frame in range(12):
                    rows.append({"trajectory_id": traj_id, "condition": label})
                    # Decisive signal separating control and perturbed
                    feat_rows.append({"signal": label * 10.0 + rng.normal(0, 0.1), "noise": rng.normal(0, 1.0)})
        manifest = pd.DataFrame(rows)
        features = pd.DataFrame(feat_rows)

        # Confirm real data achieves high accuracy
        real_res = run_grouped_ml_workflow(
            features, manifest, "condition",
            MLWorkflowConfig(n_splits=2, rf_estimators=15, bootstrap_samples=5),
            run_lofo=False,
        )
        real_bal_acc = float(
            real_res["model_summary"].loc[real_res["model_summary"]["Model"] == "random_forest", "Balanced_Accuracy_Mean"].iloc[0]
        )
        assert real_bal_acc > 0.90, f"Expected high real accuracy, got {real_bal_acc}"

        # Run permutation negative control
        null_res = evaluate_permutation_null_control(
            features,
            manifest,
            "condition",
            config=MLWorkflowConfig(n_splits=2, rf_estimators=15, bootstrap_samples=5),
            n_permutations=5,
            random_state=42,
            group_level=True,
        )

        assert null_res["shuffling_mode"] == "trajectory_level"
        assert null_res["is_chance_consistent"] is True
        # Under null hypothesis with shuffled labels, accuracy collapses to chance (~0.50)
        assert 0.30 <= null_res["mean_balanced_accuracy"] <= 0.70
        assert real_bal_acc > null_res["mean_balanced_accuracy"] + 0.20
