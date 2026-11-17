"""Scientific rigor, baseline comparison, and ablation benchmark suite for SAM-MFP."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.prioritization.baselines import (
    compare_prioritization,
    compare_sam_mfp_with_baselines,
    compute_decision_tree_importance,
    compute_mutual_info,
    compute_permutation_importance,
    compute_random_ranking,
    compute_rf_importance,
    compute_variance_importance,
)
from molscope.analysis.quality.assessment import assess_feature_quality
from molscope.analysis.discovery.representation import discover_representation_features


def _generate_synthetic_benchmark(
    n_frames: int = 600,
    n_features: int = 20,
    n_planted: int = 3,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Generate synthetic trajectory features with planted discriminating signals and complementary Pareto profiles.

    Planted features have distinct multi-objective strengths:
      - feat_00: Highest representation salience and lowest computational cost.
      - feat_01: Highest replica consistency and effective sample size (ESS).
      - feat_02: Highest uniqueness and occupancy entropy.
    Noise features are strictly inferior across all objective dimensions.
    The supervised target y depends synergistically on the planted features.
    """
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]

    # Generate planted and noise feature trajectories
    data: dict[str, np.ndarray] = {}
    for i in range(n_planted):
        data[feature_names[i]] = rng.normal(0.0, 1.0, n_frames)
    for i in range(n_planted, n_features):
        data[feature_names[i]] = rng.normal(0.0, 1.0, n_frames)

    features = pd.DataFrame(data)

    # Continuous target that depends synergistically on all planted features
    weights = [2.5, 2.0, 1.5]
    y_raw = np.zeros(n_frames)
    for i in range(n_planted):
        y_raw += weights[i % len(weights)] * data[feature_names[i]]
    y = pd.Series(y_raw + rng.normal(0, 0.4, n_frames), name="Target")

    # Multi-criteria quality profiles
    quality_rows = []
    planted_profiles = [
        # feat_00: Moderate ESS (0.78), moderate rep (0.85), entropy (0.80), uniqueness (0.82), cheap (0.02s)
        {"ESS_Ratio": 0.78, "Replica_Consistency": 0.85, "Occupancy_Entropy": 0.80, "Uniqueness": 0.82, "Cost": 0.02},
        # feat_01: Maximum ESS (0.95), maximum replica consistency (0.98), moderate (0.08s)
        {"ESS_Ratio": 0.95, "Replica_Consistency": 0.98, "Occupancy_Entropy": 0.85, "Uniqueness": 0.80, "Cost": 0.08},
        # feat_02: Maximum uniqueness (0.96), maximum occupancy entropy (0.95), moderate (0.05s)
        {"ESS_Ratio": 0.80, "Replica_Consistency": 0.88, "Occupancy_Entropy": 0.95, "Uniqueness": 0.96, "Cost": 0.05},
    ]

    for i, name in enumerate(feature_names):
        if i < n_planted:
            prof = planted_profiles[i % len(planted_profiles)]
            quality_rows.append({
                "Feature": name,
                "Resolution": "frame",
                "Valid_Fraction": 1.0,
                "ESS_Ratio": prof["ESS_Ratio"],
                "Replica_Consistency": prof["Replica_Consistency"],
                "Occupancy_Entropy": prof["Occupancy_Entropy"],
                "Uniqueness": prof["Uniqueness"],
                "Measured_Cost_Seconds": prof["Cost"],
            })
        else:
            # Noise features have uniformly inferior quality metrics
            quality_rows.append({
                "Feature": name,
                "Resolution": "frame",
                "Valid_Fraction": 1.0,
                "ESS_Ratio": float(rng.uniform(0.15, 0.40)),
                "Replica_Consistency": float(rng.uniform(0.30, 0.55)),
                "Occupancy_Entropy": float(rng.uniform(0.35, 0.60)),
                "Uniqueness": float(rng.uniform(0.40, 0.65)),
                "Measured_Cost_Seconds": float(rng.uniform(0.20, 0.60)),
            })
    quality_df = pd.DataFrame(quality_rows)

    # Representation salience
    salience_rows = []
    salience_vals = [0.92, 0.70, 0.75]
    for i, name in enumerate(feature_names):
        if i < n_planted:
            salience_rows.append({
                "Feature": name,
                "PCA_Salience": salience_vals[i % len(salience_vals)],
                "LOTO_Rank_Std_Normalized": 0.02 * (i + 1),
                "LOTO_Folds": 3,
            })
        else:
            salience_rows.append({
                "Feature": name,
                "PCA_Salience": float(rng.uniform(0.02, 0.18)),
                "LOTO_Rank_Std_Normalized": float(rng.uniform(0.40, 0.75)),
                "LOTO_Folds": 3,
            })
    salience_df = pd.DataFrame(salience_rows)

    return features, quality_df, salience_df, y


def test_sam_mfp_recovers_planted_ground_truth_signal():
    """Verify that planted features with complementary multi-objective strengths reach the Pareto front."""
    features, quality, salience, y = _generate_synthetic_benchmark(n_planted=3)
    config = SAMMFPConfig()

    candidates, notes = prioritize_sam_mfp(quality, salience, config=config)

    assert not candidates.empty
    assert "SAM_MFP_Pareto_Front" in candidates.columns
    assert "SAM_MFP_Dominator_Count" in candidates.columns

    pareto_candidates = candidates.loc[candidates["SAM_MFP_Pareto_Front"], "Feature"].tolist()

    # All planted ground-truth features must be on the Pareto front
    for planted in ["feat_00", "feat_01", "feat_02"]:
        assert planted in pareto_candidates, f"Planted signal {planted} missing from Pareto front: {pareto_candidates}"

    # Dominator counts for planted features must be exactly 0
    planted_df = candidates.loc[candidates["Feature"].isin(["feat_00", "feat_01", "feat_02"])]
    assert (planted_df["SAM_MFP_Dominator_Count"] == 0).all()

    # Noise features must be strictly dominated
    noise_df = candidates.loc[
        candidates["Feature"].str.startswith("feat_")
        & ~candidates["Feature"].isin(["feat_00", "feat_01", "feat_02"])
    ]
    assert (noise_df["SAM_MFP_Dominator_Count"] > 0).all()
    assert not noise_df["SAM_MFP_Pareto_Front"].any()


def test_sam_mfp_baselines_comparisons():
    """Verify agreement and diagnostics between SAM-MFP and RF/DT/MI/Permutation baselines."""
    features, quality, salience, y = _generate_synthetic_benchmark(n_planted=3)

    # Compute individual baselines
    rf_df = compute_rf_importance(features, y, random_state=42)
    dt_df = compute_decision_tree_importance(features, y, random_state=42)
    mi_df = compute_mutual_info(features, y, random_state=42)
    perm_df = compute_permutation_importance(features, y, random_state=42)

    # Top features in each baseline must include the planted features
    assert set(rf_df.head(3)["Feature"]) == {"feat_00", "feat_01", "feat_02"}
    assert set(dt_df.head(3)["Feature"]) == {"feat_00", "feat_01", "feat_02"}
    assert set(mi_df.head(3)["Feature"]) == {"feat_00", "feat_01", "feat_02"}
    assert set(perm_df.head(3)["Feature"]) == {"feat_00", "feat_01", "feat_02"}

    # Run SAM-MFP
    candidates, _ = prioritize_sam_mfp(quality, salience, SAMMFPConfig())

    # Test single-comparator comparison function
    comparison_rf = compare_prioritization(candidates, rf_df, top_k=3)
    assert comparison_rf["overlap_count"] == 3
    assert comparison_rf["top_k_recall"] == 1.0
    assert comparison_rf["jaccard_index"] == 1.0
    assert len(comparison_rf["overlap_features"]) == 3
    assert len(comparison_rf["summary_notes"]) >= 2

    # Test multi-baseline comparison function
    multi_comp = compare_sam_mfp_with_baselines(
        candidates, features, y, top_k=3, baselines=("rf", "dt", "mi", "permutation")
    )
    assert "summary_table" in multi_comp
    summary_table = multi_comp["summary_table"]
    assert len(summary_table) == 4
    assert (summary_table["Top_K_Recall"] == 1.0).all()
    assert set(multi_comp["consensus_top_k"]) == {"feat_00", "feat_01", "feat_02"}
    assert set(multi_comp["consensus_pareto_overlap"]) == {"feat_00", "feat_01", "feat_02"}

    # Test variance and random baselines
    var_df = compute_variance_importance(features)
    assert len(var_df) == features.shape[1]
    assert "Variance" in var_df.columns
    assert "Variance_Rank" in var_df.columns
    assert var_df["Variance"].is_monotonic_decreasing

    rand_df = compute_random_ranking(features, random_state=42)
    assert len(rand_df) == features.shape[1]
    assert "Random_Score" in rand_df.columns
    assert "Random_Rank" in rand_df.columns

    # Test compare_sam_mfp_with_baselines with variance, mi, dt, and random
    baselines_comp = compare_sam_mfp_with_baselines(
        candidates, features, y, top_k=3, baselines=("variance", "mi", "dt", "random")
    )
    assert "variance" in baselines_comp["baseline_tables"]
    assert "random" in baselines_comp["baseline_tables"]
    assert "dt" in baselines_comp["baseline_tables"]
    assert "mi" in baselines_comp["baseline_tables"]
    assert len(baselines_comp["summary_table"]) == 4


def test_sam_mfp_collinearity_and_permutation_failure_mode():
    """Demonstrate permutation importance failure under collinearity vs SAM-MFP Pareto robustness."""
    rng = np.random.default_rng(42)
    n_frames = 400

    # Two perfectly collinear features separating binary states
    y_binary = np.array([0] * (n_frames // 2) + [1] * (n_frames // 2))
    base_sig = np.concatenate([rng.normal(1.0, 0.2, n_frames // 2), rng.normal(5.0, 0.2, n_frames // 2)])
    feat_a = base_sig + rng.normal(0, 0.01, n_frames)
    feat_b = base_sig + rng.normal(0, 0.01, n_frames)  # collinear clone

    collinear_df = pd.DataFrame({
        "feat_a": feat_a,
        "feat_b": feat_b,
        "noise_1": rng.normal(0, 1, n_frames),
        "noise_2": rng.normal(0, 1, n_frames),
    })

    # In permutation importance with a fitted RF on collinear features
    perm_df = compute_permutation_importance(collinear_df, y_binary, random_state=42)
    assert len(perm_df) == 4
    assert perm_df.loc[perm_df["Feature"] == "feat_a", "Permutation_Importance_Mean"].values[0] >= 0.0
    assert perm_df.loc[perm_df["Feature"] == "feat_b", "Permutation_Importance_Mean"].values[0] >= 0.0

    # In SAM-MFP, the Uniqueness objective explicitly penalizes the collinear duplicate
    quality_df = pd.DataFrame([
        {
            "Feature": "feat_a", "Resolution": "frame", "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.85, "Replica_Consistency": 0.90, "Occupancy_Entropy": 0.85,
            "Uniqueness": 0.02, "Measured_Cost_Seconds": 0.02,
        },
        {
            "Feature": "feat_b", "Resolution": "frame", "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.80, "Replica_Consistency": 0.90, "Occupancy_Entropy": 0.85,
            "Uniqueness": 0.02, "Measured_Cost_Seconds": 0.05,
        },
    ])
    salience_df = pd.DataFrame([
        {"Feature": "feat_a", "PCA_Salience": 0.88, "LOTO_Rank_Std_Normalized": 0.02, "LOTO_Folds": 3},
        {"Feature": "feat_b", "PCA_Salience": 0.88, "LOTO_Rank_Std_Normalized": 0.02, "LOTO_Folds": 3},
    ])

    cand, _ = prioritize_sam_mfp(quality_df, salience_df, SAMMFPConfig())
    indexed = cand.set_index("Feature")
    # feat_a strictly dominates feat_b due to higher ESS and lower cost, pruning the redundant copy
    assert indexed.loc["feat_a", "SAM_MFP_Pareto_Front"]
    assert not indexed.loc["feat_b", "SAM_MFP_Pareto_Front"]
    assert indexed.loc["feat_b", "SAM_MFP_Dominator_Count"] >= 1


def test_sam_mfp_ablation_response():
    """Verify that removing individual optimization objectives modifies Pareto dominance."""
    features, quality, salience, y = _generate_synthetic_benchmark(n_planted=3)

    # Add a collinear copy of feat_00 with very low Uniqueness (0.01)
    collinear_quality = pd.DataFrame([{
        "Feature": "feat_collinear",
        "Resolution": "frame",
        "Valid_Fraction": 1.0,
        "ESS_Ratio": 0.77,  # slightly worse than feat_00 (0.78)
        "Replica_Consistency": 0.85,
        "Occupancy_Entropy": 0.80,
        "Uniqueness": 0.01,  # heavily penalized for redundancy
        "Measured_Cost_Seconds": 0.02,
    }])
    collinear_salience = pd.DataFrame([{
        "Feature": "feat_collinear",
        "PCA_Salience": 0.92,
        "LOTO_Rank_Std_Normalized": 0.02,
        "LOTO_Folds": 3,
    }])

    quality_aug = pd.concat([quality, collinear_quality], ignore_index=True)
    salience_aug = pd.concat([salience, collinear_salience], ignore_index=True)

    # 1. Full objectives: feat_collinear must be dominated by feat_00 due to Uniqueness
    cand_full, _ = prioritize_sam_mfp(quality_aug, salience_aug, SAMMFPConfig())
    indexed_full = cand_full.set_index("Feature")
    assert not indexed_full.loc["feat_collinear", "SAM_MFP_Pareto_Front"]
    assert indexed_full.loc["feat_collinear", "SAM_MFP_Dominator_Count"] >= 1
    assert "Uniqueness" in cand_full.loc[0, "SAM_MFP_Objectives"]

    # 2. Ablate Uniqueness: configure custom objectives where feat_collinear matches feat_00
    config_custom = SAMMFPConfig(custom_objectives=("PCA_Salience", "Replica_Consistency"))
    cand_custom, _ = prioritize_sam_mfp(quality_aug, salience_aug, config_custom)
    indexed_custom = cand_custom.set_index("Feature")
    # With only PCA_Salience and Replica_Consistency (where feat_collinear == feat_00), neither dominates the other
    assert indexed_custom.loc["feat_collinear", "SAM_MFP_Pareto_Front"]
    assert indexed_custom.loc["feat_collinear", "SAM_MFP_Dominator_Count"] == 0

    # 3. Ablate PCA_Salience
    config_no_salience = SAMMFPConfig(excluded_objectives=("PCA_Salience",))
    cand_no_salience, _ = prioritize_sam_mfp(quality, salience, config_no_salience)
    assert "PCA_Salience" not in cand_no_salience.loc[0, "SAM_MFP_Objectives"]


def test_sam_mfp_ranking_stability_under_frame_subsampling():
    """Test ranking and Pareto front stability under 50% and 75% frame subsampling."""
    features, quality, salience, y = _generate_synthetic_benchmark(n_frames=800, n_planted=4)

    rng = np.random.default_rng(101)

    # Planted signal recovery across 5 random 50% subsampling draws
    for draw in range(5):
        sub_idx = rng.choice(len(features), size=len(features) // 2, replace=False)
        sub_X = features.iloc[sub_idx].reset_index(drop=True)
        sub_y = y.iloc[sub_idx].reset_index(drop=True)

        rf_full = compute_rf_importance(features, y, random_state=42)
        rf_sub = compute_rf_importance(sub_X, sub_y, random_state=42)

        # Signal features must remain in top positions
        top_full = set(rf_full.head(4)["Feature"])
        top_sub = set(rf_sub.head(4)["Feature"])
        overlap = top_full & top_sub
        assert len(overlap) >= 3, f"Draw {draw}: Top-4 baseline overlap too low ({overlap})"

    # Test full trajectory pipeline with actual assessment and discovery
    n_traj = 3
    frames_per_traj = 200
    manifest = pd.DataFrame({
        "trajectory_id": [f"t{i}" for i in range(n_traj) for _ in range(frames_per_traj)],
        "replica_id": [f"r{i}" for i in range(n_traj) for _ in range(frames_per_traj)],
        "ensemble_id": ["ens_0"] * (n_traj * frames_per_traj),
    })

    t_arr = np.linspace(0, 4 * np.pi, len(manifest))
    active_df = pd.DataFrame({
        "mode_alpha": np.sin(t_arr) * 4.0 + rng.normal(0, 0.2, len(manifest)),
        "mode_beta": np.cos(2 * t_arr) * 3.0 + rng.normal(0, 0.2, len(manifest)),
        "noise_0": rng.normal(0, 1.0, len(manifest)),
        "noise_1": rng.normal(0, 1.0, len(manifest)),
        "noise_2": rng.normal(0, 1.0, len(manifest)),
    })

    q_full = assess_feature_quality(active_df, manifest)
    r_full = discover_representation_features(active_df, manifest)["representation_feature_salience"]
    c_full, _ = prioritize_sam_mfp(q_full, r_full, SAMMFPConfig())
    pareto_full = set(c_full.loc[c_full["SAM_MFP_Pareto_Front"], "Feature"])

    # 75% frame subsampling stratified by trajectory
    sub_indices = []
    for t_id in [f"t{i}" for i in range(n_traj)]:
        idx = manifest.index[manifest["trajectory_id"] == t_id].to_numpy()
        sub_indices.extend(rng.choice(idx, size=int(len(idx) * 0.75), replace=False))
    sub_indices = sorted(sub_indices)

    sub_active = active_df.iloc[sub_indices].reset_index(drop=True)
    sub_manifest = manifest.iloc[sub_indices].reset_index(drop=True)

    q_sub = assess_feature_quality(sub_active, sub_manifest)
    r_sub = discover_representation_features(sub_active, sub_manifest)["representation_feature_salience"]
    c_sub, _ = prioritize_sam_mfp(q_sub, r_sub, SAMMFPConfig())
    pareto_sub = set(c_sub.loc[c_sub["SAM_MFP_Pareto_Front"], "Feature"])

    # Both active modes must be recovered on Pareto front in both full and subsampled datasets
    assert "mode_alpha" in pareto_full and "mode_alpha" in pareto_sub
    assert "mode_beta" in pareto_full and "mode_beta" in pareto_sub
    jaccard = len(pareto_full & pareto_sub) / len(pareto_full | pareto_sub)
    assert jaccard >= 0.50, f"Pareto Jaccard overlap under subsampling was {jaccard:.2f}"


def test_sam_mfp_replica_stability_and_dropped_replicas():
    """Verify SAM-MFP behavior under dropped replicas (LOTO stability)."""
    rng = np.random.default_rng(42)
    n_traj = 4
    frames_per_traj = 150
    manifest = pd.DataFrame({
        "trajectory_id": [f"t{i}" for i in range(n_traj) for _ in range(frames_per_traj)],
        "replica_id": [f"r{i}" for i in range(n_traj) for _ in range(frames_per_traj)],
        "ensemble_id": ["ens_0"] * (n_traj * frames_per_traj),
    })

    t_arr = np.linspace(0, 4 * np.pi, len(manifest))
    # Stable feature: present across all 4 trajectories
    stable_sig = np.sin(t_arr) * 3.0 + rng.normal(0, 0.2, len(manifest))
    # Erratic feature: huge variance in trajectory 0, completely flat/noise in trajectories 1, 2, 3
    erratic_sig = rng.normal(0, 0.1, len(manifest))
    erratic_sig[:frames_per_traj] = np.sin(t_arr[:frames_per_traj]) * 15.0

    df = pd.DataFrame({
        "feat_stable": stable_sig,
        "feat_erratic": erratic_sig,
        "noise": rng.normal(0, 1.0, len(manifest)),
    })

    quality = assess_feature_quality(df, manifest)
    rep_res = discover_representation_features(df, manifest)["representation_feature_salience"]

    # feat_erratic must have high LOTO rank variance (unstable when dropping replicas)
    rep_indexed = rep_res.set_index("Feature")
    assert rep_indexed.loc["feat_erratic", "LOTO_Rank_Std_Normalized"] > rep_indexed.loc["feat_stable", "LOTO_Rank_Std_Normalized"]

    # Prioritization: feat_stable has high Representation_Stability
    cand, _ = prioritize_sam_mfp(quality, rep_res, SAMMFPConfig())
    cand_indexed = cand.set_index("Feature")
    assert cand_indexed.loc["feat_stable", "Representation_Stability"] > cand_indexed.loc["feat_erratic", "Representation_Stability"]


def test_sam_mfp_cost_tradeoff():
    """Verify that computational cost acts as an active trade-off objective."""
    quality_df = pd.DataFrame([
        {
            "Feature": "fast_good",
            "Resolution": "frame",
            "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.85,
            "Replica_Consistency": 0.90,
            "Occupancy_Entropy": 0.85,
            "Uniqueness": 0.85,
            "Measured_Cost_Seconds": 0.01,  # 100x cheaper
        },
        {
            "Feature": "slow_marginally_better",
            "Resolution": "frame",
            "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.88,
            "Replica_Consistency": 0.92,
            "Occupancy_Entropy": 0.88,
            "Uniqueness": 0.88,
            "Measured_Cost_Seconds": 10.0,  # 10s compute
        },
    ])
    salience_df = pd.DataFrame([
        {"Feature": "fast_good", "PCA_Salience": 0.80},
        {"Feature": "slow_marginally_better", "PCA_Salience": 0.85},
    ])

    # With cost efficiency enabled, fast_good is non-dominated because it is vastly cheaper
    cand, _ = prioritize_sam_mfp(quality_df, salience_df, SAMMFPConfig())
    indexed = cand.set_index("Feature")
    assert indexed.loc["fast_good", "SAM_MFP_Pareto_Front"]
    assert indexed.loc["slow_marginally_better", "SAM_MFP_Pareto_Front"]

    # When cost is excluded, slow_marginally_better dominates fast_good because it is better on every quality/salience metric
    config_no_cost = SAMMFPConfig(excluded_objectives=("Cost_Efficiency",))
    cand_no_cost, _ = prioritize_sam_mfp(quality_df, salience_df, config_no_cost)
    indexed_no_cost = cand_no_cost.set_index("Feature")
    assert indexed_no_cost.loc["slow_marginally_better", "SAM_MFP_Pareto_Front"]
    assert not indexed_no_cost.loc["fast_good", "SAM_MFP_Pareto_Front"]
    assert indexed_no_cost.loc["fast_good", "SAM_MFP_Dominator_Count"] == 1


def test_sam_mfp_state_separation_objective():
    """Verify inclusion of state separation (eta-squared) in the Pareto optimization."""
    features, quality, salience, y = _generate_synthetic_benchmark(n_planted=2)

    state_explanatory = pd.DataFrame([
        {"Feature": "feat_00", "State_Explained_Variance_Eta_Squared": 0.85},
        {"Feature": "feat_01", "State_Explained_Variance_Eta_Squared": 0.75},
        {"Feature": "feat_02", "State_Explained_Variance_Eta_Squared": 0.01},
    ])

    config = SAMMFPConfig(include_state_separation=True)
    cand, notes = prioritize_sam_mfp(quality, salience, config=config, state_explanatory=state_explanatory)

    assert "State_Explained_Variance_Eta_Squared" in cand.loc[0, "SAM_MFP_Objectives"]
    assert cand.loc[cand["Feature"] == "feat_00", "State_Explained_Variance_Eta_Squared"].values[0] == 0.85


def test_sam_mfp_eligibility_and_validation():
    """Verify input validation, eligibility filtering, and error handling in SAM-MFP."""
    # min_loto_folds < 2 must raise ValueError
    with pytest.raises(ValueError, match="min_loto_folds must be at least 2"):
        SAMMFPConfig(min_loto_folds=1)

    # Missing required 'Feature' column
    with pytest.raises(ValueError, match="missing required 'Feature' column"):
        prioritize_sam_mfp(pd.DataFrame(), pd.DataFrame({"Feature": ["f1"]}))

    # Duplicate features must raise ValueError
    dup_quality = pd.DataFrame({
        "Feature": ["f1", "f1"],
        "Resolution": ["frame", "frame"],
        "Valid_Fraction": [1.0, 1.0],
        "ESS_Ratio": [0.5, 0.5],
        "Replica_Consistency": [0.5, 0.5],
        "Occupancy_Entropy": [0.5, 0.5],
        "Uniqueness": [0.5, 0.5],
    })
    dup_salience = pd.DataFrame({"Feature": ["f1", "f2"], "PCA_Salience": [0.5, 0.5]})
    with pytest.raises(ValueError, match="Feature names must be unique"):
        prioritize_sam_mfp(dup_quality, dup_salience)

    # Feature with Resolution != 'frame' must be ineligible
    non_frame_quality = pd.DataFrame({
        "Feature": ["traj_level_feat"],
        "Resolution": ["trajectory"],
        "Valid_Fraction": [1.0],
        "ESS_Ratio": [0.5],
        "Replica_Consistency": [0.5],
        "Occupancy_Entropy": [0.5],
        "Uniqueness": [0.5],
    })
    non_frame_salience = pd.DataFrame({"Feature": ["traj_level_feat"], "PCA_Salience": [0.5]})
    cand, _ = prioritize_sam_mfp(non_frame_quality, non_frame_salience)
    assert not cand.loc[0, "SAM_MFP_Eligible"]
    assert not cand.loc[0, "SAM_MFP_Pareto_Front"]


def test_sam_mfp_single_trajectory_nan_replica_consistency_fallback():
    """Verify that when Replica_Consistency is all-NaN (single trajectory), SAM-MFP automatically adapts."""
    quality = pd.DataFrame([
        {
            "Feature": "f1",
            "Resolution": "frame",
            "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.8,
            "Replica_Consistency": np.nan,  # single trajectory: undefined
            "Occupancy_Entropy": 0.7,
            "Uniqueness": 0.9,
        },
        {
            "Feature": "f2",
            "Resolution": "frame",
            "Valid_Fraction": 1.0,
            "ESS_Ratio": 0.3,
            "Replica_Consistency": np.nan,
            "Occupancy_Entropy": 0.4,
            "Uniqueness": 0.3,
        },
    ])
    salience = pd.DataFrame([
        {"Feature": "f1", "PCA_Salience": 0.85},
        {"Feature": "f2", "PCA_Salience": 0.15},
    ])
    cand, notes = prioritize_sam_mfp(quality, salience)
    assert cand.loc[cand["Feature"] == "f1", "SAM_MFP_Eligible"].values[0]
    assert cand.loc[cand["Feature"] == "f1", "SAM_MFP_Pareto_Front"].values[0]
    assert not cand.loc[cand["Feature"] == "f2", "SAM_MFP_Pareto_Front"].values[0]
    assert any("Single trajectory available" in n for n in notes)
    assert "Replica_Consistency" not in cand.loc[0, "SAM_MFP_Objectives"]
