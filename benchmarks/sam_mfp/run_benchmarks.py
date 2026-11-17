"""Benchmark suite for System-Aware Multi-Criteria Feature Prioritization (SAM-MFP).

Executes synthetic planted-signal benchmarks and real-world 2RJY Villin Headpiece
MD benchmarks, comparing SAM-MFP Pareto candidates against:
- Empirical feature variance (Variance Fallacy)
- Mutual information with conformational state labels
- Decision-tree Gini/MDI importance
- Random ranking (null hypothesis)
- Supervised Random Forest and Permutation Importance
"""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

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
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp


def run_synthetic_benchmark(
    n_frames: int = 600,
    n_features: int = 20,
    n_planted: int = 3,
    seed: int = 42,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Execute synthetic benchmark with planted ground-truth signals and complementary trade-offs."""
    print("--- Running Synthetic Planted-Signal Benchmark ---")
    rng = np.random.default_rng(seed)
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]

    # Feature trajectories
    data = {f: rng.normal(0.0, 1.0, n_frames) for f in feature_names}
    features_df = pd.DataFrame(data)

    # Synergistic target
    weights = [2.5, 2.0, 1.5]
    y_raw = np.zeros(n_frames)
    for i in range(n_planted):
        y_raw += weights[i % len(weights)] * data[feature_names[i]]
    y = pd.Series(y_raw + rng.normal(0, 0.4, n_frames), name="Target")

    # Multi-criteria quality profiles
    quality_rows = []
    planted_profiles = [
        {"ESS_Ratio": 0.78, "Replica_Consistency": 0.85, "Occupancy_Entropy": 0.80, "Uniqueness": 0.82, "Cost": 0.02},
        {"ESS_Ratio": 0.95, "Replica_Consistency": 0.98, "Occupancy_Entropy": 0.85, "Uniqueness": 0.80, "Cost": 0.08},
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

    # Run SAM-MFP
    t0 = time.perf_counter()
    candidates, notes = prioritize_sam_mfp(quality_df, salience_df, SAMMFPConfig())
    t_sam_mfp = time.perf_counter() - t0

    pareto_feats = candidates.loc[candidates["SAM_MFP_Pareto_Front"], "Feature"].tolist()
    planted_set = {f"feat_{i:02d}" for i in range(n_planted)}
    planted_recovered = set(pareto_feats) & planted_set

    # Run all baselines
    baselines_comp = compare_sam_mfp_with_baselines(
        candidates,
        features_df,
        y,
        top_k=n_planted,
        baselines=("rf", "dt", "mi", "permutation", "variance", "random"),
        random_state=seed,
    )
    summary_table = baselines_comp["summary_table"]

    # Calculate ground truth recall for each method
    planted_metrics = []
    for b_name, b_df in baselines_comp["baseline_tables"].items():
        top_b = set(b_df.head(n_planted)["Feature"])
        rec = len(top_b & planted_set) / n_planted
        jacc = len(top_b & planted_set) / len(top_b | planted_set)
        planted_metrics.append({
            "Method": b_name,
            "Planted_Recall": rec,
            "Planted_Jaccard": jacc,
            "Top_Features": list(b_df.head(n_planted)["Feature"]),
        })

    planted_metrics.append({
        "Method": "SAM-MFP (Pareto Front)",
        "Planted_Recall": len(planted_recovered) / n_planted,
        "Planted_Jaccard": len(planted_recovered) / len(set(pareto_feats) | planted_set),
        "Top_Features": pareto_feats,
    })
    planted_df = pd.DataFrame(planted_metrics)

    res = {
        "dataset": "synthetic_planted",
        "n_features": n_features,
        "n_planted": n_planted,
        "wall_time_seconds": t_sam_mfp,
        "pareto_front_size": len(pareto_feats),
        "planted_recovered": sorted(planted_recovered),
        "planted_recall": len(planted_recovered) / n_planted,
        "summary_table": summary_table.to_dict(orient="records"),
        "planted_evaluation": planted_df.to_dict(orient="records"),
    }

    if output_dir:
        summary_table.to_csv(output_dir / "synthetic_baseline_comparison.csv", index=False)
        planted_df.to_csv(output_dir / "synthetic_planted_evaluation.csv", index=False)

    print(f"Synthetic benchmark: {len(planted_recovered)}/{n_planted} planted features on Pareto front in {t_sam_mfp:.4f}s.")
    return res


def run_real_2rjy_benchmark(
    run_dir: Path = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark"),
    output_dir: Path | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute real-world benchmark on the 2RJY Villin Headpiece MD trajectory dataset."""
    print("--- Running Real 2RJY Villin Headpiece Benchmark ---")
    quality = pd.read_csv(run_dir / "feature_quality_metrics.csv")
    salience = pd.read_csv(run_dir / "representation_feature_salience.csv")
    features_full = pd.read_parquet(run_dir / "frame_features.parquet")
    states = pd.read_csv(run_dir / "dpa_cluster_assignments.csv")

    t0 = time.perf_counter()
    candidates, notes = prioritize_sam_mfp(quality, salience, SAMMFPConfig())
    t_sam = time.perf_counter() - t0

    eligible = candidates[candidates["SAM_MFP_Eligible"]].copy()
    pareto_mask = candidates["SAM_MFP_Pareto_Front"]
    pareto_count = int(pareto_mask.sum())
    eligible_count = int(candidates["SAM_MFP_Eligible"].sum())

    eligible_feats = eligible["Feature"].tolist()
    X = features_full[eligible_feats]
    y = states["Cluster"]

    print(f"2RJY: {eligible_count} eligible features, {pareto_count} Pareto front candidates (computed in {t_sam:.3f}s).")

    # Run comparisons against baselines across multiple k thresholds
    k_values = [10, 20, 50]
    multi_k_results = []
    baseline_tables = {}

    # Compute baseline tables once
    print("Computing baselines (variance, MI, Decision Tree, Random)...")
    t0_var = time.perf_counter()
    var_df = compute_variance_importance(X)
    t_var = time.perf_counter() - t0_var

    t0_mi = time.perf_counter()
    mi_df = compute_mutual_info(X, y, random_state=seed)
    t_mi = time.perf_counter() - t0_mi

    t0_dt = time.perf_counter()
    dt_df = compute_decision_tree_importance(X, y, random_state=seed)
    t_dt = time.perf_counter() - t0_dt

    t0_rand = time.perf_counter()
    rand_df = compute_random_ranking(X, random_state=seed)
    t_rand = time.perf_counter() - t0_rand

    baseline_tables = {
        "variance": var_df,
        "mi": mi_df,
        "dt": dt_df,
        "random": rand_df,
    }

    timing_dict = {
        "sam_mfp_prioritization": t_sam,
        "variance_baseline": t_var,
        "mutual_info_baseline": t_mi,
        "decision_tree_baseline": t_dt,
        "random_baseline": t_rand,
    }

    for k in k_values:
        for b_name, b_df in baseline_tables.items():
            comp = compare_prioritization(candidates, b_df, top_k=k)
            multi_k_results.append({
                "Baseline": b_name,
                "Top_K": k,
                "Pareto_Count": comp["pareto_count"],
                "Overlap_Count": comp["overlap_count"],
                "Top_K_Recall": comp["top_k_recall"],
                "Pareto_Precision": comp["pareto_precision"],
                "Jaccard_Index": comp["jaccard_index"],
                "Spearman_Rho": comp["spearman_rho"],
                "Spearman_PValue": comp["spearman_pvalue"],
                "Divergent_Baseline_Top_Count": len(comp["divergent_baseline_top"]),
                "Divergent_Pareto_Count": len(comp["divergent_pareto_front"]),
            })

    comparison_df = pd.DataFrame(multi_k_results)

    # Identify divergent features between baselines and SAM-MFP
    # Top 10 variance features that SAM-MFP rejects (Variance Fallacy)
    comp_var_20 = compare_prioritization(candidates, var_df, top_k=20)
    var_fallacy_feats = comp_var_20["divergent_baseline_top"]
    var_fallacy_df = candidates[candidates["Feature"].isin(var_fallacy_feats)][
        ["Feature", "PCA_Salience", "ESS_Ratio", "Occupancy_Entropy", "Uniqueness", "SAM_MFP_Dominator_Count"]
    ]

    # Top 10 SAM-MFP Pareto features with lowest variance (subtle collective features)
    pareto_rows = candidates[candidates["SAM_MFP_Pareto_Front"]].merge(var_df, on="Feature", how="inner")
    subtle_pareto = pareto_rows.sort_values("Variance", ascending=True).head(10)[
        ["Feature", "Variance", "PCA_Salience", "ESS_Ratio", "Occupancy_Entropy", "Uniqueness"]
    ]

    res = {
        "dataset": "2RJY_villin_headpiece",
        "n_frames": len(features_full),
        "total_features": len(candidates),
        "eligible_features": eligible_count,
        "pareto_front_count": pareto_count,
        "timings_seconds": timing_dict,
        "baseline_comparison": comparison_df.to_dict(orient="records"),
        "variance_fallacy_rejected_features": var_fallacy_df.to_dict(orient="records"),
        "subtle_pareto_features": subtle_pareto.to_dict(orient="records"),
        "diagnostics_notes": notes,
    }

    if output_dir:
        comparison_df.to_csv(output_dir / "real_2rjy_baseline_comparison.csv", index=False)
        var_fallacy_df.to_csv(output_dir / "variance_fallacy_rejected_features.csv", index=False)
        subtle_pareto.to_csv(output_dir / "subtle_pareto_features.csv", index=False)
        candidates[candidates["SAM_MFP_Pareto_Front"]].to_csv(output_dir / "real_2rjy_pareto_candidates.csv", index=False)

    return res


def run_ablation_benchmarks(
    run_dir: Path = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark"),
    output_dir: Path | None = None,
) -> pd.DataFrame:
    """Execute systematic ablation of individual SAM-MFP optimization objectives on 2RJY."""
    print("--- Running SAM-MFP Objective Ablation Study on 2RJY ---")
    quality = pd.read_csv(run_dir / "feature_quality_metrics.csv")
    salience = pd.read_csv(run_dir / "representation_feature_salience.csv")

    ablation_configs = [
        ("Full_Default_Objectives", SAMMFPConfig()),
        ("Ablate_Uniqueness", SAMMFPConfig(excluded_objectives=("Uniqueness",))),
        ("Ablate_ESS_Ratio", SAMMFPConfig(excluded_objectives=("ESS_Ratio",))),
        ("Ablate_Occupancy_Entropy", SAMMFPConfig(excluded_objectives=("Occupancy_Entropy",))),
        ("Ablate_PCA_Salience", SAMMFPConfig(excluded_objectives=("PCA_Salience",))),
        ("Salience_Only_Single_Objective", SAMMFPConfig(custom_objectives=("PCA_Salience",))),
    ]

    ablation_records = []
    base_pareto = None

    for name, cfg in ablation_configs:
        cand, notes = prioritize_sam_mfp(quality, salience, config=cfg)
        pareto_set = set(cand.loc[cand["SAM_MFP_Pareto_Front"], "Feature"])
        if name == "Full_Default_Objectives":
            base_pareto = pareto_set
            jacc = 1.0
            overlap = len(pareto_set)
        else:
            overlap = len(pareto_set & base_pareto) if base_pareto else 0
            union = len(pareto_set | base_pareto) if base_pareto else 1
            jacc = overlap / union if union > 0 else 0.0

        ablation_records.append({
            "Ablation": name,
            "Objectives": cand.loc[0, "SAM_MFP_Objectives"],
            "Eligible_Count": int(cand["SAM_MFP_Eligible"].sum()),
            "Pareto_Front_Size": len(pareto_set),
            "Overlap_With_Full_Pareto": overlap,
            "Jaccard_With_Full_Pareto": jacc,
        })

    ablation_df = pd.DataFrame(ablation_records)
    if output_dir:
        ablation_df.to_csv(output_dir / "ablation_study_results.csv", index=False)
    print(ablation_df.to_string(index=False))
    return ablation_df


def main():
    bench_dir = Path("/Users/nirav/Docs/molscope/benchmarks/sam_mfp")
    bench_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("SYSTEM-AWARE MULTI-CRITERIA FEATURE PRIORITIZATION (SAM-MFP)")
    print("BENCHMARK AND VALIDATION SUITE")
    print("=================================================================\n")

    t_start = time.perf_counter()

    # 1. Synthetic Planted Signal Benchmark
    synthetic_res = run_synthetic_benchmark(output_dir=bench_dir)
    print()

    # 2. Real 2RJY Villin Headpiece Benchmark
    real_res = run_real_2rjy_benchmark(output_dir=bench_dir)
    print()

    # 3. Ablation Study on 2RJY
    ablation_df = run_ablation_benchmarks(output_dir=bench_dir)
    print()

    # 4. Save combined summary JSON
    total_time = time.perf_counter() - t_start
    overall_summary = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_benchmark_time_seconds": float(total_time),
        "synthetic_benchmark": synthetic_res,
        "real_2rjy_benchmark": real_res,
        "ablation_study": ablation_df.to_dict(orient="records"),
    }

    summary_json_path = bench_dir / "sam_mfp_benchmark_summary.json"
    with open(summary_json_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    print("=================================================================")
    print(f"SAM-MFP BENCHMARKS COMPLETED IN {total_time:.2f}s")
    print(f"Artifacts written to: {bench_dir}")
    print("=================================================================")


if __name__ == "__main__":
    main()
