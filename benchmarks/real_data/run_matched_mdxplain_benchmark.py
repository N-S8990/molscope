"""Head-to-head matched benchmark: MolScope vs mdxplain on Villin Headpiece (2RJY).

Dataset:
- PDB: 2RJY (Villin Headpiece HP-35, 64 residues, 1,027 atoms)
- Trajectory: simulation_run1.xtc (10,001 frames, 20 ns)
- Aligned Evaluation: 2,000 frames (stride 5, frames 0..9995)

Measured Metrics:
1. Conformational state agreement (Adjusted Rand Index & Normalized Mutual Information)
2. Overlap of top-ranked explanatory and predictive features (Jaccard similarity at k=5, 10, 20)
3. Step-by-step execution runtime (seconds) and peak RSS memory (MB)
4. Autocorrelation guardrails & generalization evaluation (purged temporal block CV)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psutil
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Ensure matplotlib cache is in writable /tmp
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"
# Add mdxplain to path
sys.path.insert(0, "/Users/nirav/Docs/mdxplain study/mdxplain")

from molscope.api import Study
from molscope.analysis.discovery.density_peaks import cluster_dpa, DPAConfig
from mdxplain import PipelineManager


def get_current_rss_mb() -> float:
    """Return current process RSS memory in megabytes."""
    return psutil.Process().memory_info().rss / (1024 * 1024)


def extract_residue_pair(feature_name: str) -> tuple[int, int] | None:
    """Extract canonical (min_seqid, max_seqid) pair from MolScope or mdxplain feature name.
    
    Examples:
    - 'CA_DISTANCE_angstrom__SYSTEM__ALA_26__SYSTEM__PRO_30' -> (26, 30)
    - 'CA_CONTACT__SYSTEM__GLU_27__SYSTEM__PRO_35' -> (27, 35)
    - 'ALA26 - PRO35' -> (26, 35)
    - 'LEU13-THR15' -> (13, 15)
    """
    # Check for MolScope naming
    match_ms = re.search(r"__([A-Z]{3})_(\d+)__SYSTEM__([A-Z]{3})_(\d+)", feature_name)
    if match_ms:
        id1 = int(match_ms.group(2))
        id2 = int(match_ms.group(4))
        return (min(id1, id2), max(id1, id2))
    
    # Check for mdxplain naming 'RES123 - RES456'
    match_mdx = re.search(r"([A-Za-z]+)(\d+)\s*[-_]\s*([A-Za-z]+)(\d+)", feature_name)
    if match_mdx:
        id1 = int(match_mdx.group(2))
        id2 = int(match_mdx.group(4))
        return (min(id1, id2), max(id1, id2))
    
    return None


def compute_jaccard(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    union = len(set_a.union(set_b))
    if union == 0:
        return 0.0
    return float(len(set_a.intersection(set_b)) / union)


def run_benchmark(output_dir: Path | str = "benchmarks/real_data") -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 72)
    print("MOLSCOPE VS MDXPLAIN MATCHED BENCHMARK (2RJY Villin Headpiece)")
    print("=" * 72)
    print(f"Target dataset: 2,000 frames (stride 5), 596 heavy atoms (HP-35 core)")
    print(f"Output directory: {output_dir.resolve()}")
    
    # ---------------------------------------------------------
    # PART 1: MOLSCOPE PIPELINE
    # ---------------------------------------------------------
    print("\n>>> Running MolScope Pipeline...")
    ms_init_mem = get_current_rss_mb()
    ms_peak_mem = ms_init_mem
    ms_t0 = time.perf_counter()
    
    # Step 1: Study Initialization
    study_out = output_dir / "molscope_output"
    study_out.mkdir(exist_ok=True)
    study = Study.from_project("studies/2rjy_benchmark/project.json", out_dir=study_out)
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    
    # Step 2: Feature Extraction
    t0 = time.perf_counter()
    study.run(extract_only=True)
    ms_t_extract = time.perf_counter() - t0
    ms_features = study.get_features()
    ms_manifest = study.get_manifest()
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope features extracted: {ms_features.shape[1]} features across {len(ms_features)} frames ({ms_t_extract:.2f}s)")
    
    # Step 3: Contact Kernel PCA (CKPCA)
    t0 = time.perf_counter()
    ms_ckpca = study.run_contact_kpca(n_components=5)
    ms_t_ckpca = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope CKPCA fitted: {ms_ckpca.coordinates.shape[1]} components ({ms_t_ckpca:.2f}s)")
    
    # Step 4: TICA (Time-lagged Independent Component Analysis)
    t0 = time.perf_counter()
    ms_tica = study.run_tica(lag_time=2, components=5)
    ms_t_tica = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope TICA fitted: {ms_tica.coordinates.shape[1]} components ({ms_t_tica:.2f}s)")
    
    # Step 5: DPA Clustering on CKPCA coordinates
    t0 = time.perf_counter()
    ms_dpa_ckpca = study.cluster_dpa(coordinate_name="CKPCA", z_score=2.0)
    ms_t_dpa_ckpca = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope DPA (on CKPCA): {ms_dpa_ckpca.n_clusters} clusters ({ms_t_dpa_ckpca:.2f}s)")
    
    # Step 6: DPA Clustering on TICA coordinates
    t0 = time.perf_counter()
    ms_dpa_tica = study.cluster_dpa(coordinate_name="TICA", z_score=2.0)
    ms_t_dpa_tica = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope DPA (on TICA): {ms_dpa_tica.n_clusters} clusters ({ms_t_dpa_tica:.2f}s)")
    
    # Step 7: Explainability - Decision Tree on DPA clusters
    t0 = time.perf_counter()
    ms_explainer = study.explain_clusters(clustering_name="dpa", max_depth=3)
    ms_t_dt = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    ms_rules_df = ms_explainer.rules_dataframe()
    ms_feat_imp_df = ms_explainer.get_feature_importances()
    print(f"  MolScope Decision Tree explainability: {len(ms_rules_df)} rules ({ms_t_dt:.2f}s)")
    
    # Step 8: Purged Temporal Block Cross-Validation (Predictive Rigor)
    t0 = time.perf_counter()
    folded_target = (ms_dpa_ckpca.assignments["Cluster"] == 0).astype(int).to_numpy()
    from molscope.analysis.predictive.workflow import purged_temporal_block_splits
    ptb_splits = purged_temporal_block_splits(ms_manifest, folded_target, n_splits=5, random_state=42, chunk_size=30, embargo=2)
    ms_t_rigor = time.perf_counter() - t0
    ms_peak_mem = max(ms_peak_mem, get_current_rss_mb())
    print(f"  MolScope Purged Temporal Block CV: {len(ptb_splits)} splits with 2-frame embargo ({ms_t_rigor:.2f}s)")
    
    ms_total_time = time.perf_counter() - ms_t0
    ms_final_mem = get_current_rss_mb()
    
    # ---------------------------------------------------------
    # PART 2: MDXPLAIN PIPELINE
    # ---------------------------------------------------------
    print("\n>>> Running mdxplain Pipeline...")
    mdx_init_mem = get_current_rss_mb()
    mdx_peak_mem = mdx_init_mem
    mdx_t0 = time.perf_counter()
    
    # Step 1: Initialize PipelineManager and Load Trajectory
    p = PipelineManager(use_memmap=False, show_progress=False)
    p.trajectory.load_trajectories("/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY/", stride=5)
    # Trim to exactly 2,000 frames to match MolScope's 2,000 frames
    p.data.trajectory_data.trajectories[0] = p.data.trajectory_data.trajectories[0][:2000]
    p.trajectory.add_labels(traj_selection="all")
    mdx_peak_mem = max(mdx_peak_mem, get_current_rss_mb())
    print(f"  mdxplain trajectory loaded: {p.data.trajectory_data.trajectories[0].n_frames} frames")
    
    # Step 2: Feature Extraction (Distances & Contacts)
    t0 = time.perf_counter()
    p.feature.add.distances()
    p.feature.add.contacts(cutoff=4.5)
    mdx_t_extract = time.perf_counter() - t0
    mdx_peak_mem = max(mdx_peak_mem, get_current_rss_mb())
    print(f"  mdxplain features extracted: distances and contacts ({mdx_t_extract:.2f}s)")
    
    # Step 3: Feature Selection (contacts only)
    p.feature_selector.create("contacts_only")
    p.feature_selector.add.contacts("contacts_only", "all")
    p.feature_selector.select("contacts_only")
    
    # Step 4: Contact Kernel PCA (CKPCA)
    t0 = time.perf_counter()
    p.decomposition.add.contact_kernel_pca(
        n_components=5, selection_name="contacts_only", decomposition_name="ContactKernelPCA"
    )
    mdx_t_ckpca = time.perf_counter() - t0
    mdx_peak_mem = max(mdx_peak_mem, get_current_rss_mb())
    mdx_decomp = p.data.decomposition_data["ContactKernelPCA"]
    print(f"  mdxplain CKPCA fitted: {mdx_decomp.data.shape[1]} components ({mdx_t_ckpca:.2f}s)")
    
    # Step 5: DPA Clustering on CKPCA
    t0 = time.perf_counter()
    p.clustering.add.dpa("ContactKernelPCA", Z=2.0, cluster_name="DPA_ContactKPC")
    mdx_t_dpa = time.perf_counter() - t0
    mdx_peak_mem = max(mdx_peak_mem, get_current_rss_mb())
    mdx_clus = p.data.get_cluster("DPA_ContactKPC")
    mdx_labels = mdx_clus.labels
    print(f"  mdxplain DPA fitted: {len(np.unique(mdx_labels))} clusters ({mdx_t_dpa:.2f}s)")
    
    # Step 6: Decision Tree Explainability
    t0 = time.perf_counter()
    p.data_selector.create_from_clusters(group_name="cluster", clustering_name="DPA_ContactKPC")
    p.comparison.create_comparison(
        name="cluster_comparison", mode="one_vs_rest", feature_selector="contacts_only", data_selector_groups="cluster"
    )
    p.feature_importance.add.decision_tree(
        comparison_name="cluster_comparison", analysis_name="feature_importance"
    )
    mdx_t_dt = time.perf_counter() - t0
    mdx_peak_mem = max(mdx_peak_mem, get_current_rss_mb())
    mdx_fi = p.data.feature_importance_data["feature_importance"]
    mdx_avg_imp = mdx_fi.get_average_importance()
    print(f"  mdxplain Decision Tree explainability fitted ({mdx_t_dt:.2f}s)")
    
    mdx_total_time = time.perf_counter() - mdx_t0
    mdx_final_mem = get_current_rss_mb()
    
    # ---------------------------------------------------------
    # PART 3: HEAD-TO-HEAD QUANTITATIVE EVALUATIONS
    # ---------------------------------------------------------
    print("\n>>> Computing Head-to-Head Comparison Metrics...")
    
    # 1. State Discovery Agreement
    ms_labels_ckpca = ms_dpa_ckpca.assignments["Cluster"].to_numpy()
    ms_labels_tica = ms_dpa_tica.assignments["Cluster"].to_numpy()
    
    # MolScope DPA on MolScope CKPCA vs mdxplain DPA on mdxplain CKPCA
    ari_ckpca = float(adjusted_rand_score(ms_labels_ckpca, mdx_labels))
    nmi_ckpca = float(normalized_mutual_info_score(ms_labels_ckpca, mdx_labels))
    
    # MolScope DPA on MolScope TICA vs mdxplain DPA on mdxplain CKPCA
    ari_tica_ckpca = float(adjusted_rand_score(ms_labels_tica, mdx_labels))
    nmi_tica_ckpca = float(normalized_mutual_info_score(ms_labels_tica, mdx_labels))
    
    # Algorithmic Isolation: Run MolScope's DPA directly on mdxplain's CKPCA coordinates
    mdx_coords = mdx_decomp.data
    ms_dpa_on_mdx_coords = cluster_dpa(mdx_coords, config=DPAConfig(z_score=2.0))
    ms_on_mdx_labels = ms_dpa_on_mdx_coords.assignments["Cluster"].to_numpy()
    ari_cross_algo = float(adjusted_rand_score(ms_on_mdx_labels, mdx_labels))
    nmi_cross_algo = float(normalized_mutual_info_score(ms_on_mdx_labels, mdx_labels))
    
    print(f"  Conformational State Agreement:")
    print(f"    - MolScope CKPCA vs mdxplain CKPCA: ARI = {ari_ckpca:.4f}, NMI = {nmi_ckpca:.4f}")
    print(f"    - MolScope TICA vs mdxplain CKPCA:  ARI = {ari_tica_ckpca:.4f}, NMI = {nmi_tica_ckpca:.4f}")
    print(f"    - MolScope DPA vs mdxplain DPA on same coords: ARI = {ari_cross_algo:.4f}, NMI = {nmi_cross_algo:.4f}")
    
    # 2. Feature Importance Overlap (Top-k Jaccard similarity)
    # Extract mdxplain top features
    mdx_meta = p.data.get_selected_metadata("contacts_only")
    mdx_ranked_indices = np.argsort(mdx_avg_imp)[::-1]
    mdx_features_ranked = []
    for rank, idx in enumerate(mdx_ranked_indices[:50]):
        pair = mdx_meta[idx]["features"]
        id1 = int(pair[0]["residue"]["seqid"])
        id2 = int(pair[1]["residue"]["seqid"])
        name1 = str(pair[0]["residue"]["full_name"])
        name2 = str(pair[1]["residue"]["full_name"])
        canon_pair = (min(id1, id2), max(id1, id2))
        mdx_features_ranked.append({
            "rank": rank + 1,
            "feature_name": f"{name1} - {name2}",
            "residue_pair": canon_pair,
            "importance": float(mdx_avg_imp[idx]),
        })
    mdx_top_df = pd.DataFrame(mdx_features_ranked)
    
    # Extract MolScope top features
    ms_features_ranked = []
    col_imp = "Gini_Importance" if "Gini_Importance" in ms_feat_imp_df.columns else ms_feat_imp_df.columns[1]
    for rank, (_, row) in enumerate(ms_feat_imp_df.iterrows()):
        feat = str(row["Feature"])
        canon_pair = extract_residue_pair(feat)
        ms_features_ranked.append({
            "rank": rank + 1,
            "feature_name": feat,
            "residue_pair": canon_pair,
            "importance": float(row[col_imp]),
        })
    ms_top_df = pd.DataFrame(ms_features_ranked)
    
    # Compute Top-k Jaccard overlap on valid residue pairs
    jaccard_results = []
    for k in [5, 10, 20]:
        ms_pairs_k = {row["residue_pair"] for _, row in ms_top_df.head(k).iterrows() if row["residue_pair"] is not None}
        mdx_pairs_k = {row["residue_pair"] for _, row in mdx_top_df.head(k).iterrows() if row["residue_pair"] is not None}
        jacc = compute_jaccard(ms_pairs_k, mdx_pairs_k)
        overlap_pairs = sorted(list(ms_pairs_k.intersection(mdx_pairs_k)))
        jaccard_results.append({
            "top_k": k,
            "jaccard_similarity": round(jacc, 4),
            "molscope_pairs_count": len(ms_pairs_k),
            "mdxplain_pairs_count": len(mdx_pairs_k),
            "intersection_count": len(overlap_pairs),
            "intersecting_residue_pairs": [f"{p[0]}-{p[1]}" for p in overlap_pairs],
        })
        print(f"  Top-{k} Feature Overlap: Jaccard = {jacc:.4f} (intersection: {overlap_pairs})")
    
    jaccard_df = pd.DataFrame(jaccard_results)
    
    # 3. Performance Summary
    timing_records = [
        {"Stage": "Feature Extraction", "MolScope_Seconds": ms_t_extract, "mdxplain_Seconds": mdx_t_extract, "Notes": "MolScope extracts contacts, distances, DSSP, dihedrals; mdxplain extracts distances & contacts"},
        {"Stage": "Dimensionality Reduction (CKPCA)", "MolScope_Seconds": ms_t_ckpca, "mdxplain_Seconds": mdx_t_ckpca, "Notes": "Contact Kernel PCA (5 components)"},
        {"Stage": "Clustering (DPA)", "MolScope_Seconds": ms_t_dpa_ckpca, "mdxplain_Seconds": mdx_t_dpa, "Notes": "Density Peak Advanced (Z=2.0)"},
        {"Stage": "Explainability (Decision Tree)", "MolScope_Seconds": ms_t_dt, "mdxplain_Seconds": mdx_t_dt, "Notes": "Decision tree rules and Gini importance"},
        {"Stage": "Total Pipeline", "MolScope_Seconds": ms_total_time, "mdxplain_Seconds": mdx_total_time, "Notes": "Full pipeline including setup and validation"},
        {"Stage": "Peak RSS Memory (MB)", "MolScope_Seconds": ms_peak_mem, "mdxplain_Seconds": mdx_peak_mem, "Notes": "Maximum memory footprint"},
    ]
    timing_df = pd.DataFrame(timing_records)
    
    clustering_records = [
        {"Comparison": "MolScope DPA (CKPCA) vs mdxplain DPA (CKPCA)", "Adjusted_Rand_Index": round(ari_ckpca, 4), "Normalized_Mutual_Info": round(nmi_ckpca, 4), "Notes": "Both pipelines end-to-end on Contact KPCA"},
        {"Comparison": "MolScope DPA (TICA) vs mdxplain DPA (CKPCA)", "Adjusted_Rand_Index": round(ari_tica_ckpca, 4), "Normalized_Mutual_Info": round(nmi_tica_ckpca, 4), "Notes": "Kinetic coordinates vs purely structural coordinates"},
        {"Comparison": "MolScope DPA vs mdxplain DPA (Same Coordinates)", "Adjusted_Rand_Index": round(ari_cross_algo, 4), "Normalized_Mutual_Info": round(nmi_cross_algo, 4), "Notes": "DPA algorithmic isolation on identical mdxplain CKPCA coordinates"},
    ]
    clustering_df = pd.DataFrame(clustering_records)
    
    # Save CSVs
    timing_df.to_csv(output_dir / "molscope_vs_mdxplain_timing_memory.csv", index=False)
    clustering_df.to_csv(output_dir / "clustering_agreement_metrics.csv", index=False)
    jaccard_df.to_csv(output_dir / "feature_overlap_topk.csv", index=False)
    ms_top_df.to_csv(output_dir / "molscope_top_features.csv", index=False)
    mdx_top_df.to_csv(output_dir / "mdxplain_top_features.csv", index=False)
    
    # Save assignments CSV
    assignments_df = pd.DataFrame({
        "Frame": np.arange(2000),
        "MolScope_DPA_CKPCA_Cluster": ms_labels_ckpca,
        "MolScope_DPA_TICA_Cluster": ms_labels_tica,
        "mdxplain_DPA_CKPCA_Cluster": mdx_labels,
    })
    assignments_df.to_csv(output_dir / "cluster_assignments.csv", index=False)
    
    # Save JSON summary
    benchmark_data = {
        "dataset": {
            "name": "Villin Headpiece HP-35 (PDB: 2RJY)",
            "frames_analyzed": 2000,
            "stride": 5,
            "heavy_atoms": 596,
            "total_atoms": 1027,
            "residues": 64,
        },
        "performance": {
            "molscope": {
                "wall_time_seconds": {
                    "feature_extraction": float(ms_t_extract),
                    "contact_kpca": float(ms_t_ckpca),
                    "tica": float(ms_t_tica),
                    "dpa_ckpca": float(ms_t_dpa_ckpca),
                    "dpa_tica": float(ms_t_dpa_tica),
                    "decision_tree": float(ms_t_dt),
                    "purged_temporal_cv": float(ms_t_rigor),
                    "total": float(ms_total_time),
                },
                "memory_mb": {
                    "initial_rss": float(ms_init_mem),
                    "peak_rss": float(ms_peak_mem),
                    "delta_rss": float(ms_peak_mem - ms_init_mem),
                },
            },
            "mdxplain": {
                "wall_time_seconds": {
                    "feature_extraction": float(mdx_t_extract),
                    "contact_kpca": float(mdx_t_ckpca),
                    "dpa": float(mdx_t_dpa),
                    "decision_tree": float(mdx_t_dt),
                    "total": float(mdx_total_time),
                },
                "memory_mb": {
                    "initial_rss": float(mdx_init_mem),
                    "peak_rss": float(mdx_peak_mem),
                    "delta_rss": float(mdx_peak_mem - mdx_init_mem),
                },
            },
        },
        "conformational_states": {
            "molscope_ckpca_clusters": int(ms_dpa_ckpca.n_clusters),
            "molscope_tica_clusters": int(ms_dpa_tica.n_clusters),
            "mdxplain_ckpca_clusters": int(len(np.unique(mdx_labels))),
            "clustering_agreement": clustering_records,
        },
        "feature_overlap": jaccard_results,
    }
    
    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(benchmark_data, f, indent=2)
        
    print("\n" + "=" * 72)
    print("MATCHED BENCHMARK COMPLETE!")
    print(f"Results written to: {output_dir.resolve()}")
    print("=" * 72)
    return benchmark_data


if __name__ == "__main__":
    run_benchmark()
