"""Execute the 10,001-frame Villin Headpiece (2RJY) benchmark through MolScope's full pipeline.

Outputs:
- Feature extraction (Cα distances, contacts, secondary structure, sidechain dihedrals)
- Dimensionality reduction: TICA & Contact KPCA
- State discovery: Density Peak Advanced (DPA) clustering
- Free Energy Landscapes: 2D FES on TIC1 vs TIC2
- State explainability: Decision tree boolean rules and Gini importance
- 3D molecular visualization: PyMOL & ChimeraX export scripts
- Output destination: /Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import psutil

import numpy as np
import pandas as pd

from molscope.api import Study


def main():
    out_dir = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)
    process = psutil.Process()

    start_mem = process.memory_info().rss / (1024 * 1024)
    start_time = time.perf_counter()

    print("Step 1: Initializing MolScope Study for 2RJY Villin Headpiece...")
    study = Study.from_project("studies/2rjy_benchmark/project.json", out_dir=out_dir)

    print("Step 2: Extracting biophysical features (stride 5, 2000 frames)...")
    t0_extract = time.perf_counter()
    study.run(extract_only=True)
    t_extract = time.perf_counter() - t0_extract
    features = study.get_features()
    manifest = study.get_manifest()
    print(f"Extracted {features.shape[1]} features across {features.shape[0]} frames in {t_extract:.2f}s")

    print("Step 3: Fitting Time-lagged Independent Component Analysis (TICA)...")
    t0_tica = time.perf_counter()
    tica_res = study.run_tica(lag_time=2, components=5)
    t_tica = time.perf_counter() - t0_tica
    print(f"TICA complete: implied timescales = {tica_res.implied_timescales[:3]} (fitted in {t_tica:.2f}s)")

    print("Step 4: Fitting Contact Kernel PCA (CKPCA)...")
    t0_ckpca = time.perf_counter()
    ckpca_res = study.run_contact_kpca(n_components=5)
    t_ckpca = time.perf_counter() - t0_ckpca
    print(f"CKPCA complete: variance ratios = {ckpca_res.explained_variance_ratio[:3]} (fitted in {t_ckpca:.2f}s)")

    print("Step 5: Density Peak Advanced (DPA) state clustering...")
    t0_dpa = time.perf_counter()
    dpa_res = study.cluster_dpa(coordinate_name="TICA")
    t_dpa = time.perf_counter() - t0_dpa
    print(f"DPA complete: discovered {dpa_res.n_clusters} conformational states in {t_dpa:.2f}s")
    for _, row in dpa_res.populations.iterrows():
        st_id = row.iloc[0]
        pop = row.iloc[1]
        print(f"  State {st_id}: {pop} frames ({float(pop)/len(features)*100:.1f}%)")

    print("Step 6: Generating 2D Free Energy Landscapes...")
    fes_png = out_dir / "free_energy_landscape_tic1_tic2.png"
    fes_html = out_dir / "free_energy_landscape_tic1_tic2.html"
    study.plot_free_energy_landscape(
        dim1="TIC1",
        dim2="TIC2",
        temperature_k=300.0,
        units="kcal/mol",
    )
    print(f"Saved free energy surface: {fes_png}")

    print("Step 7: Decision tree rule extraction and cluster explainability...")
    t0_dt = time.perf_counter()
    explainer = study.explain_clusters(clustering_name="dpa", max_depth=3)
    t_dt = time.perf_counter() - t0_dt
    rules_df = explainer.rules_dataframe()
    print(f"Extracted {len(rules_df)} explanatory decision rules in {t_dt:.2f}s:")
    for _, row in rules_df.head(6).iterrows():
        print(f"  Cluster {row['Cluster']}: Precision={row['Precision']:.2f}, Recall={row['Recall']:.2f} | {row['Rule']}")

    print("Step 8: Exporting 3D molecular visualization scripts for PyMOL & ChimeraX...")
    viz_files = study.export_3d(score_column="Kinetic_Salience", out_dir=out_dir / "3d_visualization")
    print(f"Exported 3D visualizers: {list(viz_files.keys())}")

    total_time = time.perf_counter() - start_time
    peak_mem = process.memory_info().rss / (1024 * 1024)

    feat_imp_df = explainer.get_feature_importances()
    top_feats = []
    if not feat_imp_df.empty:
        col = "Gini_Importance" if "Gini_Importance" in feat_imp_df.columns else ("Importance" if "Importance" in feat_imp_df.columns else feat_imp_df.columns[1])
        for _, row in feat_imp_df.head(8).iterrows():
            top_feats.append({"feature": str(row["Feature"]), "importance": float(row[col])})

    benchmark_metrics = {
        "dataset": "2RJY Villin Headpiece (HP-35)",
        "n_frames_analyzed": int(len(features)),
        "n_features_extracted": int(features.shape[1]),
        "wall_time_seconds": {
            "feature_extraction": float(t_extract),
            "tica_dim_reduction": float(t_tica),
            "contact_kpca": float(t_ckpca),
            "dpa_clustering": float(t_dpa),
            "decision_tree_explainability": float(t_dt),
            "total_pipeline_time": float(total_time),
        },
        "memory_mb": {
            "initial": float(start_mem),
            "peak": float(peak_mem),
            "delta": float(peak_mem - start_mem),
        },
        "dpa_clusters": {
            "n_clusters": int(dpa_res.n_clusters),
            "populations": dpa_res.populations.to_dict(orient="records"),
        },
        "top_discriminating_features": top_feats,
    }

    with open(out_dir / "benchmark_summary.json", "w") as f:
        json.dump(benchmark_metrics, f, indent=2)

    # Write tables
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    features.to_csv(tables_dir / "extracted_features.csv", index=False)
    manifest.to_csv(tables_dir / "manifest.csv", index=False)
    dpa_res.assignments.to_csv(tables_dir / "dpa_cluster_assignments.csv", index=False)
    tica_res.coordinates.to_csv(tables_dir / "tica_coordinates.csv", index=False)
    if not feat_imp_df.empty:
        feat_imp_df.to_csv(tables_dir / "decision_tree_feature_importances.csv", index=False)

    print("\n==========================================")
    print("MOLSCOPE 2RJY BENCHMARK COMPLETED SUCCESSFULLY!")
    print(f"Total time: {total_time:.2f}s | Peak RAM: {peak_mem:.1f} MB")
    print(f"Outputs written to: {out_dir}")
    print("==========================================")


if __name__ == "__main__":
    main()
