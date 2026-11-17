"""Worked example demonstrating MolScope interpretability & visualization tools on 2RJY (HP-35 Villin Headpiece).

Demonstrates:
1. Free Energy Landscape 2D surface plots on leading PCA components (PC1 vs PC2) with detected conformational basins.
2. Cluster membership over simulation time (cluster timeline) and empirical state transition matrix.
3. Feature importance ranking bar plots and Decision Tree if-then rule visualization.
4. PDB Beta-factor importance mapping writing per-residue importance for PyMOL, ChimeraX, and Mol* (Biomolecular Structure Viewer sidecar).
5. PyMOL visualization script generator.
6. Consolidated offline HTML report dashboard.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure non-interactive backend and writable matplotlib config
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from molscope.api import Study
from molscope.reporting.free_energy import compute_pca_free_energy_landscape, plot_pca_free_energy
from molscope.visualization.pdb_mapping import (
    extract_residue_scores,
    generate_3d_visualization,
    map_feature_importance_to_pdb,
    generate_pymol_script,
)
from molscope.visualization.timeline import (
    compute_transition_matrix,
    generate_cluster_timeline_plots,
    plot_cluster_timeline,
    plot_state_transitions,
)
from molscope.visualization.rules import (
    plot_decision_tree_rules,
    plot_feature_importance,
    render_decision_tree_html,
)
from molscope.analysis.discovery.decision_tree import DecisionTreeExplainer
from molscope.reporting.html_report import generate_html_report


DEFAULT_REFERENCE_PDB = Path("/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY/protein.pdb")
DEFAULT_PRECOMPUTED_DIR = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark")
DEFAULT_OUTPUT_DIR = Path("/Users/nirav/Docs/molscope-results/runs/2rjy_interpretability")


def run_interpretability_pipeline(
    output_dir: str | Path,
    reference_pdb: str | Path | None = None,
    precomputed_dir: str | Path | None = None,
    project_json: str | Path | None = None,
) -> Path:
    """Run end-to-end interpretability and visualization workflow on 2RJY."""
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    viz_dir = out_dir / "3d_visualization"
    viz_dir.mkdir(exist_ok=True)

    pdb_path = Path(reference_pdb or DEFAULT_REFERENCE_PDB).resolve()
    if not pdb_path.is_file():
        # Fallback to test fixture if external path is absent
        pdb_path = Path(__file__).parents[1] / "tests" / "fixtures" / "1A22.pdb"

    print("=" * 70)
    print("MolScope Interpretability & Visualization: 2RJY Villin Headpiece")
    print("=" * 70)
    print(f"Output directory : {out_dir}")
    print(f"Reference PDB    : {pdb_path}")

    # -------------------------------------------------------------------------
    # 1. Load coordinates, states, features, and manifest
    # -------------------------------------------------------------------------
    precomputed = Path(precomputed_dir or DEFAULT_PRECOMPUTED_DIR).resolve()
    df_coords: pd.DataFrame | None = None
    df_states: pd.DataFrame | None = None
    df_importances: pd.DataFrame | None = None
    df_manifest: pd.DataFrame | None = None
    df_features: pd.DataFrame | None = None

    if precomputed.is_dir() and (precomputed / "representation_coordinates.csv").is_file():
        print(f"\n[1/6] Loading precomputed benchmark data from: {precomputed}...")
        df_coords = pd.read_csv(precomputed / "representation_coordinates.csv")
        df_states = pd.read_csv(precomputed / "state_assignments.csv")
        if (precomputed / "decision_tree_feature_importances.csv").is_file():
            df_importances = pd.read_csv(precomputed / "decision_tree_feature_importances.csv")
        elif (precomputed / "representation_feature_salience.csv").is_file():
            df_importances = pd.read_csv(precomputed / "representation_feature_salience.csv")
        if (precomputed / "frame_manifest.parquet").is_file():
            df_manifest = pd.read_parquet(precomputed / "frame_manifest.parquet")
        if (precomputed / "frame_features.parquet").is_file():
            df_features = pd.read_parquet(precomputed / "frame_features.parquet")
    elif project_json and Path(project_json).is_file():
        print(f"\n[1/6] Running feature extraction and state discovery from: {project_json}...")
        study = Study.from_project(project_json, out_dir=out_dir)
        study.run()
        df_coords = study._pipeline.table("representation_coordinates")
        df_states = study._pipeline.table("state_assignments")
        df_importances = study._pipeline.table("representation_feature_salience")
        df_manifest = study.get_manifest()
        df_features = study.get_features()
    else:
        # Generate synthetic coordinates and states for demonstration
        print("\n[1/6] Synthesizing Villin Headpiece HP-35 conformational states...")
        n_frames = 2000
        t_ps = np.linspace(0, 10000, n_frames)
        # Bimodal conformational distribution
        c1 = np.random.multivariate_normal([-3.5, -2.0], [[1.2, 0.4], [0.4, 0.8]], n_frames // 2)
        c2 = np.random.multivariate_normal([4.0, 2.5], [[1.5, -0.3], [-0.3, 1.0]], n_frames // 2)
        coords = np.vstack([c1, c2])
        df_coords = pd.DataFrame(coords, columns=["PC1", "PC2"])
        df_coords["trajectory_id"] = "run1"
        df_coords["time_ps"] = t_ps

        states = np.array([0] * (n_frames // 2) + [1] * (n_frames // 2))
        df_states = pd.DataFrame({
            "trajectory_id": "run1",
            "time_ps": t_ps,
            "State": states,
            "State_Label": [f"state_{s}" for s in states],
        })
        df_importances = pd.DataFrame([
            {"Feature": "CA_DISTANCE_angstrom__SYSTEM__THR_24__SYSTEM__SER_43", "Gini_Importance": 0.70},
            {"Feature": "CA_DISTANCE_angstrom__SYSTEM__ASN_23__SYSTEM__LYS_71", "Gini_Importance": 0.15},
            {"Feature": "CA_DISTANCE_angstrom__SYSTEM__ASP_34__SYSTEM__TRP_64", "Gini_Importance": 0.08},
            {"Feature": "SIDECHAIN_CHI1__SYSTEM__PHE_58", "Gini_Importance": 0.05},
        ])

    print(f"  Frames available : {len(df_coords):,}")
    print(f"  States           : {df_states['State'].nunique()} unique macrostates")

    # -------------------------------------------------------------------------
    # 2. Free Energy Landscape 2D surface plot on leading PCA components
    # -------------------------------------------------------------------------
    print("\n[2/6] Computing 2D Free Energy Landscape on leading PCA components (PC1 vs PC2)...")
    fes_outputs = plot_pca_free_energy(
        representation_coordinates=df_coords,
        output_dir=figures_dir,
        pc_x="PC1",
        pc_y="PC2",
        prefix="pca_free_energy_landscape",
        temperature_k=300.0,
        units="kcal/mol",
        title="2RJY Villin Headpiece: Free Energy Landscape (PC1 vs PC2)",
        interactive=True,
    )
    print(f"  -> Static surface PNG : {fes_outputs['static']}")
    if "interactive" in fes_outputs:
        print(f"  -> Interactive HTML   : {fes_outputs['interactive']}")
    if "basins_csv" in fes_outputs:
        df_basins = pd.read_csv(fes_outputs["basins_csv"])
        print(f"  -> Detected {len(df_basins)} free energy basins:")
        for _, b in df_basins.iterrows():
            print(f"     * Basin {int(b['basin_id'])} at (PC1={b['x']:.2f}, PC2={b['y']:.2f}): ΔG = {b['free_energy']:.2f} kcal/mol")

    # -------------------------------------------------------------------------
    # 3. Cluster Membership Timeline and State Transitions
    # -------------------------------------------------------------------------
    print("\n[3/6] Generating cluster timeline and state transition probability matrix...")
    timeline_outputs = generate_cluster_timeline_plots(
        state_assignments=df_states,
        output_dir=figures_dir,
        prefix="cluster_timeline",
        time_col="time_ps",
        state_col="State",
        trajectory_col="trajectory_id",
        title="2RJY Cluster Membership Timeline & State Transitions",
    )
    print(f"  -> Cluster timeline PNG    : {timeline_outputs['timeline']}")
    print(f"  -> State transitions PNG   : {timeline_outputs['transitions']}")

    # Copy transition matrix CSV to tables/
    t_mat = pd.read_csv(timeline_outputs["transition_matrix_csv"], index_col=0)
    t_mat.to_csv(tables_dir / "state_transition_matrix.csv")
    print(f"  -> Transition matrix table : {tables_dir / 'state_transition_matrix.csv'}")
    print("     Empirical 1-step jump probabilities:")
    for row_name, row in t_mat.iterrows():
        transitions_str = ", ".join(f"{col}: {val:.1%}" for col, val in row.items())
        print(f"     {row_name} -> {transitions_str}")

    # -------------------------------------------------------------------------
    # 4. Feature Importance Bar Plots & Decision Tree If-Then Rules
    # -------------------------------------------------------------------------
    print("\n[4/6] Visualizing feature importances and Decision Tree if-then rules...")
    feat_imp_png = figures_dir / "feature_importance_ranking.png"
    plot_feature_importance(
        importance_df=df_importances,
        output_path=feat_imp_png,
        title="2RJY Villin Headpiece: Top Discriminative Features (Gini Importance)",
        color="#2b6cb0",
        top_n=12,
    )
    print(f"  -> Feature importance bar plot : {feat_imp_png}")

    # Decision tree rules
    dt_rules_png = figures_dir / "decision_tree_rules.png"
    if df_features is not None and not df_features.empty and df_states is not None:
        labels = df_states["State_Label"] if "State_Label" in df_states.columns else df_states["State"]
        explainer = DecisionTreeExplainer(max_depth=3, min_samples_leaf=10, random_state=42)
        explainer.fit(df_features, labels)
        plot_decision_tree_rules(explainer, dt_rules_png, top_n=6, title="2RJY Conformational State Decision Rules")
        rules_df = explainer.rules_dataframe()
        rules_df.to_csv(tables_dir / "decision_tree_rules.csv", index=False)
        print(f"  -> Decision tree rules diagram : {dt_rules_png}")
        print(f"  -> Extracted {len(rules_df)} boolean rules:")
        for _, r in rules_df.head(4).iterrows():
            print(f"     * State {r['Cluster']} (F1={r['F1']:.2f}): {r['Rule']}")
    else:
        # Use existing rules or synthetic rule table
        if precomputed.is_dir() and (precomputed / "decision_tree_rules.csv").is_file():
            df_rules = pd.read_csv(precomputed / "decision_tree_rules.csv")
        else:
            df_rules = pd.DataFrame([
                {"Cluster": "state_0", "Rule": "IF THR_24__SER_43 <= 8.2 Å THEN State 0", "Precision": 0.98, "Recall": 0.96, "F1": 0.97, "Support": 960, "Total_Samples": 980},
                {"Cluster": "state_1", "Rule": "IF THR_24__SER_43 > 8.2 Å THEN State 1", "Precision": 0.96, "Recall": 0.97, "F1": 0.965, "Support": 970, "Total_Samples": 1010},
            ])
        plot_decision_tree_rules(df_rules, dt_rules_png, top_n=4, title="2RJY Conformational State Decision Rules")
        df_rules.to_csv(tables_dir / "decision_tree_rules.csv", index=False)
        print(f"  -> Decision tree rules diagram : {dt_rules_png}")

    # -------------------------------------------------------------------------
    # 5. PDB Beta-Factor Importance Mapping & PyMOL Visualization Script
    # -------------------------------------------------------------------------
    print("\n[5/6] Mapping per-residue importance into PDB Beta-factors & generating 3D scripts...")
    viz_3d = generate_3d_visualization(
        topology_path=pdb_path,
        importance_df=df_importances,
        output_dir=viz_dir,
        basename="2rjy_importance",
        top_label_count=8,
        normalize=True,
        max_scale=100.0,
    )
    print(f"  -> Annotated PDB (B-factors 0-100) : {viz_3d['pdb']}")
    print(f"  -> One-click PyMOL script (.pml)   : {viz_3d['pymol_script']}")
    print(f"  -> UCSF ChimeraX script (.cxc)     : {viz_3d['chimerax_script']}")
    print(f"  -> Residue scores table            : {viz_3d['residue_scores_csv']}")
    print(f"  -> Top scored residues             : {viz_3d['top_residues']}")

    # Also save standard molscope_3d.pdb for HTML report compatibility
    generate_3d_visualization(
        topology_path=pdb_path,
        importance_df=df_importances,
        output_dir=viz_dir,
        basename="molscope_3d",
        top_label_count=8,
    )

    # -------------------------------------------------------------------------
    # 6. Consolidated Interactive HTML Report Dashboard
    # -------------------------------------------------------------------------
    print("\n[6/6] Assembling consolidated interactive HTML dashboard...")
    dt_rules_for_report = pd.read_csv(tables_dir / "decision_tree_rules.csv") if (tables_dir / "decision_tree_rules.csv").is_file() else None
    report_path = generate_html_report(
        destination=out_dir,
        outputs={
            "sam_mfp_candidates": df_importances,
            "state_populations": df_states.groupby(["trajectory_id", "State"]).size().reset_index(name="Population") if "trajectory_id" in df_states.columns else df_states["State"].value_counts().reset_index(name="Population"),
            "decision_tree_rules": dt_rules_for_report,
        },
        manifest=df_manifest if df_manifest is not None else pd.DataFrame({"replica_id": ["run1"] * len(df_states)}),
        features=df_features if df_features is not None else pd.DataFrame(np.zeros((len(df_states), 10))),
    )
    print(f"  -> Consolidated HTML Dashboard     : {report_path}")

    print("\n" + "=" * 70)
    print("SUCCESS: 2RJY Interpretability & Visualization Complete!")
    print("=" * 70)
    print(f"Open dashboard: file://{report_path}")
    print("PyMOL command : pymol", viz_3d["pymol_script"])
    print("ChimeraX      : chimerax", viz_3d["chimerax_script"])
    print("Sidecar viewer: [Biomolecular Structure Viewer](sidecar://science/biomolecular_structure_viewer/ui.html)")
    print("=" * 70)
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MolScope interpretability & visualization tools on 2RJY Villin Headpiece."
    )
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR), help="Destination directory for figures and reports.")
    parser.add_argument("--pdb", default=str(DEFAULT_REFERENCE_PDB), help="Reference PDB structure file.")
    parser.add_argument("--precomputed-dir", default=str(DEFAULT_PRECOMPUTED_DIR), help="Directory containing precomputed benchmark tables.")
    parser.add_argument("--project-json", default=None, help="Optional project.json to run from scratch.")

    args = parser.parse_args()
    run_interpretability_pipeline(
        output_dir=args.out_dir,
        reference_pdb=args.pdb,
        precomputed_dir=args.precomputed_dir,
        project_json=args.project_json,
    )


if __name__ == "__main__":
    main()
