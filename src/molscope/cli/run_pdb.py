"""Execute an end-to-end 3-stage PDB pipeline:
1. OpenMM MD Simulation: Minimization, equilibration, and Langevin production.
2. Labeling & Clustering: Biophysical feature extraction, PCA, and DPA/GMM state discovery.
3. Machine Learning & Global Feature Ranking: Regularized predictive ML and ranking across all features.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from molscope.core.project import (
    EnsembleSpec,
    OpenMMSimulationSpec,
    ProjectSpec,
    ReplicaSpec,
    SystemSpec,
)
from molscope.execution.discovery import DiscoveryPipelineConfig
from molscope.execution.pipeline import MDContactMLPipeline
from molscope.simulation.openmm import build_openmm_plan, execute_openmm_plan
from molscope.visualization.pdb_mapping import map_feature_importance_to_pdb


def run_pdb_pipeline(
    pdb_path: str | Path,
    out_dir: str | Path,
    *,
    production_steps: int = 50_000,
    report_interval_steps: int = 500,
    temperature_kelvin: float = 300.0,
    ph: float = 7.4,
    states: int = 2,
    supervised_splits: int = 3,
    platform: str | None = None,
) -> Path:
    """Run simulation, clustering/labeling, and ML feature prioritization end-to-end."""
    pdb = Path(pdb_path).resolve()
    if not pdb.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {pdb}")

    destination = Path(out_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    sim_output_root = destination / "sim_output"
    sim_output_root.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # STAGE 1: OpenMM Molecular Dynamics Simulation
    # ---------------------------------------------------------
    print(f"\n[Stage 1/3] Initiating OpenMM simulation for {pdb.name}...")
    print(f"  Production steps: {production_steps} ({production_steps * 2.0 / 1000.0:.1f} ps)")
    print(f"  Report interval: {report_interval_steps} steps")
    print(f"  Temperature: {temperature_kelvin} K | pH: {ph}")

    protocol = OpenMMSimulationSpec(
        forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
        nonbonded_method="NoCutoff",
        constraints="HBonds",
        temperature_kelvin=temperature_kelvin,
        friction_per_picosecond=1.0,
        timestep_femtoseconds=2.0,
        minimization_max_iterations=500,
        equilibration_steps=1_000,
        production_steps=production_steps,
        report_interval_steps=report_interval_steps,
        ph=ph,
        platform=platform,
    )

    system_spec = SystemSpec(
        system_id="system_1",
        input_structure=str(pdb),
        simulation_protocol=protocol,
        ensembles=[
            EnsembleSpec(
                ensemble_id="production",
                replicas=[
                    ReplicaSpec(
                        replica_id="replica_1",
                        trajectory="simulations/system_1/production/replica_1/trajectory.dcd",
                    )
                ],
            )
        ],
    )

    sim_project = ProjectSpec(
        schema_version="1",
        project_id=f"sim_{pdb.stem}",
        systems=[system_spec],
    )

    plan = build_openmm_plan(sim_project, sim_output_root)
    execute_openmm_plan(plan)

    job_outputs = plan["jobs"][0]
    prepared_pdb = job_outputs["prepared_topology"]
    trajectory_dcd = job_outputs["trajectory"]
    print(f"  Trajectory successfully generated: {trajectory_dcd}")
    print("[Stage 1/3 Complete] OpenMM simulation finished successfully.")

    # ---------------------------------------------------------
    # STAGE 2: Feature Extraction, Labeling & Clustering
    # ---------------------------------------------------------
    print("\n[Stage 2/3] Extracting biophysical features, discovering states & DPA clusters...")
    analysis_system = SystemSpec(
        system_id="system_1",
        input_structure=prepared_pdb,
        ensembles=[
            EnsembleSpec(
                ensemble_id="production",
                replicas=[ReplicaSpec(replica_id="replica_1", trajectory=trajectory_dcd)],
            )
        ],
    )

    analysis_project = ProjectSpec(
        schema_version="1",
        project_id=f"analysis_{pdb.stem}",
        systems=[analysis_system],
    )

    pipeline = MDContactMLPipeline(
        DiscoveryPipelineConfig(
            state_count=states,
            supervised_splits=supervised_splits,
        )
    ).load_project(analysis_project)

    # ---------------------------------------------------------
    # STAGE 3: Machine Learning & Global Feature Ranking
    # ---------------------------------------------------------
    print("\n[Stage 3/3] Training regularized ML model and ranking all features across all data...")
    res_dest = pipeline.run(target_column=None).write_results(destination)

    # Map residue importances to PDB Beta-factors for 3D visualization
    try:
        glob_imp_path = res_dest / "global_feature_importance_all_features.csv"
        if glob_imp_path.is_file():
            df_imp = pd.read_csv(glob_imp_path)
            mapped_pdb_path = res_dest / "mapped_importance.pdb"
            map_feature_importance_to_pdb(prepared_pdb, df_imp, mapped_pdb_path)
            print(f"  Residue importances mapped to PDB Beta-factors: {mapped_pdb_path}")
    except Exception as exc:
        print(f"  Note: PDB Beta-factor mapping skipped: {exc}")

    print("\n=======================================================")
    print("MolScope End-to-End Pipeline Completed Successfully!")
    print(f"All results, clusters, ML models, and reports written to:")
    print(f"  {res_dest}")
    print("=======================================================\n")
    return res_dest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run an end-to-end 3-stage MolScope workflow: PDB -> Simulation -> Clustering/Labeling -> ML Feature Ranking."
    )
    parser.add_argument("--pdb", required=True, help="Input structure PDB file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for simulation and analysis artifacts.")
    parser.add_argument("--steps", type=int, default=50_000, help="Production simulation steps (default: 50,000 = 100 ps).")
    parser.add_argument("--report-interval", type=int, default=500, help="Reporting frame interval steps (default: 500 = 1 ps).")
    parser.add_argument("--temperature", type=float, default=300.0, help="Simulation temperature in Kelvin (default: 300.0).")
    parser.add_argument("--ph", type=float, default=7.4, help="Protonation state target pH (default: 7.4).")
    parser.add_argument("--states", type=int, default=2, help="Number of conformational macrostates to discover (default: 2).")
    parser.add_argument("--supervised-splits", type=int, default=3, help="Number of cross-validation folds (default: 3).")
    parser.add_argument("--platform", default=None, help="OpenMM platform (Reference, CPU, CUDA, OpenCL; default: automatic).")

    args = parser.parse_args()
    run_pdb_pipeline(
        pdb_path=args.pdb,
        out_dir=args.out_dir,
        production_steps=args.steps,
        report_interval_steps=args.report_interval,
        temperature_kelvin=args.temperature,
        ph=args.ph,
        states=args.states,
        supervised_splits=args.supervised_splits,
        platform=args.platform,
    )


if __name__ == "__main__":
    main()
