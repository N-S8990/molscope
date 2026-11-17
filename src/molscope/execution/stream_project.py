"""Run the limited streaming workflow from project.json with explicit identities."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from molscope.core.project import load_project_spec
from molscope.execution.stream_analysis import StreamAnalysisConfig, analyze_stream
from molscope.execution.streaming import CA, GEOMETRY, StreamingConfig, extract_streaming
from molscope.profiling.system import profile_system
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.quality.assessment import _replica_consistency
from scipy.stats import rankdata
import numpy as np
import pandas as pd


def streaming_config(project, *, chunk_size=1000, memory_budget_mb=512,
                     geometry_selection="all", imaging_policy="as_stored_nonperiodic"):
    parameters = dict(project.analysis.generic_feature_parameters)
    unsupported = set(parameters) - {"max_frames", "frame_stride", "use_pbc"}
    if unsupported:
        raise ValueError(f"Generic parameters unsupported by streaming: {sorted(unsupported)}")
    if parameters.pop("use_pbc", False) is not False:
        raise ValueError("Streaming requires use_pbc=false and an explicit as-stored imaging policy")
    ca_parameters = project.analysis.plugin_parameters.get(CA, {})
    unsupported = set(ca_parameters) - {"selection", "contact_cutoff_angstrom", "minimum_sequence_separation", "max_pairs"}
    if unsupported:
        raise ValueError(f"C-alpha parameters unsupported by streaming: {sorted(unsupported)}")
    for name, values in project.analysis.plugin_parameters.items():
        if name != CA and values:
            raise ValueError(f"Configured plugin {name} has no streaming parameter implementation")
    mapped = {("ca_selection" if key == "selection" else key): value for key, value in ca_parameters.items()}
    return StreamingConfig(chunk_size=chunk_size, memory_budget_mb=memory_budget_mb,
                           geometry_selection=geometry_selection, imaging_policy=imaging_policy,
                           plugins=(GEOMETRY,) + project.analysis.plugins,
                           **parameters, **mapped)


def run_stream_project(project_path, out_dir, *, analysis_config=None, chunk_size=1000,
                       memory_budget_mb=512, geometry_selection="all",
                       imaging_policy="as_stored_nonperiodic"):
    project = load_project_spec(project_path)
    missing = project.validate_paths()
    if missing:
        raise ValueError(f"Missing project inputs: {missing}")
    config = streaming_config(project, chunk_size=chunk_size, memory_budget_mb=memory_budget_mb,
                              geometry_selection=geometry_selection, imaging_policy=imaging_policy)
    analysis_config = analysis_config or StreamAnalysisConfig(memory_budget_mb=int(memory_budget_mb))
    inputs = project.trajectory_inputs()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=False)
    report = {"project_id": project.project_id, "status": "running", "trajectories": [],
              "scope": "Independent sample-trained per-trajectory descriptive analyses; state IDs are not comparable between trajectories.",
              "omitted_stages": ["predictive ML", "cross-replica stability", "SAM-MFP prioritization"],
              "extraction_configuration": asdict(config),
              "declared_labels": [asdict(label) for label in project.labels],
              "declared_prediction_tasks_not_executed": [asdict(task) for task in project.prediction_tasks]}
    (root / "project_run.json").write_text(json.dumps(report, indent=2) + "\n")
    for index, item in enumerate(inputs):
        # Numeric paths avoid filesystem collisions and traversal from user IDs.
        relative = f"trajectory_{index:05d}"
        directory = root / relative
        directory.mkdir()
        identity = {"trajectory_id": item.trajectory_id, "system_id": item.system_id,
                    "ensemble_id": item.ensemble_id, "replica_id": item.replica_id,
                    "metadata": item.metadata}
        (directory / "identity.json").write_text(json.dumps(identity, indent=2) + "\n")
        profile = profile_system(item.topology, item.trajectory)
        (directory / "system_profile.json").write_text(json.dumps(profile.to_dict(), indent=2) + "\n")
        manifest = extract_streaming(item.topology, item.trajectory, directory / "features", config=config)
        analysis = analyze_stream(manifest.parent, directory / "analysis", analysis_config)
        report["trajectories"].append({**identity, "directory": relative,
                                       "frames_analyzed": analysis["frames_analyzed"]})

    if analysis_config.compute_sam_mfp and len(inputs) > 0:
        traj_quality_list, traj_salience_list, traj_exp_list = [], [], []
        for item_entry in report["trajectories"]:
            t_dir = root / item_entry["directory"] / "analysis"
            if (t_dir / "feature_quality.csv").is_file():
                traj_quality_list.append(pd.read_csv(t_dir / "feature_quality.csv"))
            if (t_dir / "representation_salience.csv").is_file():
                traj_salience_list.append(pd.read_csv(t_dir / "representation_salience.csv"))
            if (t_dir / "state_explanatory_ranking.csv").is_file():
                traj_exp_list.append(pd.read_csv(t_dir / "state_explanatory_ranking.csv"))

        if traj_quality_list and traj_salience_list:
            feature_cols = traj_quality_list[0]["Feature"].tolist()
            # Aggregate quality metrics across trajectories
            agg_quality = traj_quality_list[0].copy()
            for col in ["ESS_Ratio", "Occupancy_Entropy", "Uniqueness"]:
                stacked = np.column_stack([df[col].to_numpy() for df in traj_quality_list])
                agg_quality[col] = np.nanmedian(stacked, axis=1)

            # Cross-replica stability across trajectories
            if len(traj_salience_list) >= 2:
                sal_stacked = np.column_stack([df["PCA_Salience"].to_numpy() for df in traj_salience_list])
                mean_sal = np.nanmean(sal_stacked, axis=1)
                ranks_stacked = np.column_stack([rankdata(-sal_stacked[:, col_idx], method="average") for col_idx in range(sal_stacked.shape[1])])
                loto_std = np.std(ranks_stacked, axis=1, ddof=1)
                loto_norm = loto_std / max(1, len(feature_cols) - 1)
                loto_folds = len(traj_salience_list)
                agg_salience = pd.DataFrame({
                    "Feature": feature_cols,
                    "PCA_Salience": mean_sal,
                    "PCA_Rank": rankdata(-mean_sal, method="average"),
                    "LOTO_Rank_Median": np.median(ranks_stacked, axis=1),
                    "LOTO_Rank_Std": loto_std,
                    "LOTO_Folds": loto_folds,
                    "LOTO_Rank_Std_Normalized": loto_norm,
                })
            else:
                agg_salience = traj_salience_list[0].copy()

            agg_exp = traj_exp_list[0].copy() if traj_exp_list else None
            if traj_exp_list and len(traj_exp_list) >= 2:
                exp_stacked = np.column_stack([df["State_Explained_Variance_Eta_Squared"].to_numpy() for df in traj_exp_list])
                agg_exp["State_Explained_Variance_Eta_Squared"] = np.nanmean(exp_stacked, axis=1)

            proj_candidates, proj_notes = prioritize_sam_mfp(
                agg_quality, agg_salience,
                SAMMFPConfig(min_loto_folds=analysis_config.min_loto_folds, include_state_separation=True),
                state_explanatory=agg_exp,
            )
            proj_candidates.to_csv(root / "project_sam_mfp_candidates.csv", index=False)
            agg_quality.to_csv(root / "project_feature_quality.csv", index=False)
            report["project_sam_mfp_candidates_file"] = "project_sam_mfp_candidates.csv"
            report["project_sam_mfp_candidate_count"] = int(proj_candidates["SAM_MFP_Pareto_Front"].sum())
            report["project_sam_mfp_candidates"] = proj_candidates.loc[proj_candidates["SAM_MFP_Pareto_Front"], "Feature"].tolist()
            report["omitted_stages"] = ["predictive ML"]

    report["status"] = "completed"
    temporary = root / "project_run.json.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(root / "project_run.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--batch-rows", type=int, default=512)
    parser.add_argument("--sample-rows", type=int, default=2000)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--components", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--memory-budget-mb", type=int, default=512)
    parser.add_argument("--geometry-selection", default="all")
    parser.add_argument("--imaging-policy", choices=["as_stored_nonperiodic"], default="as_stored_nonperiodic")
    parser.add_argument("--sam-mfp", action="store_true", default=False, help="Compute bounded-memory SAM-MFP Pareto candidates")
    args = parser.parse_args()
    analysis = StreamAnalysisConfig(batch_rows=args.batch_rows, sample_rows=args.sample_rows,
                                    states=args.states, components=args.components, seed=args.seed,
                                    memory_budget_mb=args.memory_budget_mb, compute_sam_mfp=args.sam_mfp)
    report = run_stream_project(args.project, args.out_dir, analysis_config=analysis,
                                chunk_size=args.chunk_size, memory_budget_mb=args.memory_budget_mb,
                                geometry_selection=args.geometry_selection, imaging_policy=args.imaging_policy)
    print(json.dumps({"status": report["status"], "trajectories": len(report["trajectories"])}))


if __name__ == "__main__":
    main()
