"""Integration tests for streaming SAM-MFP candidate selection, replica stability, and in-memory parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molscope.analysis.discovery.representation import RepresentationDiscoveryConfig, discover_representation_features
from molscope.analysis.discovery.states import StateDiscoveryConfig, discover_conformational_states, rank_state_explanatory_features
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.quality.assessment import assess_feature_quality
from molscope.execution.stream_analysis import StreamAnalysisConfig, analyze_stream
from molscope.execution.stream_project import run_stream_project
from molscope.project_spec import AnalysisSpec, ProjectSpec


def _create_synthetic_chunk_store(
    path: Path,
    n_trajectories: int = 1,
    frames_per_traj: int = 100,
    chunk_rows: int = 25,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create a valid chunk store with manifest.json and CSV chunks."""
    path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    total_frames = n_trajectories * frames_per_traj

    t_arr = np.linspace(0, 10, total_frames)
    # Plant distinct signals
    f_signal_a = np.sin(t_arr) * 3.0 + rng.normal(0, 0.1, total_frames)
    f_signal_b = np.cos(2 * t_arr) * 2.0 + rng.normal(0, 0.1, total_frames)
    f_noise = rng.normal(0, 1.0, total_frames)
    # Plant erratic feature: large variance in traj 0, flat in remaining
    f_erratic = rng.normal(0, 0.05, total_frames)
    if n_trajectories > 1:
        f_erratic[:frames_per_traj] = np.sin(t_arr[:frames_per_traj]) * 10.0

    df_features = pd.DataFrame({
        "sig_a": f_signal_a,
        "sig_b": f_signal_b,
        "noise": f_noise,
        "erratic": f_erratic,
    })

    traj_col = [f"traj_{i}" for i in range(n_trajectories) for _ in range(frames_per_traj)]
    replica_col = [f"rep_{i}" for i in range(n_trajectories) for _ in range(frames_per_traj)]
    ensemble_col = ["ens_0"] * total_frames

    df_manifest = pd.DataFrame({
        "trajectory_id": traj_col,
        "replica_id": replica_col,
        "ensemble_id": ensemble_col,
        "frame_index": np.arange(total_frames),
        "frame_order": np.arange(total_frames),
        "time_ps": np.arange(total_frames, dtype=float),
    })

    full_table = pd.concat([df_manifest, df_features], axis=1)

    chunks = []
    chunk_idx = 0
    for start in range(0, total_frames, chunk_rows):
        chunk_slice = full_table.iloc[start : start + chunk_rows]
        chunk_file = f"chunk_{chunk_idx:06d}.csv"
        chunk_slice.to_csv(path / chunk_file, index=False)
        chunks.append({"file": chunk_file, "rows": len(chunk_slice)})
        chunk_idx += 1

    manifest_payload = {
        "schema_version": 1,
        "complete": True,
        "feature_columns": ["sig_a", "sig_b", "noise", "erratic"],
        "chunks": chunks,
        "total_rows": total_frames,
    }
    (path / "manifest.json").write_text(json.dumps(manifest_payload, indent=2))

    return df_features, df_manifest


def test_stream_analysis_sam_mfp_single_trajectory(tmp_path):
    """Test streaming SAM-MFP on a single trajectory with bounded batches."""
    store_dir = tmp_path / "single_store"
    out_dir = tmp_path / "single_analysis"
    df_features, _ = _create_synthetic_chunk_store(store_dir, n_trajectories=1, frames_per_traj=120, chunk_rows=30)

    config = StreamAnalysisConfig(
        compute_sam_mfp=True,
        batch_rows=30,
        sample_rows=100,
        components=2,
        states=2,
        seed=42,
    )
    report = analyze_stream(store_dir, out_dir, config=config)

    assert report["status"] == "completed"
    assert "SAM-MFP" in report["scope"]
    assert report["sam_mfp_candidate_count"] >= 1
    assert len(report["sam_mfp_candidates"]) >= 1

    # Check that output CSV files exist and are well-formed
    cand_path = out_dir / "sam_mfp_candidates.csv"
    q_path = out_dir / "feature_quality.csv"
    sal_path = out_dir / "representation_salience.csv"
    exp_path = out_dir / "state_explanatory_ranking.csv"

    assert cand_path.is_file()
    assert q_path.is_file()
    assert sal_path.is_file()
    assert exp_path.is_file()

    cand_df = pd.read_csv(cand_path)
    assert set(cand_df["Feature"]) == set(df_features.columns)
    assert "SAM_MFP_Pareto_Front" in cand_df.columns
    assert "SAM_MFP_Dominator_Count" in cand_df.columns
    assert cand_df["SAM_MFP_Pareto_Front"].any()

    # Single trajectory: Replica_Consistency is NaN, and single-trajectory note is present
    q_df = pd.read_csv(q_path)
    assert q_df["Replica_Consistency"].isna().all()
    assert any("Single trajectory provided" in line for line in report["limitations"])


def test_stream_analysis_sam_mfp_multi_trajectory_replica_stability(tmp_path):
    """Test streaming SAM-MFP with multiple replica trajectories and LOTO stability."""
    store_dir = tmp_path / "multi_store"
    out_dir = tmp_path / "multi_analysis"
    df_features, df_manifest = _create_synthetic_chunk_store(
        store_dir, n_trajectories=3, frames_per_traj=100, chunk_rows=25, seed=42,
    )

    config = StreamAnalysisConfig(
        compute_sam_mfp=True,
        min_loto_folds=2,
        batch_rows=25,
        sample_rows=300,
        components=3,
        states=2,
        seed=42,
    )
    report = analyze_stream(store_dir, out_dir, config=config)

    assert report["status"] == "completed"
    cand_df = pd.read_csv(out_dir / "sam_mfp_candidates.csv")
    q_df = pd.read_csv(out_dir / "feature_quality.csv")
    sal_df = pd.read_csv(out_dir / "representation_salience.csv")

    # Multi-replica: Replica_Consistency must be finite
    assert q_df["Replica_Consistency"].notna().all()
    assert (q_df["Replica_Consistency"] > 0).all()

    # LOTO Folds must be 3
    assert (sal_df["LOTO_Folds"] == 3).all()

    # Erratic feature must have finite LOTO instability
    sal_indexed = sal_df.set_index("Feature")
    assert sal_indexed.loc["erratic", "LOTO_Rank_Std_Normalized"] > 0.0

    # Representation_Stability must be included in SAM-MFP objectives
    assert "Representation_Stability" in cand_df["SAM_MFP_Objectives"].iloc[0]


def test_streaming_parity_with_in_memory_workflow(tmp_path):
    """Task 4 verification: verify streaming matches in-memory workflow within numerical tolerance."""
    store_dir = tmp_path / "parity_store"
    out_dir = tmp_path / "parity_analysis"

    n_trajectories = 2
    frames_per_traj = 100
    total_frames = n_trajectories * frames_per_traj

    df_features, df_manifest = _create_synthetic_chunk_store(
        store_dir, n_trajectories=n_trajectories, frames_per_traj=frames_per_traj, chunk_rows=25, seed=42,
    )

    # 1. In-memory workflow execution
    q_mem = assess_feature_quality(df_features, df_manifest)
    rep_mem = discover_representation_features(
        df_features, df_manifest, RepresentationDiscoveryConfig(n_components=2),
    )
    states_mem = discover_conformational_states(
        rep_mem["representation_coordinates"], df_manifest, StateDiscoveryConfig(n_states=2, random_state=42),
    )
    exp_mem = rank_state_explanatory_features(df_features, states_mem["state_assignments"])
    cand_mem, _ = prioritize_sam_mfp(
        q_mem, rep_mem["representation_feature_salience"],
        SAMMFPConfig(include_state_separation=True),
        state_explanatory=exp_mem,
    )

    # 2. Streaming workflow execution (sample_rows >= total_frames to retain full dataset in reservoir)
    config = StreamAnalysisConfig(
        compute_sam_mfp=True,
        batch_rows=25,
        sample_rows=total_frames,
        components=2,
        states=2,
        seed=42,
    )
    analyze_stream(store_dir, out_dir, config=config)

    # Load streaming outputs
    stream_cand = pd.read_csv(out_dir / "sam_mfp_candidates.csv").set_index("Feature")
    stream_q = pd.read_csv(out_dir / "feature_quality.csv").set_index("Feature")
    stream_sal = pd.read_csv(out_dir / "representation_salience.csv").set_index("Feature")
    stream_exp = pd.read_csv(out_dir / "state_explanatory_ranking.csv").set_index("Feature")

    mem_cand = cand_mem.set_index("Feature")
    mem_q = q_mem.set_index("Feature")
    mem_sal = rep_mem["representation_feature_salience"].set_index("Feature")
    mem_exp = exp_mem.set_index("Feature")

    feature_order = df_features.columns.tolist()

    # Numerical parity checks
    # Quality metrics
    np.testing.assert_allclose(
        stream_q.loc[feature_order, "Valid_Fraction"],
        mem_q.loc[feature_order, "Valid_Fraction"],
        atol=1e-7,
        err_msg="Valid_Fraction parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[feature_order, "ESS_Ratio"],
        mem_q.loc[feature_order, "ESS_Ratio"],
        atol=0.4,
        err_msg="ESS_Ratio parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[feature_order, "Occupancy_Entropy"],
        mem_q.loc[feature_order, "Occupancy_Entropy"],
        atol=1e-5,
        err_msg="Occupancy_Entropy parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[feature_order, "Uniqueness"],
        mem_q.loc[feature_order, "Uniqueness"],
        atol=1e-5,
        err_msg="Uniqueness parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[feature_order, "Replica_Consistency"],
        mem_q.loc[feature_order, "Replica_Consistency"],
        atol=1e-5,
        err_msg="Replica_Consistency parity mismatch",
    )

    # Representation salience & LOTO stability
    np.testing.assert_allclose(
        stream_sal.loc[feature_order, "PCA_Salience"],
        mem_sal.loc[feature_order, "PCA_Salience"],
        atol=1e-5,
        err_msg="PCA_Salience parity mismatch",
    )
    np.testing.assert_allclose(
        stream_sal.loc[feature_order, "LOTO_Rank_Std_Normalized"],
        mem_sal.loc[feature_order, "LOTO_Rank_Std_Normalized"],
        atol=1e-5,
        err_msg="LOTO_Rank_Std_Normalized parity mismatch",
    )

    # State explanatory ANOVA eta-squared
    np.testing.assert_allclose(
        stream_exp.loc[feature_order, "State_Explained_Variance_Eta_Squared"],
        mem_exp.loc[feature_order, "State_Explained_Variance_Eta_Squared"],
        atol=1e-5,
        err_msg="State_Explained_Variance_Eta_Squared parity mismatch",
    )

    # SAM-MFP Pareto Candidates
    np.testing.assert_array_equal(
        stream_cand.loc[feature_order, "SAM_MFP_Pareto_Front"].to_numpy(dtype=bool),
        mem_cand.loc[feature_order, "SAM_MFP_Pareto_Front"].to_numpy(dtype=bool),
        err_msg="SAM_MFP_Pareto_Front boolean parity mismatch",
    )
    np.testing.assert_array_equal(
        stream_cand.loc[feature_order, "SAM_MFP_Dominator_Count"].to_numpy(dtype=int),
        mem_cand.loc[feature_order, "SAM_MFP_Dominator_Count"].to_numpy(dtype=int),
        err_msg="SAM_MFP_Dominator_Count integer parity mismatch",
    )


def test_2rjy_subset_streaming_vs_in_memory_parity(tmp_path):
    """Task 4 verification on real 2RJY Villin Headpiece subset."""
    pdb_path = Path("data/2RJY/protein.pdb")
    xtc_path = Path("data/2RJY/simulation_run1.xtc")

    if not (pdb_path.is_file() and xtc_path.is_file()):
        pytest.skip("2RJY benchmark inputs not available locally")

    from molscope.execution.streaming import StreamingConfig, extract_streaming

    store_dir = tmp_path / "2rjy_store"
    out_dir = tmp_path / "2rjy_analysis"

    # Extract 50 frames with chunk size 25
    streaming_cfg = StreamingConfig(
        chunk_size=25,
        max_frames=50,
        geometry_selection="protein",
        ca_selection="protein and name CA and resid 40:50",
        max_pairs=100,
    )
    manifest_path = extract_streaming(pdb_path, xtc_path, store_dir, config=streaming_cfg)
    manifest = json.loads(manifest_path.read_text())

    # Assemble feature matrix for in-memory comparison
    chunks_dfs = [pd.read_csv(store_dir / c["file"]) for c in manifest["chunks"]]
    full_table = pd.concat(chunks_dfs, ignore_index=True)
    feature_cols = manifest["feature_columns"]

    df_features = full_table[feature_cols].copy()
    df_manifest = pd.DataFrame({
        "trajectory_id": ["2rjy_traj"] * len(full_table),
        "replica_id": ["2rjy_rep"] * len(full_table),
        "ensemble_id": ["2rjy_ens"] * len(full_table),
        "frame_index": full_table["frame_index"].to_numpy(),
        "frame_order": full_table["frame_order"].to_numpy(),
    })

    # In-memory execution
    q_mem = assess_feature_quality(df_features, df_manifest)
    rep_mem = discover_representation_features(
        df_features, df_manifest, RepresentationDiscoveryConfig(n_components=3),
    )
    states_mem = discover_conformational_states(
        rep_mem["representation_coordinates"], df_manifest, StateDiscoveryConfig(n_states=2, random_state=42),
    )
    exp_mem = rank_state_explanatory_features(df_features, states_mem["state_assignments"])
    cand_mem, _ = prioritize_sam_mfp(
        q_mem, rep_mem["representation_feature_salience"],
        SAMMFPConfig(include_state_separation=True),
        state_explanatory=exp_mem,
    )

    # Streaming execution
    stream_cfg = StreamAnalysisConfig(
        compute_sam_mfp=True,
        batch_rows=25,
        sample_rows=50,
        components=3,
        states=2,
        seed=42,
    )
    analyze_stream(store_dir, out_dir, config=stream_cfg)

    stream_cand = pd.read_csv(out_dir / "sam_mfp_candidates.csv").set_index("Feature")
    stream_q = pd.read_csv(out_dir / "feature_quality.csv").set_index("Feature")
    stream_sal = pd.read_csv(out_dir / "representation_salience.csv").set_index("Feature")
    stream_exp = pd.read_csv(out_dir / "state_explanatory_ranking.csv").set_index("Feature")

    mem_cand = cand_mem.set_index("Feature")
    mem_q = q_mem.set_index("Feature")
    mem_sal = rep_mem["representation_feature_salience"].set_index("Feature")
    mem_exp = exp_mem.set_index("Feature")

    # Parity checks on active molecular features
    active_cols = rep_mem["representation_feature_salience"]["Feature"].tolist()
    assert len(active_cols) > 0

    np.testing.assert_allclose(
        stream_q.loc[active_cols, "ESS_Ratio"],
        mem_q.loc[active_cols, "ESS_Ratio"],
        atol=1e-5,
        err_msg="2RJY ESS_Ratio parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[active_cols, "Occupancy_Entropy"],
        mem_q.loc[active_cols, "Occupancy_Entropy"],
        atol=1e-5,
        err_msg="2RJY Occupancy_Entropy parity mismatch",
    )
    np.testing.assert_allclose(
        stream_q.loc[active_cols, "Uniqueness"],
        mem_q.loc[active_cols, "Uniqueness"],
        atol=1e-5,
        err_msg="2RJY Uniqueness parity mismatch",
    )
    np.testing.assert_allclose(
        stream_sal.loc[active_cols, "PCA_Salience"],
        mem_sal.loc[active_cols, "PCA_Salience"],
        atol=1e-5,
        err_msg="2RJY PCA_Salience parity mismatch",
    )
    np.testing.assert_allclose(
        stream_exp.loc[active_cols, "State_Explained_Variance_Eta_Squared"],
        mem_exp.loc[active_cols, "State_Explained_Variance_Eta_Squared"],
        atol=1e-5,
        err_msg="2RJY State_Explained_Variance_Eta_Squared parity mismatch",
    )
    np.testing.assert_array_equal(
        stream_cand.loc[active_cols, "SAM_MFP_Pareto_Front"].to_numpy(dtype=bool),
        mem_cand.loc[active_cols, "SAM_MFP_Pareto_Front"].to_numpy(dtype=bool),
        err_msg="2RJY SAM_MFP_Pareto_Front mismatch",
    )
    np.testing.assert_array_equal(
        stream_cand.loc[active_cols, "SAM_MFP_Dominator_Count"].to_numpy(dtype=int),
        mem_cand.loc[active_cols, "SAM_MFP_Dominator_Count"].to_numpy(dtype=int),
        err_msg="2RJY SAM_MFP_Dominator_Count mismatch",
    )

