"""Bounded-batch summaries and sample-trained state discovery for chunk stores.

This supports bounded-memory summaries, sample-trained state discovery,
and optional streaming-compatible SAM-MFP Pareto candidate selection with
replica stability integration. All frames contribute to moments, ESS,
and state separation; a seeded uniform reservoir supplies PCA and cluster
training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp
from molscope.analysis.quality.assessment import (
    _maximum_spearman_correlation,
    _occupancy_entropy,
    _replica_consistency,
)


@dataclass(frozen=True)
class StreamAnalysisConfig:
    batch_rows: int = 512
    sample_rows: int = 2000
    components: int = 3
    states: int = 2
    seed: int = 42
    memory_budget_mb: int = 512
    compute_sam_mfp: bool = False
    min_loto_folds: int = 2
    max_autocorrelation_lag: int = 100
    entropy_bins: int = 8

    def __post_init__(self):
        for key in ("batch_rows", "sample_rows", "components", "states", "memory_budget_mb", "min_loto_folds", "max_autocorrelation_lag", "entropy_bins"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not isinstance(self.compute_sam_mfp, bool):
            raise ValueError("compute_sam_mfp must be a boolean")
        if self.sample_rows < max(2, self.states):
            raise ValueError("sample_rows must be at least two and at least states")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")


class RunningMoments:
    """Parallel-Welford summary requiring O(feature count) retained memory."""
    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def update(self, values: np.ndarray) -> None:
        if not len(values):
            return
        count = len(values)
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += np.sum((values - mean) ** 2, axis=0) + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.count = total


class StreamingAutocovariance:
    """Bounded-memory lag-k autocovariance accumulator across batches with trajectory awareness."""
    def __init__(self, width: int, max_lag: int, traj_moments: dict[str, RunningMoments]):
        self.width = width
        self.max_lag = max_lag
        self.traj_moments = traj_moments
        self.traj_data = {
            t_id: {
                "autocov": np.zeros((max_lag + 1, width), dtype=np.float64),
                "counts": np.zeros(max_lag + 1, dtype=np.int64),
                "prev_buffer": np.empty((0, width), dtype=np.float64),
            }
            for t_id in traj_moments
        }
        self.prev_traj = None

    def update(self, values: np.ndarray, trajectory_ids: np.ndarray | None = None) -> None:
        if not len(values):
            return
        if trajectory_ids is not None:
            change_indices = np.where(trajectory_ids[:-1] != trajectory_ids[1:])[0] + 1
            start = 0
            for end in list(change_indices) + [len(values)]:
                traj_slice = values[start:end]
                traj_id = str(trajectory_ids[start])
                self._update_slice(traj_slice, traj_id)
                start = end
        else:
            default_id = next(iter(self.traj_moments.keys()), "default")
            self._update_slice(values, default_id)

    def _update_slice(self, raw_values: np.ndarray, traj_id: str) -> None:
        if traj_id not in self.traj_data:
            self.traj_data[traj_id] = {
                "autocov": np.zeros((self.max_lag + 1, self.width), dtype=np.float64),
                "counts": np.zeros(self.max_lag + 1, dtype=np.int64),
                "prev_buffer": np.empty((0, self.width), dtype=np.float64),
            }
        mean = self.traj_moments[traj_id].mean if traj_id in self.traj_moments else raw_values.mean(axis=0)
        centered = raw_values - mean
        t_data = self.traj_data[traj_id]
        n_prev = len(t_data["prev_buffer"])
        n_curr = len(centered)
        t_data["autocov"][0] += np.sum(centered ** 2, axis=0)
        t_data["counts"][0] += n_curr
        combined = np.vstack([t_data["prev_buffer"], centered]) if n_prev > 0 else centered
        n_comb = len(combined)
        for k in range(1, min(self.max_lag + 1, n_comb)):
            i_start = max(0, n_prev - k)
            i_end = n_comb - k
            if i_end > i_start:
                x_left = combined[i_start:i_end]
                x_right = combined[i_start + k : i_end + k]
                t_data["autocov"][k] += np.sum(x_left * x_right, axis=0)
                t_data["counts"][k] += (i_end - i_start)
        t_data["prev_buffer"] = combined[-self.max_lag:] if n_comb > self.max_lag else combined

    def compute_ess(self, tolerance: float = 1e-12) -> tuple[np.ndarray, np.ndarray]:
        """Compute median ESS and ESS ratio across trajectories."""
        all_ess = []
        all_lens = []
        for t_id, t_mom in self.traj_moments.items():
            n_sub = t_mom.count
            if n_sub < 4:
                continue
            all_lens.append(n_sub)
            t_data = self.traj_data[t_id]
            denom = t_mom.m2
            eff_max_lag = min(self.max_lag, n_sub // 2)
            sub_ess = np.full(self.width, np.nan)
            for j in range(self.width):
                if denom[j] <= tolerance:
                    continue
                rhos = []
                for k in range(1, eff_max_lag + 1):
                    if t_data["counts"][k] == 0:
                        break
                    rho = t_data["autocov"][k, j] / denom[j]
                    if rho <= 0:
                        break
                    rhos.append(float(rho))
                it = 1.0 + 2.0 * sum(rhos)
                sub_ess[j] = float(np.clip(n_sub / it, 1.0, float(n_sub)))
            all_ess.append(sub_ess)

        if not all_ess:
            return np.full(self.width, np.nan), np.full(self.width, np.nan)

        arr = np.asarray(all_ess)
        all_nan = np.all(np.isnan(arr), axis=0)
        med_ess = np.full(self.width, np.nan)
        if not np.all(all_nan):
            med_ess[~all_nan] = np.nanmedian(arr[:, ~all_nan], axis=0)
        med_frames = float(np.median(all_lens)) if all_lens else 0.0
        ess_ratio = np.where(med_frames > 0, med_ess / med_frames, np.nan)
        return med_ess, ess_ratio


class Reservoir:
    """Algorithm R uniform reservoir, deterministic regardless of batch sizes."""
    def __init__(self, capacity: int, width: int, seed: int):
        self.values = np.empty((capacity, width), dtype=np.float64)
        self.capacity = capacity
        self.seen = 0
        self.rng = np.random.default_rng(seed)
        self.metadata: list[dict] = []

    def update(self, values: np.ndarray, metadata: list[dict] | None = None) -> None:
        for i, row in enumerate(values):
            index = self.seen if self.seen < self.capacity else int(self.rng.integers(self.seen + 1))
            if index < self.capacity:
                self.values[index] = row
                if metadata is not None:
                    if len(self.metadata) < self.capacity:
                        self.metadata.append(metadata[i])
                    else:
                        self.metadata[index] = metadata[i]
            self.seen += 1

    @property
    def sample(self):
        return self.values[:min(self.seen, self.capacity)]

    @property
    def sample_metadata(self) -> list[dict]:
        return self.metadata[:min(self.seen, self.capacity)]



def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_store(source: Path):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not (manifest.get("complete") is True or manifest.get("status") == "completed"):
        raise ValueError("Streaming extraction must be completed before analysis")
    columns = manifest.get("feature_columns", [])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Store requires unique feature_columns")
    chunks = []
    for item in manifest.get("chunks", []):
        name = item["file"] if isinstance(item, dict) else item
        path = (source / name).resolve()
        if source.resolve() not in path.parents or not path.is_file():
            raise ValueError("Invalid or missing chunk path")
        digest = _sha256(path)
        if isinstance(item, dict) and item.get("sha256") and item["sha256"] != digest:
            raise ValueError(f"Chunk checksum mismatch: {name}")
        chunks.append((path, digest))
    if not chunks:
        raise ValueError("Store has no committed chunks")
    return manifest, columns, chunks


def analyze_stream(source: str | Path, output_dir: str | Path,
                   config: StreamAnalysisConfig = StreamAnalysisConfig()) -> dict:
    source, destination = Path(source), Path(output_dir)
    if destination.exists():
        raise FileExistsError("Analysis output exists; choose a fresh output directory (completed or partial outputs are never overwritten)")
    manifest, columns, chunks = _load_store(source)
    width = len(columns)
    # Includes conservative array copies/SVD workspace allowance. This is an
    # allocation guard, not an OS RSS cap; CSV parsing and libraries add overhead.
    estimate = 8 * width * (config.batch_rows * 4 + config.sample_rows * 8)
    if estimate > config.memory_budget_mb * 1024 ** 2:
        raise ValueError("Estimated analysis array workspace exceeds memory budget; reduce sample_rows, batch_rows, or feature selection")
    moments, reservoir = RunningMoments(width), Reservoir(config.sample_rows, width, config.seed)

    default_identity = manifest.get("identity", {}) if isinstance(manifest.get("identity"), dict) else {}
    if not default_identity and (source.parent / "identity.json").is_file():
        try:
            default_identity = json.loads((source.parent / "identity.json").read_text())
        except Exception:
            pass

    def _extract_metadata(table: pd.DataFrame, n_rows: int) -> list[dict]:
        has_traj = "trajectory_id" in table
        has_rep = "replica_id" in table
        has_ens = "ensemble_id" in table
        t_id = default_identity.get("trajectory_id", "trajectory_0")
        r_id = default_identity.get("replica_id", "replica_0")
        e_id = default_identity.get("ensemble_id", "ensemble_0")
        if has_traj or has_rep or has_ens:
            t_col = table["trajectory_id"].tolist() if has_traj else [t_id] * n_rows
            r_col = table["replica_id"].tolist() if has_rep else [r_id] * n_rows
            e_col = table["ensemble_id"].tolist() if has_ens else [e_id] * n_rows
            return [{"trajectory_id": t, "replica_id": r, "ensemble_id": e} for t, r, e in zip(t_col, r_col, e_col)]
        return [{"trajectory_id": t_id, "replica_id": r_id, "ensemble_id": e_id}] * n_rows

    def batches():
        for path, expected_hash in chunks:
            if _sha256(path) != expected_hash:
                raise ValueError(f"Chunk changed during analysis: {path.name}")
            for table in pd.read_csv(path, chunksize=config.batch_rows):
                if not set(columns).issubset(table):
                    raise ValueError("Chunk feature schema mismatch")
                values = table[columns].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite features found; streaming analysis does not impute unavailable values")
                yield table, values

    traj_moments: dict[str, RunningMoments] = {}
    for table, values in batches():
        batch_meta = _extract_metadata(table, len(values))
        if config.compute_sam_mfp:
            traj_ids = [m["trajectory_id"] for m in batch_meta]
            for t_id in dict.fromkeys(traj_ids):
                if t_id not in traj_moments:
                    traj_moments[t_id] = RunningMoments(width)
                t_mask = [t == t_id for t in traj_ids]
                traj_moments[t_id].update(values[t_mask])
        moments.update(values)
        reservoir.update(values, batch_meta)
    if "total_rows" in manifest and manifest["total_rows"] != moments.count:
        raise ValueError("Manifest total_rows does not match committed chunks")
    if moments.count < max(2, config.states):
        raise ValueError("Too few frames for requested states")
    sample = reservoir.sample
    active = np.ptp(sample, axis=0) > 1e-12
    if not active.any():
        raise ValueError("Training sample has no variable features")
    scaler = StandardScaler().fit(sample[:, active])
    standardized = scaler.transform(sample[:, active])
    components = min(config.components, standardized.shape[0] - 1, standardized.shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(standardized)
    training_coordinates = pca.transform(standardized)
    if len(np.unique(training_coordinates, axis=0)) < config.states:
        raise ValueError("Training sample has fewer distinct coordinates than requested states")
    model = KMeans(n_clusters=config.states, n_init=10, random_state=config.seed).fit(training_coordinates)
    destination.mkdir(parents=True)
    (destination / "assignments").mkdir()
    # All model parameters are data arrays; no executable pickle is required.
    np.savez(destination / "sample_model.npz", active_mask=active, scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, pca_mean=pca.mean_, pca_components=pca.components_,
             pca_explained_variance_ratio=pca.explained_variance_ratio_,
             cluster_centers=model.cluster_centers_)
    counts = np.zeros(config.states, dtype=np.int64)
    state_sums = np.zeros((config.states, width), dtype=np.float64)
    autocov_tracker = StreamingAutocovariance(width, config.max_autocorrelation_lag, traj_moments) if config.compute_sam_mfp else None
    assignment_files = []
    for batch_index, (table, values) in enumerate(batches()):
        coordinates = pca.transform(scaler.transform(values[:, active]))
        labels = model.predict(coordinates)
        counts += np.bincount(labels, minlength=config.states)
        if config.compute_sam_mfp:
            batch_meta = _extract_metadata(table, len(values))
            traj_ids = np.array([m["trajectory_id"] for m in batch_meta])
            autocov_tracker.update(values, traj_ids)
            for k in range(config.states):
                k_mask = (labels == k)
                if np.any(k_mask):
                    state_sums[k] += values[k_mask].sum(axis=0)
        identities = table.drop(columns=columns).copy()
        identities["State"] = labels
        for component in range(components):
            identities[f"PC{component + 1}"] = coordinates[:, component]
        name = f"assignments/chunk_{batch_index:08d}.csv"
        identities.to_csv(destination / name, index=False)
        assignment_files.append(name)
    summary_table = pd.DataFrame({"Feature": columns, "Count": moments.count, "Mean": moments.mean,
                                  "Std": np.sqrt(moments.m2 / max(1, moments.count - 1)),
                                  "Minimum": moments.minimum, "Maximum": moments.maximum,
                                  "Variable_In_Training_Sample": active})
    summary_table.to_csv(destination / "feature_summary.csv", index=False)
    pd.DataFrame({"State": range(config.states), "Frame_Count": counts,
                  "Population": counts / counts.sum()}).to_csv(destination / "state_populations.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(range(config.states), counts / counts.sum())
    axis.set(xlabel="Sample-trained candidate state", ylabel="Fraction of all analyzed frames",
             title="Descriptive state populations")
    figure.tight_layout(); figure.savefig(destination / "state_populations.png", dpi=160); plt.close(figure)

    sam_mfp_report = {}
    sam_candidates = None
    if config.compute_sam_mfp:
        ess, ess_ratio = autocov_tracker.compute_ess()
        entropy = np.array([_occupancy_entropy(sample[:, j], bins=config.entropy_bins, tolerance=1e-12) for j in range(width)])

        valid_fraction = np.ones(width, dtype=float)
        resolution = np.where(active, "frame", "trajectory")

        frame_indices = np.where(active)[0]
        uniqueness = np.full(width, np.nan)
        if len(frame_indices) > 0:
            frame_sample_df = pd.DataFrame(sample[:, frame_indices], columns=[columns[i] for i in frame_indices])
            max_corr = _maximum_spearman_correlation(frame_sample_df, max_rows=config.sample_rows)
            for i, col_name in enumerate([columns[k] for k in frame_indices]):
                uniqueness[frame_indices[i]] = 1.0 - max_corr[col_name]

        raw_meta = reservoir.sample_metadata
        meta_df = pd.DataFrame(raw_meta) if raw_meta else pd.DataFrame({
            "trajectory_id": ["trajectory_0"] * len(sample),
            "replica_id": ["replica_0"] * len(sample),
            "ensemble_id": ["ensemble_0"] * len(sample),
        })
        has_multi_trajs = len(pd.unique(meta_df["trajectory_id"])) >= 2

        if has_multi_trajs:
            replica_consistency = np.array([_replica_consistency(sample[:, j], meta_df, tolerance=1e-12) for j in range(width)])
        else:
            replica_consistency = np.full(width, np.nan)

        pca_salience = np.full(width, np.nan)
        pca_salience[active] = np.abs(pca.components_).T @ pca.explained_variance_ratio_
        pca_ranks = np.full(width, np.nan)
        pca_ranks[active] = rankdata(-pca_salience[active], method="average")

        loto_std = np.full(width, np.nan)
        loto_norm = np.full(width, np.nan)
        loto_folds = np.zeros(width, dtype=int)
        loto_med = np.full(width, np.nan)

        if has_multi_trajs:
            unique_trajs = pd.unique(meta_df["trajectory_id"])
            fold_ranks = []
            for t_id in unique_trajs:
                train_mask = (meta_df["trajectory_id"] != t_id).to_numpy()
                if train_mask.sum() >= 2:
                    train_sample = sample[train_mask][:, active]
                    train_scaled = StandardScaler().fit_transform(train_sample)
                    k_comp = min(config.components, train_scaled.shape[0] - 1, train_scaled.shape[1])
                    if k_comp >= 1:
                        f_pca = PCA(n_components=k_comp, svd_solver="full").fit(train_scaled)
                        f_sal = np.abs(f_pca.components_).T @ f_pca.explained_variance_ratio_
                        fold_ranks.append(rankdata(-f_sal, method="average"))
            if len(fold_ranks) >= 2:
                r_arr = np.asarray(fold_ranks, dtype=float)
                loto_std_active = np.std(r_arr, axis=0, ddof=1)
                loto_std[active] = loto_std_active
                loto_norm[active] = loto_std_active / max(1, int(active.sum()) - 1)
                loto_folds[active] = len(fold_ranks)
                loto_med[active] = np.median(r_arr, axis=0)
            else:
                loto_folds[active] = len(fold_ranks) if fold_ranks else 1
                loto_norm[active] = 0.0
                loto_med[active] = pca_ranks[active]
        else:
            loto_folds[active] = 1
            loto_norm[active] = 0.0
            loto_med[active] = pca_ranks[active]

        between = np.zeros(width, dtype=float)
        for k in range(config.states):
            mean_k = state_sums[k] / max(1, counts[k])
            between += counts[k] * (mean_k - moments.mean) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            eta_squared = np.where(active, np.where(moments.m2 > 1e-12, between / np.maximum(1e-12, moments.m2), 0.0), np.nan)

        quality_df = pd.DataFrame({
            "Feature": columns,
            "Valid_Fraction": valid_fraction,
            "Resolution": resolution,
            "Median_ESS": ess,
            "ESS_Ratio": ess_ratio,
            "Replica_Consistency": replica_consistency,
            "Occupancy_Entropy": entropy,
            "Uniqueness": uniqueness,
        })
        salience_df = pd.DataFrame({
            "Feature": columns,
            "PCA_Salience": pca_salience,
            "PCA_Rank": pca_ranks,
            "LOTO_Rank_Median": loto_med,
            "LOTO_Rank_Std": loto_std,
            "LOTO_Folds": loto_folds,
            "LOTO_Rank_Std_Normalized": loto_norm,
        })
        state_exp_df = pd.DataFrame({
            "Feature": columns,
            "State_Explained_Variance_Eta_Squared": eta_squared,
        })

        sam_config = SAMMFPConfig(min_loto_folds=config.min_loto_folds, include_state_separation=True)
        sam_candidates, sam_notes = prioritize_sam_mfp(quality_df, salience_df, sam_config, state_explanatory=state_exp_df)

        sam_candidates.to_csv(destination / "sam_mfp_candidates.csv", index=False)
        quality_df.to_csv(destination / "feature_quality.csv", index=False)
        salience_df.to_csv(destination / "representation_salience.csv", index=False)
        state_exp_df.to_csv(destination / "state_explanatory_ranking.csv", index=False)

        sam_mfp_report = {
            "sam_mfp_candidate_count": int(sam_candidates["SAM_MFP_Pareto_Front"].sum()),
            "sam_mfp_candidates": sam_candidates.loc[sam_candidates["SAM_MFP_Pareto_Front"], "Feature"].tolist(),
            "sam_mfp_objectives": sam_candidates["SAM_MFP_Objectives"].iloc[0] if not sam_candidates.empty else "",
            "sam_mfp_notes": sam_notes,
        }

    if config.compute_sam_mfp:
        limitations = [
            "PCA, clustering, uniqueness and occupancy entropy are fitted on a seeded uniform reservoir sample.",
            "ESS ratio and candidate state separation (ANOVA eta-squared) cover all analyzed frames.",
            "Target-free SAM-MFP Pareto candidates were evaluated in bounded memory.",
            "Array budget is an estimate, not a hard process memory cap.",
            "Million-frame capacity has not been validated by this run unless its recorded frame count reaches that size.",
        ]
        if not has_multi_trajs:
            limitations.append("Single trajectory provided: evaluated Pareto candidates without cross-replica LOTO thresholding.")
    else:
        limitations = [
            "PCA and clustering are fitted to a seeded uniform reservoir sample.",
            "States are descriptive hypotheses; no replica stability or SAM-MFP was computed.",
            "Array budget is an estimate, not a hard process memory cap.",
            "Million-frame capacity has not been validated by this run unless its recorded frame count reaches that size.",
        ]

    report = {"status": "completed", "scope": "sample-trained descriptive state discovery" if not config.compute_sam_mfp else "sample-trained descriptive discovery with bounded-memory SAM-MFP candidate selection",
              "configuration": asdict(config), "frames_analyzed": moments.count,
              "training_sample_rows": len(sample), "feature_count": width,
              "active_feature_count": int(active.sum()), "estimated_array_workspace_bytes": estimate,
              "assignment_files": assignment_files,
              "source_chunks": [{"file": str(path), "sha256": digest} for path, digest in chunks],
              **sam_mfp_report,
              "limitations": limitations}
    state_rows = [{"state": int(i), "frames": int(count), "fraction": float(count / moments.count)}
                  for i, count in enumerate(counts)]
    state_rows.sort(key=lambda row: row["frames"], reverse=True)
    largest_state = state_rows[0] if state_rows else None
    variable = summary_table.loc[summary_table["Variable_In_Training_Sample"]].nlargest(8, "Std")
    feature_rows = "".join(
        f"<tr><td>{html.escape(str(row.Feature))}</td><td>{row.Mean:.4g}</td><td>{row.Std:.4g}</td><td>{row.Minimum:.4g}</td><td>{row.Maximum:.4g}</td></tr>"
        for row in variable.itertuples(index=False)
    ) or "<tr><td colspan='5'>No variable feature measurements in the training sample.</td></tr>"
    population_rows = "".join(
        f"<tr><td>State {row['state']}</td><td>{row['frames']:,}</td><td>{row['fraction']:.1%}</td></tr>"
        for row in sorted(state_rows, key=lambda row: row["state"])
    )
    imbalance = (f"State {largest_state['state']} is the most populated candidate ({largest_state['fraction']:.1%} of frames)."
                 if largest_state else "No candidate states were assigned.")
    report["largest_candidate_state"] = largest_state
    report["top_variable_features_by_sample_standard_deviation"] = variable["Feature"].astype(str).tolist()

    sam_html = ""
    if config.compute_sam_mfp and sam_candidates is not None:
        cand_rows = "".join(
            f"<tr><td>{html.escape(str(r.Feature))}</td><td>{'Yes (Pareto)' if r.SAM_MFP_Pareto_Front else 'Dominated'}</td>"
            f"<td>{int(r.SAM_MFP_Dominator_Count)}</td><td>{r.PCA_Salience:.4g}</td><td>{r.ESS_Ratio:.4g}</td>"
            f"<td>{r.Occupancy_Entropy:.4g}</td><td>{r.Uniqueness:.4g}</td></tr>"
            for r in sam_candidates.head(15).itertuples(index=False)
        )
        sam_html = f"""
      <section id='sam_mfp'><h2>SAM-MFP Multi-Criteria Prioritization</h2>
      <p>Target-free Pareto candidate features balancing representation salience, sampling efficiency (ESS), occupancy entropy, and uniqueness in bounded memory.</p>
      <table><thead><tr><th>Feature</th><th>Pareto Front</th><th>Dominators</th><th>PCA Salience</th><th>ESS Ratio</th><th>Entropy</th><th>Uniqueness</th></tr></thead>
      <tbody>{cand_rows}</tbody></table>
      <p><a href='sam_mfp_candidates.csv'>Download complete SAM-MFP Pareto candidates (CSV)</a> · <a href='feature_quality.csv'>Feature quality metrics (CSV)</a></p></section>
        """

    css = """
      :root{color-scheme:light;--ink:#17212b;--muted:#586777;--line:#d9e1e8;--blue:#245b78;--paper:#fff;--wash:#f3f7fa}
      *{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
      main{max-width:1120px;margin:36px auto;padding:0 22px 48px}header{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin-bottom:18px}
      h1{font-size:30px;line-height:1.2;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}h3{font-size:15px;margin:24px 0 8px;color:var(--blue)}p{margin:6px 0;color:var(--muted)}
      .eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;font-weight:700;color:var(--blue)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
      .card,section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px}.metric{font-size:26px;font-weight:700;line-height:1.2}.label{font-size:12px;color:var(--muted);margin-top:5px}
      section{margin-top:14px}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted);font-weight:650;background:#f8fafb}td:first-child{overflow-wrap:anywhere}
      img{display:block;max-width:100%;height:auto;margin:12px auto 2px}.note{border-left:3px solid #d19531;padding:10px 13px;background:#fff9eb;border-radius:3px;color:#564625}
      nav a{color:var(--blue);text-decoration:none;margin-right:16px}footer{font-size:12px;color:var(--muted);padding:20px 4px}
      @media(max-width:680px){main{margin:18px auto;padding:0 12px 30px}.cards{grid-template-columns:1fr}header,section{padding:18px}table{font-size:12px}th,td{padding:7px 5px}}
    """
    scope_note = "This run performs sample-trained descriptive discovery with bounded-memory SAM-MFP candidate selection." if config.compute_sam_mfp else "This run performs sample-trained descriptive state discovery. It does not calculate replica stability, SAM-MFP prioritization, supervised prediction, or kinetic rates."
    text = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
      <title>MolScope | Streaming analysis</title><style>{css}</style></head><body><main>
      <header><div class='eyebrow'>MolScope · descriptive analysis</div><h1>Streaming trajectory report</h1>
      <p>State discovery was fitted on a reproducible sample and applied to all analyzed frames.</p>
      <nav><a href='#results'>Results</a><a href='#features'>Variable features</a>{'<a href="#sam_mfp">SAM-MFP</a>' if config.compute_sam_mfp else ''}<a href='#methods'>Scope and methods</a><a href='analysis_summary.json'>Full analysis record</a></nav></header>
      <div class='cards'><div class='card'><div class='metric'>{moments.count:,}</div><div class='label'>frames analyzed</div></div>
      <div class='card'><div class='metric'>{width:,}</div><div class='label'>measured feature columns</div></div>
      <div class='card'><div class='metric'>{len(sample):,}</div><div class='label'>frames used to fit PCA and clustering</div></div></div>
      <section id='results'><h2>Results at a glance</h2><p>{config.states} candidate states were assigned across all analyzed frames. {imbalance}</p>
      <table><thead><tr><th>Candidate state</th><th>Frames</th><th>Population</th></tr></thead><tbody>{population_rows}</tbody></table>
      <h3>State populations</h3><img src='state_populations.png' alt='Fraction of analyzed frames assigned to each candidate state'>
      <p>State labels are descriptive clusters in the fitted PCA representation; they are not experimentally validated states.</p></section>
      <section id='features'><h2>Most variable measured features</h2><p>Sorted by standard deviation in the training sample. Units remain those of the extracted feature.</p>
      <table><thead><tr><th>Feature</th><th>Mean</th><th>Std. dev.</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>{feature_rows}</tbody></table>
      <p><a href='feature_summary.csv'>Download complete feature summary</a></p></section>
      {sam_html}
      <section id='methods'><h2>Scope and interpretation</h2><p class='note'>{scope_note}</p>
      <h3>Method and resource record</h3><table><tbody><tr><th>Training sample</th><td>{len(sample):,} uniformly reservoir-sampled frames; seed {config.seed}</td></tr>
      <tr><th>Representation</th><td>Standard scaling followed by {components}-component PCA on {int(active.sum()):,} variable features</td></tr>
      <tr><th>Clustering</th><td>K-means with {config.states} requested states; 10 initializations</td></tr>
      <tr><th>Estimated array workspace</th><td>{estimate / (1024 * 1024):.1f} MiB (estimate, not a process memory limit)</td></tr></tbody></table>
      <h3>Limitations</h3><ul>{''.join(f'<li>{html.escape(line)}</li>' for line in report['limitations'])}</ul>
      <p><a href='state_populations.csv'>State population data</a> · <a href='feature_summary.csv'>Feature summary data</a> · <a href='analysis_summary.json'>Chunk hashes and complete provenance</a></p></section>
      <footer>MolScope streaming analysis · {moments.count:,} frames processed</footer></main></body></html>"""
    (destination / "report.html").write_text(text, encoding="utf-8")
    # Commit the completed marker only after every artifact has been written.
    pending = destination / "analysis_summary.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(destination / "analysis_summary.json")
    return report
    (destination / "report.html").write_text(text, encoding="utf-8")
    # Commit the completed marker only after every artifact has been written.
    pending = destination / "analysis_summary.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(destination / "analysis_summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    for name, default in (("batch-rows", 512), ("sample-rows", 2000), ("components", 3), ("states", 2), ("seed", 42), ("memory-budget-mb", 512)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--sam-mfp", action="store_true", default=False, help="Compute bounded-memory SAM-MFP Pareto candidates")
    parser.add_argument("--min-loto-folds", type=int, default=2, help="Minimum LOTO folds for cross-replica stability")
    parser.add_argument("--max-autocorrelation-lag", type=int, default=100, help="Maximum lag for online ESS autocorrelation")
    parser.add_argument("--entropy-bins", type=int, default=8, help="Number of histogram bins for occupancy entropy")
    args = parser.parse_args()
    report = analyze_stream(args.source, args.out_dir, StreamAnalysisConfig(
        batch_rows=args.batch_rows, sample_rows=args.sample_rows, components=args.components,
        states=args.states, seed=args.seed, memory_budget_mb=args.memory_budget_mb,
        compute_sam_mfp=args.sam_mfp, min_loto_folds=args.min_loto_folds,
        max_autocorrelation_lag=args.max_autocorrelation_lag, entropy_bins=args.entropy_bins))
    print(json.dumps({"status": report["status"], "frames_analyzed": report["frames_analyzed"]}))


if __name__ == "__main__":
    main()
