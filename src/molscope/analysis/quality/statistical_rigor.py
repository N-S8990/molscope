"""Statistical ML rigor diagnostics for conformational state discovery and trajectory sampling.

Provides diagnostics for:
1. Seed stability (pairwise Adjusted Rand Index across random initializations).
2. Bootstrap resampling cluster stability (with empirical confidence intervals).
3. Intrinsic clustering validity indices (Silhouette, Calinski-Harabasz, Davies-Bouldin).
4. Simulation convergence and sampling adequacy (integrated autocorrelation time,
   effective sample size, and split-half stationarity/drift testing).
5. Feature ranking stability across seeds, folds, and replica subsets (Spearman rho, Kendall tau, top-k overlap).
6. Group-level trajectory bootstrap uncertainty quantification.
7. Permutation-test and shuffled-label negative control diagnostics.
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def _extract_coordinate_matrix(coordinates: pd.DataFrame | np.ndarray) -> np.ndarray:
    """Extract numeric coordinate matrix from DataFrame or numpy array."""
    if isinstance(coordinates, np.ndarray):
        if coordinates.ndim == 1:
            mat = coordinates[:, np.newaxis]
        elif coordinates.ndim == 2:
            mat = coordinates
        else:
            raise ValueError(f"Coordinates must be 1D or 2D, got shape {coordinates.shape}")
        mat = np.asarray(mat, dtype=float)
        if not np.isfinite(mat).all():
            raise ValueError("Coordinates contain non-finite values.")
        return mat

    if isinstance(coordinates, pd.DataFrame):
        ckpca_cols = [c for c in coordinates if c.startswith("CKPCA") and (c[5:].isdigit() or (c.startswith("CKPCA_") and c[6:].isdigit()))]
        tic_cols = [c for c in coordinates if c.startswith("TIC") and (c[3:].isdigit() or (c.startswith("TIC_") and c[4:].isdigit()))]
        pc_cols = [c for c in coordinates if c.startswith("PC") and c[2:].isdigit()]

        if ckpca_cols:
            cols = sorted(ckpca_cols, key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x[5:]))
        elif tic_cols:
            cols = sorted(tic_cols, key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x[3:]))
        elif pc_cols:
            cols = sorted(pc_cols, key=lambda x: int(x[2:]))
        else:
            meta_cols = {
                "trajectory_id", "replica_id", "ensemble_id",
                "frame_index", "frame_order", "time_ps", "time",
                "State", "State_Label", "condition",
            }
            candidate_cols = [c for c in coordinates.columns if c not in meta_cols]
            numeric = coordinates[candidate_cols].apply(pd.to_numeric, errors="coerce")
            valid = numeric.columns[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
            if not valid:
                raise ValueError("No valid numeric coordinate columns found in DataFrame.")
            cols = valid

        mat = coordinates[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(mat).all():
            raise ValueError("Coordinates contain non-finite values.")
        return mat

    raise TypeError(f"Coordinates must be a pandas DataFrame or numpy ndarray, got {type(coordinates)}")


def evaluate_seed_stability(
    coordinates: pd.DataFrame | np.ndarray,
    config: Any = None,
    seeds: Sequence[int] = (42, 100, 2024),
    *,
    method: str = "kmeans",
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate clustering seed stability via pairwise Adjusted Rand Index (ARI).

    Parameters
    ----------
    coordinates : pd.DataFrame | np.ndarray
        Conformational coordinates (e.g. PCA, TIC, CKPCA).
    config : Any, optional
        StateDiscoveryConfig, dict, or object specifying n_states/n_clusters.
    seeds : Sequence[int]
        Random seeds to evaluate across. Default is (42, 100, 2024).
    method : str
        Clustering method (default 'kmeans').
    **kwargs : Any
        Additional options (e.g. n_clusters, n_restarts).

    Returns
    -------
    dict[str, Any]
        Dictionary containing pairwise ARI map, pairwise comparisons DataFrame,
        mean ARI, median ARI, min ARI, max ARI, and labels by seed.
    """
    X = _extract_coordinate_matrix(coordinates)
    if len(seeds) < 2:
        raise ValueError("At least two seeds are required for pairwise seed stability evaluation.")

    n_clusters = 2
    if config is not None:
        if hasattr(config, "n_states"):
            n_clusters = config.n_states
        elif hasattr(config, "n_clusters"):
            n_clusters = config.n_clusters
        elif isinstance(config, dict):
            n_clusters = config.get("n_states", config.get("n_clusters", 2))
        elif isinstance(config, int):
            n_clusters = config
    if "n_clusters" in kwargs:
        n_clusters = kwargs["n_clusters"]
    elif "n_states" in kwargs:
        n_clusters = kwargs["n_states"]

    if len(X) <= n_clusters:
        raise ValueError(f"Sample count ({len(X)}) must exceed cluster count ({n_clusters}).")

    labels_by_seed: dict[int, np.ndarray] = {}
    for seed in seeds:
        if method.lower() == "kmeans":
            model = KMeans(n_clusters=n_clusters, random_state=seed, n_init=1)
            labels = model.fit_predict(X)
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        labels_by_seed[seed] = labels

    pairs = list(itertools.combinations(seeds, 2))
    pairwise_ari: dict[tuple[int, int], float] = {}
    comparison_rows = []
    aris: list[float] = []

    for s1, s2 in pairs:
        ari = float(adjusted_rand_score(labels_by_seed[s1], labels_by_seed[s2]))
        pairwise_ari[(s1, s2)] = ari
        aris.append(ari)
        comparison_rows.append({
            "Seed_1": s1,
            "Seed_2": s2,
            "Adjusted_Rand_Index": ari,
        })

    pairwise_df = pd.DataFrame(comparison_rows)
    mean_ari = float(np.mean(aris))
    median_ari = float(np.median(aris))
    min_ari = float(np.min(aris))
    max_ari = float(np.max(aris))
    std_ari = float(np.std(aris, ddof=1)) if len(aris) > 1 else 0.0

    return {
        "pairwise_ari": pairwise_ari,
        "pairwise_comparisons": pairwise_df,
        "mean_ari": mean_ari,
        "median_ari": median_ari,
        "min_ari": min_ari,
        "max_ari": max_ari,
        "std_ari": std_ari,
        "labels_by_seed": labels_by_seed,
        "seeds": list(seeds),
        "n_clusters": n_clusters,
    }


def evaluate_bootstrap_cluster_stability(
    coordinates: pd.DataFrame | np.ndarray,
    n_bootstraps: int = 20,
    config: Any = None,
    *,
    random_state: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Evaluate cluster stability under bootstrap resampling.

    Fits a reference clustering on the full dataset, then for each bootstrap resample,
    fits a model and predicts cluster assignments across all frames, computing the
    Adjusted Rand Index against the reference clustering.

    Parameters
    ----------
    coordinates : pd.DataFrame | np.ndarray
        Conformational coordinates.
    n_bootstraps : int
        Number of bootstrap iterations (default 20).
    config : Any, optional
        StateDiscoveryConfig, dict, or object specifying n_states/n_clusters.
    random_state : int
        Base random state for bootstrap resampling.
    **kwargs : Any
        Additional keyword arguments.

    Returns
    -------
    dict[str, Any]
        Dictionary with bootstrap ARIs, mean ARI, median ARI, standard error / std,
        95% empirical bootstrap confidence interval, and reference labels.
    """
    X = _extract_coordinate_matrix(coordinates)
    if n_bootstraps < 2:
        raise ValueError("n_bootstraps must be at least 2.")

    n_clusters = 2
    if config is not None:
        if hasattr(config, "n_states"):
            n_clusters = config.n_states
        elif hasattr(config, "n_clusters"):
            n_clusters = config.n_clusters
        elif isinstance(config, dict):
            n_clusters = config.get("n_states", config.get("n_clusters", 2))
        elif isinstance(config, int):
            n_clusters = config
    if "n_clusters" in kwargs:
        n_clusters = kwargs["n_clusters"]
    elif "n_states" in kwargs:
        n_clusters = kwargs["n_states"]

    n_samples = len(X)
    if n_samples <= n_clusters:
        raise ValueError(f"Sample count ({n_samples}) must exceed cluster count ({n_clusters}).")

    ref_model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10).fit(X)
    ref_labels = ref_model.labels_

    bootstrap_aris: list[float] = []
    boot_rows = []

    for b in range(n_bootstraps):
        rng = np.random.default_rng(random_state + b + 1)
        boot_indices = rng.choice(n_samples, size=n_samples, replace=True)
        X_boot = X[boot_indices]
        boot_model = KMeans(n_clusters=n_clusters, random_state=random_state + b + 1, n_init=5).fit(X_boot)
        pred_labels = boot_model.predict(X)
        ari = float(adjusted_rand_score(ref_labels, pred_labels))
        bootstrap_aris.append(ari)
        boot_rows.append({"Bootstrap": b + 1, "Adjusted_Rand_Index": ari})

    mean_ari = float(np.mean(bootstrap_aris))
    median_ari = float(np.median(bootstrap_aris))
    std_ari = float(np.std(bootstrap_aris, ddof=1)) if len(bootstrap_aris) > 1 else 0.0
    ci_lower = float(np.quantile(bootstrap_aris, 0.025))
    ci_upper = float(np.quantile(bootstrap_aris, 0.975))

    return {
        "bootstrap_aris": bootstrap_aris,
        "bootstrap_results": pd.DataFrame(boot_rows),
        "mean_ari": mean_ari,
        "median_ari": median_ari,
        "std_ari": std_ari,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "n_bootstraps": n_bootstraps,
        "n_clusters": n_clusters,
        "reference_labels": ref_labels,
    }


def compute_clustering_validity_indices(
    coordinates: pd.DataFrame | np.ndarray,
    labels: Sequence[int] | np.ndarray | pd.Series,
) -> dict[str, float]:
    """Compute intrinsic clustering validity indices: Silhouette, Calinski-Harabasz, and Davies-Bouldin.

    Parameters
    ----------
    coordinates : pd.DataFrame | np.ndarray
        Conformational coordinate matrix.
    labels : Sequence[int] | np.ndarray | pd.Series
        Cluster assignment labels for each frame.

    Returns
    -------
    dict[str, float]
        Dictionary containing silhouette score, calinski_harabasz score, and davies_bouldin index.
    """
    X = _extract_coordinate_matrix(coordinates)
    labels_arr = np.asarray(labels)
    if len(X) != len(labels_arr):
        raise ValueError(f"Length of coordinates ({len(X)}) must match labels ({len(labels_arr)}).")

    unique_labels = np.unique(labels_arr)
    n_clusters = len(unique_labels)
    if n_clusters < 2 or n_clusters >= len(X):
        raise ValueError(
            f"Clustering validity indices require 2 <= n_clusters <= n_samples - 1. Got {n_clusters} clusters for {len(X)} samples."
        )

    silhouette = float(silhouette_score(X, labels_arr))
    calinski = float(calinski_harabasz_score(X, labels_arr))
    davies = float(davies_bouldin_score(X, labels_arr))

    return {
        "silhouette": silhouette,
        "silhouette_score": silhouette,
        "calinski_harabasz": calinski,
        "calinski_harabasz_score": calinski,
        "davies_bouldin": davies,
        "davies_bouldin_score": davies,
        "davies_bouldin_index": davies,
        "n_clusters": n_clusters,
        "n_samples": len(X),
    }


def _integrated_autocorrelation_time(series: np.ndarray, max_lag: int | None = None) -> float:
    """Compute integrated autocorrelation time using the initial positive sequence."""
    finite = series[np.isfinite(series)]
    n = len(finite)
    if n < 4:
        return 1.0
    centered = finite - np.mean(finite)
    denom = float(np.dot(centered, centered))
    if denom <= 1e-15:
        return 1.0
    if max_lag is None:
        max_lag = min(n // 2, 250)
    else:
        max_lag = min(max_lag, n // 2)

    autocorrs: list[float] = []
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if rho <= 0:
            break
        autocorrs.append(rho)

    tau_int = 1.0 + 2.0 * sum(autocorrs)
    return float(max(1.0, min(tau_int, float(n))))


def check_simulation_convergence(
    coordinates: pd.DataFrame | np.ndarray,
    time_column: str = "time_ps",
    *,
    min_effective_samples: float = 20.0,
    max_tau_fraction: float = 0.1,
    drift_z_threshold: float = 3.0,
) -> dict[str, Any]:
    """Diagnose sampling adequacy and flag unconverged simulations.

    Evaluates integrated autocorrelation time, effective sample size (ESS),
    and split-half stationarity (drift testing) to detect unconverged or
    insufficiently sampled molecular simulations.

    Parameters
    ----------
    coordinates : pd.DataFrame | np.ndarray
        Simulation coordinates or trajectory feature table.
    time_column : str
        Name of timestamp column if present in DataFrame (default 'time_ps').
    min_effective_samples : float
        Minimum ESS required to consider sampling adequate (default 20.0).
    max_tau_fraction : float
        Maximum allowable ratio of autocorrelation time to trajectory length (default 0.1).
    drift_z_threshold : float
        Z-score threshold for difference between first and second half of trajectory (default 3.0).

    Returns
    -------
    dict[str, Any]
        Dictionary with convergence status, sampling adequacy, autocorrelation time,
        effective sample size, detected flags, and per-series diagnostic details.
    """
    records = []
    flags: list[str] = []
    warnings_list: list[str] = []

    if isinstance(coordinates, pd.DataFrame):
        df = coordinates.copy()
        order_col = time_column if time_column in df.columns else ("frame_order" if "frame_order" in df.columns else ("frame_index" if "frame_index" in df.columns else None))
        has_traj = "trajectory_id" in df.columns
        groups = df.groupby("trajectory_id", sort=False) if has_traj else [(None, df)]

        # Find coordinate columns
        ckpca_cols = [c for c in df if c.startswith("CKPCA") and (c[5:].isdigit() or (c.startswith("CKPCA_") and c[6:].isdigit()))]
        tic_cols = [c for c in df if c.startswith("TIC") and (c[3:].isdigit() or (c.startswith("TIC_") and c[4:].isdigit()))]
        pc_cols = [c for c in df if c.startswith("PC") and c[2:].isdigit()]
        if ckpca_cols:
            eval_cols = sorted(ckpca_cols, key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x[5:]))[:3]
        elif tic_cols:
            eval_cols = sorted(tic_cols, key=lambda x: int(x.split("_")[-1]) if "_" in x else int(x[3:]))[:3]
        elif pc_cols:
            eval_cols = sorted(pc_cols, key=lambda x: int(x[2:]))[:3]
        else:
            meta = {
                "trajectory_id", "replica_id", "ensemble_id",
                "frame_index", "frame_order", "time_ps", "time",
                "State", "State_Label", "condition",
            }
            cand = [c for c in df.columns if c not in meta]
            numeric = df[cand].apply(pd.to_numeric, errors="coerce")
            valid = numeric.columns[np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
            eval_cols = valid[:5] if valid else []

        if not eval_cols:
            raise ValueError("No numeric coordinate columns found to evaluate convergence.")

        for traj_id, group in groups:
            if order_col is not None and order_col in group.columns:
                group = group.sort_values(order_col)

            for col in eval_cols:
                series = group[col].to_numpy(dtype=float)
                n_frames = len(series)
                tau = _integrated_autocorrelation_time(series)
                ess = float(n_frames / tau)

                # Split-half drift test
                n_half = n_frames // 2
                h1, h2 = series[:n_half], series[n_half:]
                m1, m2 = float(np.mean(h1)), float(np.mean(h2))
                v1, v2 = float(np.var(h1, ddof=1)), float(np.var(h2, ddof=1))
                tau1 = _integrated_autocorrelation_time(h1)
                tau2 = _integrated_autocorrelation_time(h2)
                ess1 = max(1.0, len(h1) / tau1)
                ess2 = max(1.0, len(h2) / tau2)
                se_diff = np.sqrt(v1 / ess1 + v2 / ess2) if (v1 + v2) > 1e-14 else 1e-7
                z_drift = float(abs(m1 - m2) / se_diff)

                drift_detected = bool(z_drift > drift_z_threshold and abs(m1 - m2) > 0.05 * (np.std(series) + 1e-12))
                adequate_sampling = bool(ess >= min_effective_samples and (tau / n_frames) <= max_tau_fraction)
                col_converged = bool(adequate_sampling and not drift_detected)

                record = {
                    "trajectory_id": traj_id if traj_id is not None else "default",
                    "coordinate": col,
                    "n_frames": n_frames,
                    "autocorrelation_time": tau,
                    "effective_sample_size": ess,
                    "tau_fraction": float(tau / n_frames),
                    "drift_z_score": z_drift,
                    "drift_detected": drift_detected,
                    "sampling_adequate": adequate_sampling,
                    "is_converged": col_converged,
                }
                records.append(record)

    else:
        # Numpy array input
        mat = _extract_coordinate_matrix(coordinates)
        n_frames = len(mat)
        n_features = mat.shape[1]
        for col_idx in range(min(n_features, 5)):
            series = mat[:, col_idx]
            tau = _integrated_autocorrelation_time(series)
            ess = float(n_frames / tau)

            n_half = n_frames // 2
            h1, h2 = series[:n_half], series[n_half:]
            m1, m2 = float(np.mean(h1)), float(np.mean(h2))
            v1, v2 = float(np.var(h1, ddof=1)), float(np.var(h2, ddof=1))
            tau1 = _integrated_autocorrelation_time(h1)
            tau2 = _integrated_autocorrelation_time(h2)
            ess1 = max(1.0, len(h1) / tau1)
            ess2 = max(1.0, len(h2) / tau2)
            se_diff = np.sqrt(v1 / ess1 + v2 / ess2) if (v1 + v2) > 1e-14 else 1e-7
            z_drift = float(abs(m1 - m2) / se_diff)

            drift_detected = bool(z_drift > drift_z_threshold and abs(m1 - m2) > 0.05 * (np.std(series) + 1e-12))
            adequate_sampling = bool(ess >= min_effective_samples and (tau / n_frames) <= max_tau_fraction)
            col_converged = bool(adequate_sampling and not drift_detected)

            records.append({
                "trajectory_id": "array",
                "coordinate": f"dim_{col_idx}",
                "n_frames": n_frames,
                "autocorrelation_time": tau,
                "effective_sample_size": ess,
                "tau_fraction": float(tau / n_frames),
                "drift_z_score": z_drift,
                "drift_detected": drift_detected,
                "sampling_adequate": adequate_sampling,
                "is_converged": col_converged,
            })

    details_df = pd.DataFrame(records)
    all_converged = bool(details_df["is_converged"].all())
    all_sampling_adequate = bool(details_df["sampling_adequate"].all())
    any_drift = bool(details_df["drift_detected"].any())

    if not all_sampling_adequate:
        flags.append("insufficient_effective_samples")
        warnings_list.append(
            "Simulation has high autocorrelation time or insufficient effective samples (ESS < threshold)."
        )
    if any_drift:
        flags.append("significant_drift_between_halves")
        warnings_list.append(
            "Significant conformational drift detected between trajectory halves; equilibrium may not be reached."
        )

    mean_tau = float(details_df["autocorrelation_time"].mean())
    mean_ess = float(details_df["effective_sample_size"].mean())

    return {
        "converged": all_converged,
        "is_converged": all_converged,
        "sampling_adequate": all_sampling_adequate,
        "autocorrelation_time": mean_tau,
        "effective_sample_size": mean_ess,
        "flags": flags,
        "warnings": warnings_list,
        "details": details_df,
    }


def evaluate_feature_ranking_stability(
    rankings: Sequence[pd.DataFrame | Sequence[str]],
    *,
    feature_column: str = "Feature_Group",
    rank_column: str | None = None,
    score_column: str | None = None,
    top_k: int | None = None,
    stability_threshold: float = 0.50,
) -> dict[str, Any]:
    """Evaluate stability of feature importance rankings across random seeds or replica splits.

    Computes pairwise rank correlations (Spearman rho, Kendall tau), top-k Jaccard
    overlap, and per-feature rank variance across multiple ranking runs.

    Parameters
    ----------
    rankings : Sequence[pd.DataFrame | Sequence[str]]
        Multiple feature rankings (e.g. from LOFO, Random Forest, or SAM-MFP).
    feature_column : str
        Column name identifying feature names (default 'Feature_Group', fallback 'Feature').
    rank_column : str | None
        Column name containing rank (e.g. 'Rank' or 'Mean_Rank'). Inferred if None.
    score_column : str | None
        Column name containing continuous importance score to rank by if rank_column is None.
    top_k : int | None
        Optional integer to compute top-k Jaccard overlap stability.

    Returns
    -------
    dict[str, Any]
        Dictionary with pairwise correlations, mean/median Spearman rho and Kendall tau,
        top-k overlap, per-feature rank summary (mean, std, min, max rank), and stability flag.
    """
    if len(rankings) < 2:
        raise ValueError("At least two rankings are required for ranking stability evaluation.")

    parsed_rankings: list[dict[str, float]] = []
    ordered_lists: list[list[str]] = []

    for r in rankings:
        if isinstance(r, pd.DataFrame):
            df = r.copy()
            feat_col = feature_column if feature_column in df.columns else ("Feature" if "Feature" in df.columns else None)
            if feat_col is None:
                raise ValueError(f"Could not find feature column '{feature_column}' or 'Feature' in ranking DataFrame.")

            if rank_column and rank_column in df.columns:
                df = df.sort_values(rank_column)
                ranks = {row[feat_col]: float(row[rank_column]) for _, row in df.iterrows()}
                ordered = df[feat_col].tolist()
            elif score_column and score_column in df.columns:
                df = df.sort_values(score_column, ascending=False)
                ranks = {row[feat_col]: float(i + 1) for i, (_, row) in enumerate(df.iterrows())}
                ordered = df[feat_col].tolist()
            elif "Rank" in df.columns:
                df = df.sort_values("Rank")
                ranks = {row[feat_col]: float(row["Rank"]) for _, row in df.iterrows()}
                ordered = df[feat_col].tolist()
            elif "Mean_Rank" in df.columns:
                df = df.sort_values("Mean_Rank")
                ranks = {row[feat_col]: float(row["Mean_Rank"]) for _, row in df.iterrows()}
                ordered = df[feat_col].tolist()
            elif "Trajectory_Mean_Accuracy_Drop_Mean" in df.columns:
                df = df.sort_values("Trajectory_Mean_Accuracy_Drop_Mean", ascending=False)
                ranks = {row[feat_col]: float(i + 1) for i, (_, row) in enumerate(df.iterrows())}
                ordered = df[feat_col].tolist()
            else:
                ranks = {row[feat_col]: float(i + 1) for i, (_, row) in enumerate(df.iterrows())}
                ordered = df[feat_col].tolist()
        elif isinstance(r, (list, tuple)):
            ordered = list(r)
            ranks = {feat: float(i + 1) for i, feat in enumerate(ordered)}
        else:
            raise TypeError(f"Each ranking must be a DataFrame or sequence of feature names, got {type(r)}")

        parsed_rankings.append(ranks)
        ordered_lists.append(ordered)

    all_features = sorted(set().union(*[p.keys() for p in parsed_rankings]))
    n_features = len(all_features)
    worst_rank = float(n_features) + 1.0

    rank_matrix = np.zeros((len(rankings), n_features), dtype=float)
    for i, p in enumerate(parsed_rankings):
        for j, feat in enumerate(all_features):
            rank_matrix[i, j] = p.get(feat, worst_rank)

    pairs = list(itertools.combinations(range(len(rankings)), 2))
    spearman_coeffs = []
    kendall_coeffs = []
    top_k_overlaps = []
    pairwise_rows = []

    for i1, i2 in pairs:
        r1 = rank_matrix[i1]
        r2 = rank_matrix[i2]
        s_res = spearmanr(r1, r2)
        s_val = float(s_res.statistic) if hasattr(s_res, "statistic") else float(s_res[0])
        spearman_coeffs.append(s_val)

        k_res = kendalltau(r1, r2)
        k_val = float(k_res.statistic) if hasattr(k_res, "statistic") else float(k_res[0])
        kendall_coeffs.append(k_val)

        row = {
            "Run_1": i1 + 1,
            "Run_2": i2 + 1,
            "Spearman_Rho": s_val,
            "Kendall_Tau": k_val,
        }

        if top_k is not None and top_k > 0:
            set1 = set(ordered_lists[i1][:top_k])
            set2 = set(ordered_lists[i2][:top_k])
            union = set1 | set2
            jaccard = float(len(set1 & set2) / len(union)) if union else 1.0
            top_k_overlaps.append(jaccard)
            row[f"Top_{top_k}_Jaccard"] = jaccard

        pairwise_rows.append(row)

    mean_spearman = float(np.mean(spearman_coeffs))
    median_spearman = float(np.median(spearman_coeffs))
    mean_kendall = float(np.mean(kendall_coeffs))
    median_kendall = float(np.median(kendall_coeffs))
    mean_top_k = float(np.mean(top_k_overlaps)) if top_k_overlaps else np.nan

    feat_summary_rows = []
    for j, feat in enumerate(all_features):
        feat_ranks = rank_matrix[:, j]
        record = {
            "Feature": feat,
            "Mean_Rank": float(np.mean(feat_ranks)),
            "Median_Rank": float(np.median(feat_ranks)),
            "Std_Rank": float(np.std(feat_ranks, ddof=1)) if len(feat_ranks) > 1 else 0.0,
            "Min_Rank": float(np.min(feat_ranks)),
            "Max_Rank": float(np.max(feat_ranks)),
        }
        if top_k is not None and top_k > 0:
            freq = float(np.mean([feat in set(ordered_lists[i][:top_k]) for i in range(len(rankings))]))
            record[f"Top_{top_k}_Frequency"] = freq
        feat_summary_rows.append(record)

    summary_df = pd.DataFrame(feat_summary_rows).sort_values("Mean_Rank").reset_index(drop=True)

    return {
        "pairwise_comparisons": pd.DataFrame(pairwise_rows),
        "mean_spearman_rho": mean_spearman,
        "median_spearman_rho": median_spearman,
        "mean_kendall_tau": mean_kendall,
        "median_kendall_tau": median_kendall,
        "mean_top_k_jaccard": mean_top_k,
        "feature_stability_summary": summary_df,
        "n_rankings": len(rankings),
        "n_features": n_features,
        "is_stable": bool(mean_spearman >= stability_threshold),
    }


def evaluate_group_bootstrap_uncertainty(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    target_column: str,
    config: Any = None,
    n_bootstraps: int = 15,
    *,
    random_state: int = 42,
    group_column: str = "trajectory_id",
    run_lofo: bool = False,
) -> dict[str, Any]:
    """Evaluate generalization uncertainty via group-level (trajectory) bootstrap resampling.

    Resamples whole independent trajectories with replacement rather than individual
    correlated frames, preventing pseudo-replication bias.

    Parameters
    ----------
    features : pd.DataFrame
        Frame-level feature matrix.
    manifest : pd.DataFrame
        Frame-level trajectory metadata manifest.
    target_column : str
        Target column name in manifest.
    config : MLWorkflowConfig, optional
        Configuration for ML workflow.
    n_bootstraps : int
        Number of group bootstrap iterations (default 15).
    random_state : int
        Base random seed.
    group_column : str
        Name of trajectory / replica grouping column (default 'trajectory_id').
    run_lofo : bool
        Whether to run LOFO importance within each bootstrap iteration (default False).

    Returns
    -------
    dict[str, Any]
        Dictionary containing per-bootstrap metrics, summary table with mean, std error,
        95% empirical bootstrap confidence intervals (2.5%, 97.5%), and group count.
    """
    from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow, validate_ml_inputs

    validate_ml_inputs(features, manifest, target_column)
    if n_bootstraps < 2:
        raise ValueError("n_bootstraps must be at least 2.")

    if group_column not in manifest.columns:
        raise ValueError(f"group_column '{group_column}' not found in manifest.")

    unique_groups = manifest[group_column].drop_duplicates().to_numpy()
    if len(unique_groups) < 2:
        raise ValueError("Group bootstrap requires at least 2 independent groups.")

    if config is None:
        config = MLWorkflowConfig()

    boot_rows = []
    group_indices = {g: manifest.index[manifest[group_column] == g].to_numpy() for g in unique_groups}

    for b in range(n_bootstraps):
        rng = np.random.default_rng(random_state + b + 1)
        sampled_groups = rng.choice(unique_groups, size=len(unique_groups), replace=True)

        selected_row_indices = []
        new_group_ids = []
        for instance_idx, g in enumerate(sampled_groups):
            idx_list = group_indices[g]
            selected_row_indices.extend(idx_list)
            new_group_ids.extend([f"{g}__boot_{instance_idx}"] * len(idx_list))

        boot_features = features.iloc[selected_row_indices].reset_index(drop=True)
        boot_manifest = manifest.iloc[selected_row_indices].copy().reset_index(drop=True)
        boot_manifest[group_column] = new_group_ids

        if boot_manifest[target_column].nunique() < 2:
            continue

        try:
            res = run_grouped_ml_workflow(
                boot_features,
                boot_manifest,
                target_column,
                config,
                run_lofo=run_lofo,
            )
            summary = res["model_summary"]
            rf_row = summary[summary["Model"] == "random_forest"]
            row_to_use = rf_row.iloc[0] if not rf_row.empty else summary.iloc[0]

            record = {
                "Bootstrap": b + 1,
                "Model": row_to_use["Model"],
                "Frame_Accuracy": float(row_to_use["Frame_Accuracy_Mean"]),
                "Trajectory_Mean_Accuracy": float(row_to_use["Trajectory_Mean_Accuracy_Mean"]),
                "Balanced_Accuracy": float(row_to_use["Balanced_Accuracy_Mean"]),
                "F1": float(row_to_use["F1_Mean"]),
                "ROC_AUC": float(row_to_use["ROC_AUC_Mean"]) if "ROC_AUC_Mean" in row_to_use and np.isfinite(row_to_use["ROC_AUC_Mean"]) else np.nan,
            }
            boot_rows.append(record)
        except Exception as e:
            warnings.warn(f"Bootstrap sample {b+1} failed: {e}", UserWarning, stacklevel=2)
            continue

    if not boot_rows:
        raise RuntimeError("All bootstrap iterations failed to evaluate.")

    boot_df = pd.DataFrame(boot_rows)
    metrics = ["Frame_Accuracy", "Trajectory_Mean_Accuracy", "Balanced_Accuracy", "F1", "ROC_AUC"]
    summary_records = []
    for m in metrics:
        vals = boot_df[m].dropna().to_numpy(dtype=float)
        if len(vals) == 0:
            continue
        summary_records.append({
            "Metric": m,
            "Mean": float(np.mean(vals)),
            "Std_Error": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            "Median": float(np.median(vals)),
            "CI_Lower": float(np.quantile(vals, 0.025)),
            "CI_Upper": float(np.quantile(vals, 0.975)),
            "N_Successful": len(vals),
        })

    summary_df = pd.DataFrame(summary_records)
    return {
        "bootstrap_results": boot_df,
        "bootstrap_summary": summary_df,
        "n_bootstraps": n_bootstraps,
        "n_successful": len(boot_df),
        "unique_groups": len(unique_groups),
    }


def evaluate_permutation_null_control(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    target_column: str,
    config: Any = None,
    n_permutations: int = 5,
    *,
    random_state: int = 42,
    group_level: bool = True,
) -> dict[str, Any]:
    """Execute label-shuffled permutation negative control to test for data leakage and overfitting.

    In a rigorous, leakage-free setup, shuffling target labels destroys the true
    biological/physical signal, yielding performance centered near chance (~50% balanced
    accuracy / ROC AUC for binary classification). If shuffled-label models achieve
    high accuracy, spurious correlation or data leakage is present.

    Parameters
    ----------
    features : pd.DataFrame
        Feature matrix.
    manifest : pd.DataFrame
        Trajectory manifest.
    target_column : str
        Target column name.
    config : MLWorkflowConfig, optional
        ML configuration.
    n_permutations : int
        Number of shuffled permutations (default 5).
    random_state : int
        Random seed.
    group_level : bool
        If True and targets are trajectory-level, shuffles labels across independent
        trajectories (preserving intra-trajectory autocorrelation and length). If False
        or target is frame-level, shuffles across frames.

    Returns
    -------
    dict[str, Any]
        Dictionary with permutation results, mean balanced accuracy, mean ROC AUC,
        chance consistency check (True if ~50%), and null distribution DataFrame.
    """
    from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow, validate_ml_inputs

    validate_ml_inputs(features, manifest, target_column)
    if n_permutations < 1:
        raise ValueError("n_permutations must be at least 1.")

    if config is None:
        config = MLWorkflowConfig()

    group_labels = manifest.groupby("trajectory_id", sort=False)[target_column].nunique()
    is_trajectory_level = bool((group_labels == 1).all())

    perm_rows = []
    for p in range(n_permutations):
        rng = np.random.default_rng(random_state + p + 1)
        perm_manifest = manifest.copy()

        if group_level and is_trajectory_level:
            trajs = manifest.drop_duplicates("trajectory_id").reset_index(drop=True)
            shuffled_targets = rng.permutation(trajs[target_column].to_numpy())
            mapping = dict(zip(trajs["trajectory_id"], shuffled_targets))
            perm_manifest[target_column] = perm_manifest["trajectory_id"].map(mapping)
        else:
            perm_manifest[target_column] = rng.permutation(manifest[target_column].to_numpy())

        res = run_grouped_ml_workflow(
            features,
            perm_manifest,
            target_column,
            config,
            run_lofo=False,
        )
        summary = res["model_summary"]
        for _, row in summary.iterrows():
            perm_rows.append({
                "Permutation": p + 1,
                "Model": row["Model"],
                "Frame_Accuracy": float(row["Frame_Accuracy_Mean"]),
                "Trajectory_Mean_Accuracy": float(row["Trajectory_Mean_Accuracy_Mean"]),
                "Balanced_Accuracy": float(row["Balanced_Accuracy_Mean"]),
                "ROC_AUC": float(row["ROC_AUC_Mean"]) if "ROC_AUC_Mean" in row and np.isfinite(row["ROC_AUC_Mean"]) else np.nan,
                "Trajectory_Vote_Accuracy": float(row["Trajectory_Vote_Accuracy_Mean"]) if "Trajectory_Vote_Accuracy_Mean" in row and np.isfinite(row["Trajectory_Vote_Accuracy_Mean"]) else np.nan,
                "Trajectory_Vote_Balanced_Accuracy": float(row["Trajectory_Vote_Balanced_Accuracy_Mean"]) if "Trajectory_Vote_Balanced_Accuracy_Mean" in row and np.isfinite(row["Trajectory_Vote_Balanced_Accuracy_Mean"]) else np.nan,
            })

    null_df = pd.DataFrame(perm_rows)
    rf_null = null_df[null_df["Model"] == "random_forest"]
    df_to_use = rf_null if not rf_null.empty else null_df

    mean_bal_acc = float(df_to_use["Balanced_Accuracy"].mean())
    median_bal_acc = float(df_to_use["Balanced_Accuracy"].median())
    auc_vals = df_to_use["ROC_AUC"].dropna()
    mean_auc = float(auc_vals.mean()) if not auc_vals.empty else np.nan

    is_chance_consistent = bool(0.30 <= mean_bal_acc <= 0.70)

    return {
        "null_distribution": null_df,
        "mean_balanced_accuracy": mean_bal_acc,
        "median_balanced_accuracy": median_bal_acc,
        "mean_roc_auc": mean_auc,
        "is_chance_consistent": is_chance_consistent,
        "n_permutations": n_permutations,
        "shuffling_mode": "trajectory_level" if (group_level and is_trajectory_level) else "frame_level",
    }
