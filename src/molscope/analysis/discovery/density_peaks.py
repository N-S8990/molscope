"""Density Peak Advanced (DPA) Clustering for molecular conformation coordinates.

DPA identifies conformational state centers as points that are both local density
maxima and separated by anomalously large distances from other high-density frames.
Non-center frames are then recursively assigned to the same cluster as their
nearest neighbor with higher local density.

Mathematical Formulation:
--------------------------
1. Pairwise Distances:
   Given N conformational coordinate vectors x_i in R^d (e.g., from PCA, CKPCA, or TICA),
   the pairwise distance matrix D_ij = ||x_i - x_j|| is computed.

2. Local Density (rho_i):
   Computed via a continuous Gaussian kernel with cutoff distance d_c:
       rho_i = sum_j exp( - (d_ij / d_c)^2 )
   where d_c is either user-specified or estimated from the percentile of pairwise distances.

3. Higher-Density Distance (delta_i):
   The minimum distance from point i to any frame with higher local density:
       delta_i = min_{j: rho_j > rho_i} d_ij
   For the global density maximum (point with highest rho), delta is defined as:
       delta_max = max_j d_{max, j}
   The nearest higher-density neighbor is recorded as:
       nneigh_i = argmin_{j: rho_j > rho_i} d_ij

4. Cluster Center Identification:
   Cluster centers are outliers characterized by both high rho and anomalously large delta,
   evaluated via the decision metric:
       gamma_i = rho_i * delta_i
   Points exceeding a Z-score threshold (Z_{gamma, i} >= z_score) or designated via top-k
   ranking are flagged as cluster centers.

5. Cluster Assignment:
   Frames are sorted in descending order of local density. Non-center frames are assigned
   to the same cluster as their nearest higher-density neighbor:
       cluster_i = cluster_{nneigh_i}
   This guarantees deterministic, single-pass cluster propagation without artificial shape bias.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist

from molscope.analysis.quality.assessment import validate_manifest


@dataclass(frozen=True)
class DPAConfig:
    """Configuration parameters for Density Peak Advanced (DPA) Clustering."""

    cutoff_distance: float | None = None
    percentile: float = 2.0
    z_score: float = 2.0
    n_clusters: int | None = None
    min_clusters: int = 1
    metric: str = "euclidean"

    def __post_init__(self) -> None:
        if self.cutoff_distance is not None and self.cutoff_distance <= 0.0:
            raise ValueError(f"cutoff_distance must be positive, got {self.cutoff_distance}.")
        if self.percentile <= 0.0 or self.percentile >= 100.0:
            raise ValueError(f"percentile must be in (0, 100), got {self.percentile}.")
        if self.z_score < 0.0:
            raise ValueError(f"z_score must be non-negative, got {self.z_score}.")
        if self.n_clusters is not None and self.n_clusters < 1:
            raise ValueError(f"n_clusters must be >= 1, got {self.n_clusters}.")
        if self.min_clusters < 1:
            raise ValueError(f"min_clusters must be >= 1, got {self.min_clusters}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DPAResult:
    """Results from Density Peak Advanced (DPA) Clustering."""

    assignments: pd.DataFrame
    cluster_centers: np.ndarray
    n_clusters: int
    rho: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    z_scores: np.ndarray
    cutoff_distance: float
    decision_graph: pd.DataFrame
    populations: pd.DataFrame
    aggregate_transitions: pd.DataFrame
    per_trajectory_transitions: pd.DataFrame
    training_coordinates: np.ndarray
    coordinate_columns: list[str]

    @property
    def state_populations(self) -> dict[int, int]:
        """Mapping from cluster ID to frame count."""
        counts = self.assignments["Cluster"].value_counts().to_dict()
        return {int(k): int(v) for k, v in sorted(counts.items())}

    def predict(self, new_coordinates: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Assign out-of-sample frames to the nearest cluster center in coordinate space.

        Parameters
        ----------
        new_coordinates : pd.DataFrame | np.ndarray
            New coordinate frames matching the coordinate columns.

        Returns
        -------
        np.ndarray
            Array of assigned integer cluster IDs.
        """
        if isinstance(new_coordinates, pd.DataFrame):
            missing = set(self.coordinate_columns) - set(new_coordinates.columns)
            if missing:
                raise KeyError(f"Coordinates DataFrame missing required columns: {sorted(missing)}")
            sub = new_coordinates[self.coordinate_columns].apply(pd.to_numeric, errors="coerce")
            x_new = sub.to_numpy(dtype=float)
        else:
            x_new = np.asarray(new_coordinates, dtype=float)
            if x_new.shape[1] != len(self.coordinate_columns):
                raise ValueError(
                    f"Expected {len(self.coordinate_columns)} coordinates, got {x_new.shape[1]}."
                )

        if not np.isfinite(x_new).all():
            raise ValueError("Coordinates contain non-finite (NaN or Inf) values.")

        center_coords = self.training_coordinates[self.cluster_centers]
        dists = cdist(x_new, center_coords, metric="euclidean")
        return np.argmin(dists, axis=1)

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "assignments": self.assignments,
            "cluster_centers": self.cluster_centers,
            "n_clusters": self.n_clusters,
            "rho": self.rho,
            "delta": self.delta,
            "gamma": self.gamma,
            "z_scores": self.z_scores,
            "cutoff_distance": self.cutoff_distance,
            "decision_graph": self.decision_graph,
            "populations": self.populations,
            "aggregate_transitions": self.aggregate_transitions,
            "per_trajectory_transitions": self.per_trajectory_transitions,
            # Aliases for compatibility
            "state_assignments": self.assignments,
            "state_populations": self.populations,
            "aggregate_state_transitions": self.aggregate_transitions,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Key {key!r} not found in DPAResult.")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [
            "assignments",
            "cluster_centers",
            "n_clusters",
            "rho",
            "delta",
            "gamma",
            "z_scores",
            "cutoff_distance",
            "decision_graph",
            "populations",
            "aggregate_transitions",
            "per_trajectory_transitions",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "assignments": self.assignments,
            "cluster_centers": self.cluster_centers,
            "n_clusters": self.n_clusters,
            "rho": self.rho,
            "delta": self.delta,
            "gamma": self.gamma,
            "z_scores": self.z_scores,
            "cutoff_distance": self.cutoff_distance,
            "decision_graph": self.decision_graph,
            "populations": self.populations,
            "aggregate_transitions": self.aggregate_transitions,
            "per_trajectory_transitions": self.per_trajectory_transitions,
        }


def _extract_coordinate_matrix(
    coordinates: pd.DataFrame | np.ndarray,
    coordinate_prefix: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Extract numeric coordinate columns from DataFrame or array."""
    if isinstance(coordinates, np.ndarray):
        values = np.asarray(coordinates, dtype=float)
        col_names = [f"coord_{i + 1}" for i in range(values.shape[1])]
    elif isinstance(coordinates, pd.DataFrame):
        if coordinates.empty or len(coordinates) == 0 or len(coordinates.columns) == 0:
            raise ValueError("Coordinate matrix is empty.")
        all_cols = list(coordinates.columns)
        metadata_cols = {
            "trajectory_id",
            "replica_id",
            "ensemble_id",
            "frame_index",
            "frame_order",
            "time_ps",
            "State",
            "State_Label",
            "Cluster",
            "Cluster_Label",
        }

        if coordinate_prefix is not None:
            prefix = coordinate_prefix.strip().upper()
            if prefix in ("PCA", "PC"):
                matched = [c for c in all_cols if c.startswith("PC") and c[2:].isdigit()]
            elif prefix in ("TICA", "TIC"):
                matched = [
                    c for c in all_cols
                    if (c.startswith("TIC_") and c[4:].isdigit()) or (c.startswith("TIC") and c[3:].isdigit())
                ]
            elif prefix in ("CKPCA", "CONTACT_KPCA"):
                matched = [
                    c for c in all_cols
                    if (c.startswith("CKPCA_") and c[6:].isdigit()) or (c.startswith("CKPCA") and c[5:].isdigit())
                ]
            else:
                matched = [c for c in all_cols if c.startswith(coordinate_prefix)]
            if not matched:
                raise ValueError(f"No coordinate columns found matching prefix {coordinate_prefix!r}.")
            selected_cols = matched
        else:
            # Auto-detect PC, TIC, CKPCA, or numeric columns
            ckpca_cols = [c for c in all_cols if c.startswith("CKPCA_") and c[6:].isdigit()]
            tic_cols = [c for c in all_cols if (c.startswith("TIC_") and c[4:].isdigit()) or (c.startswith("TIC") and c[3:].isdigit())]
            pc_cols = [c for c in all_cols if c.startswith("PC") and c[2:].isdigit()]

            if ckpca_cols:
                selected_cols = sorted(ckpca_cols, key=lambda x: int(x[6:]))
            elif tic_cols:
                selected_cols = sorted(
                    tic_cols,
                    key=lambda x: int(x[4:]) if x.startswith("TIC_") else int(x[3:]),
                )
            elif pc_cols:
                selected_cols = sorted(pc_cols, key=lambda x: int(x[2:]))
            else:
                # Fall back to all numeric non-metadata columns
                candidate_cols = [c for c in all_cols if c not in metadata_cols]
                numeric = coordinates[candidate_cols].apply(pd.to_numeric, errors="coerce")
                valid_cols = [c for c in candidate_cols if numeric[c].notna().all()]
                if not valid_cols:
                    raise ValueError("No valid numeric coordinate columns found for DPA clustering.")
                selected_cols = valid_cols

        numeric_df = coordinates[selected_cols].apply(pd.to_numeric, errors="coerce")
        values = numeric_df.to_numpy(dtype=float)
        col_names = selected_cols
    else:
        raise TypeError(f"coordinates must be DataFrame or ndarray, got {type(coordinates).__name__}.")

    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Coordinate matrix is empty.")

    if not np.isfinite(values).all():
        raise ValueError("Coordinate matrix contains non-finite (NaN or Inf) values.")

    return values, col_names


def _estimate_cutoff_distance(
    x: np.ndarray,
    cutoff_distance: float | None,
    percentile: float,
    metric: str = "euclidean",
) -> float:
    """Determine the cutoff distance d_c for Gaussian kernel density estimation."""
    if cutoff_distance is not None:
        return float(cutoff_distance)

    n_samples = x.shape[0]
    if n_samples < 2:
        return 1.0

    sq_or_cond = pdist(x, metric=metric)
    if len(sq_or_cond) == 0:
        return 1.0

    d_c = float(np.percentile(sq_or_cond, percentile))
    if d_c > 1e-12:
        return d_c

    # If percentile yielded 0 (e.g. duplicate points), choose minimum positive distance
    pos_dists = sq_or_cond[sq_or_cond > 1e-12]
    if len(pos_dists) > 0:
        return float(np.min(pos_dists))
    return 1.0


def _compute_transitions(
    assignments: pd.DataFrame,
    n_clusters: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-trajectory and aggregate transition counts between cluster states."""
    if "trajectory_id" not in assignments.columns:
        # Create empty placeholder tables if trajectory identity is not tracked
        agg = pd.DataFrame(columns=["Source_State", "Target_State", "Transition_Count", "Conditional_Probability"])
        per_traj = pd.DataFrame(columns=["trajectory_id", "Source_State", "Target_State", "Transition_Count", "Conditional_Probability"])
        return per_traj, agg

    order_col = "frame_order" if "frame_order" in assignments.columns else ("frame_index" if "frame_index" in assignments.columns else None)
    rows = []

    for trajectory_id, subset in assignments.groupby("trajectory_id", sort=False):
        if order_col is not None:
            subset = subset.sort_values(order_col)
        labels = subset["Cluster"].to_numpy(dtype=int)
        counts = np.zeros((n_clusters, n_clusters), dtype=int)
        if len(labels) > 1:
            for src, tgt in zip(labels[:-1], labels[1:]):
                counts[src, tgt] += 1

        for src in range(n_clusters):
            tot = counts[src].sum()
            for tgt in range(n_clusters):
                rows.append({
                    "trajectory_id": trajectory_id,
                    "Source_State": src,
                    "Target_State": tgt,
                    "Transition_Count": int(counts[src, tgt]),
                    "Conditional_Probability": float(counts[src, tgt] / tot) if tot > 0 else np.nan,
                })

    per_trajectory = pd.DataFrame(rows)
    if not per_trajectory.empty:
        aggregate = per_trajectory.groupby(["Source_State", "Target_State"], as_index=False)["Transition_Count"].sum()
        aggregate["Conditional_Probability"] = aggregate.groupby("Source_State")["Transition_Count"].transform(
            lambda v: v / v.sum() if v.sum() > 0 else np.nan
        )
    else:
        aggregate = pd.DataFrame(columns=["Source_State", "Target_State", "Transition_Count", "Conditional_Probability"])

    return per_trajectory, aggregate


def cluster_dpa(
    coordinates: pd.DataFrame | np.ndarray,
    manifest: pd.DataFrame | None = None,
    config: DPAConfig | None = None,
    coordinate_prefix: str | None = None,
    **kwargs: Any,
) -> DPAResult:
    """Cluster low-dimensional conformational coordinates using Density Peak Advanced (DPA).

    Parameters
    ----------
    coordinates : pd.DataFrame | np.ndarray
        Low-dimensional reaction coordinates (PCA, CKPCA, or TICA).
    manifest : pd.DataFrame | None, default=None
        Optional frame manifest table containing trajectory identities.
    config : DPAConfig | None, default=None
        Configuration settings. If None, built from keyword arguments.
    coordinate_prefix : str | None, default=None
        Explicit prefix for coordinate columns (e.g., "PCA", "CKPCA", "TICA").
    **kwargs : Any
        Passed to DPAConfig if config is None.

    Returns
    -------
    DPAResult
        Clustering results including cluster assignments, decision graph, and transitions.
    """
    if config is None:
        config = DPAConfig(**kwargs)

    x, coord_cols = _extract_coordinate_matrix(coordinates, coordinate_prefix=coordinate_prefix)
    n_samples, n_dims = x.shape

    if manifest is not None:
        validate_manifest(manifest, n_samples)

    if n_samples < 2:
        raise ValueError(f"DPA clustering requires at least 2 frames, got {n_samples}.")

    # 1. Pairwise Distances D_ij
    d_mat = cdist(x, x, metric=config.metric)

    # 2. Cutoff Distance d_c
    d_c = _estimate_cutoff_distance(x, config.cutoff_distance, config.percentile, metric=config.metric)

    # 3. Local Density rho_i via Gaussian kernel
    # rho_i = sum_j exp( - (d_ij / d_c)^2 )
    # Including self-term j=i where exp(0) = 1.0 ensures rho_i >= 1.0 everywhere
    rho = np.sum(np.exp(-((d_mat / d_c) ** 2)), axis=1)

    # 4. Higher-Density Distance delta_i and Nearest Higher-Density Neighbor
    # Strict ordering to break ties deterministically: sort by (-rho[i], i)
    # The first element in sorted_indices is the global density peak.
    order = np.lexsort((np.arange(n_samples), -rho))

    delta = np.zeros(n_samples, dtype=float)
    nneigh = np.full(n_samples, -1, dtype=int)

    # Global maximum peak: delta is the maximum distance to any frame
    delta[order[0]] = np.max(d_mat[order[0]])
    nneigh[order[0]] = order[0]

    for k in range(1, n_samples):
        i = order[k]
        higher_indices = order[:k]
        dists_to_higher = d_mat[i, higher_indices]
        min_idx = int(np.argmin(dists_to_higher))
        delta[i] = dists_to_higher[min_idx]
        nneigh[i] = higher_indices[min_idx]

    # 5. Decision Metric gamma_i = rho_i * delta_i
    gamma = rho * delta

    # Z-scores of gamma: Z_{gamma, i} = (gamma_i - mean(gamma)) / std(gamma)
    gamma_mean = float(np.mean(gamma))
    gamma_std = float(np.std(gamma, ddof=1)) if n_samples > 1 else 1.0
    z_scores = (gamma - gamma_mean) / (gamma_std + 1e-12)

    # 6. Identify Cluster Centers
    if config.n_clusters is not None:
        # User explicitly requested top-k clusters by gamma
        n_centers = min(config.n_clusters, n_samples)
        centers = np.argsort(gamma)[::-1][:n_centers]
    else:
        # Detect outliers with high gamma (Z-score >= threshold)
        outlier_mask = z_scores >= config.z_score
        candidate_centers = np.where(outlier_mask)[0]

        if len(candidate_centers) < config.min_clusters:
            # Guarantee at least min_clusters
            n_req = min(config.min_clusters, n_samples)
            centers = np.argsort(gamma)[::-1][:n_req]
        else:
            # Sort detected centers descending by gamma
            centers = candidate_centers[np.argsort(gamma[candidate_centers])[::-1]]

    n_clusters = len(centers)

    # 7. Cluster Assignments for Non-Center Frames
    # Map centers to cluster IDs 0, 1, ..., n_clusters - 1
    cluster_labels = np.full(n_samples, -1, dtype=int)
    for c_id, center_idx in enumerate(centers):
        cluster_labels[center_idx] = c_id

    # Traverse frames in descending order of density
    for k in range(n_samples):
        i = order[k]
        if cluster_labels[i] == -1:
            # Assign to the same cluster as nearest higher-density neighbor
            cluster_labels[i] = cluster_labels[nneigh[i]]

    # 8. Core vs. Halo Analysis (Border density estimation)
    # Border frames are within distance d_c of another cluster
    cluster_roles = np.full(n_samples, "core", dtype=object)
    for c_id in range(n_clusters):
        c_mask = cluster_labels == c_id
        other_mask = ~c_mask
        if np.any(other_mask):
            c_indices = np.where(c_mask)[0]
            other_indices = np.where(other_mask)[0]
            sub_dists = d_mat[np.ix_(c_indices, other_indices)]
            border_c = c_indices[np.any(sub_dists <= d_c, axis=1)]
            if len(border_c) > 0:
                rho_border_max = float(np.max(rho[border_c]))
                # Points with density strictly less than the maximum border density are halo
                halo_c = c_indices[rho[c_indices] < rho_border_max]
                cluster_roles[halo_c] = "halo"

    # 9. Build Decision Graph DataFrame
    is_center_arr = np.zeros(n_samples, dtype=bool)
    is_center_arr[centers] = True

    decision_graph = pd.DataFrame({
        "Frame_Index": np.arange(n_samples),
        "rho": rho,
        "delta": delta,
        "gamma": gamma,
        "Z_Score": z_scores,
        "Is_Center": is_center_arr,
        "Cluster": cluster_labels,
        "Cluster_Label": [f"cluster_{c}" for c in cluster_labels],
        "Nearest_Higher_Density_Neighbor": nneigh,
        "Cluster_Role": cluster_roles,
    })

    # 10. Build Full Assignments DataFrame
    if manifest is not None:
        assignments_df = manifest.reset_index(drop=True).copy()
    elif isinstance(coordinates, pd.DataFrame):
        assignments_df = coordinates.reset_index(drop=True).copy()
    else:
        assignments_df = pd.DataFrame(index=pd.RangeIndex(n_samples))

    # Add coordinate columns if not already in assignments_df
    for col_idx, col_name in enumerate(coord_cols):
        if col_name not in assignments_df.columns:
            assignments_df[col_name] = x[:, col_idx]

    assignments_df["Cluster"] = cluster_labels
    assignments_df["Cluster_Label"] = [f"cluster_{c}" for c in cluster_labels]
    assignments_df["State"] = cluster_labels  # Alias for MolScope state discovery compatibility
    assignments_df["State_Label"] = [f"state_{c}" for c in cluster_labels]
    assignments_df["rho"] = rho
    assignments_df["delta"] = delta
    assignments_df["gamma"] = gamma
    assignments_df["Is_Center"] = is_center_arr
    assignments_df["Cluster_Role"] = cluster_roles

    # 11. Population Statistics
    pop_rows = []
    if "trajectory_id" in assignments_df.columns:
        for trajectory_id, subset in assignments_df.groupby("trajectory_id", sort=False):
            counts = subset["Cluster"].value_counts().reindex(range(n_clusters), fill_value=0)
            tot = len(subset)
            for c_id, count in counts.items():
                pop_rows.append({
                    "trajectory_id": trajectory_id,
                    "Cluster": int(c_id),
                    "Cluster_Label": f"cluster_{c_id}",
                    "Frame_Count": int(count),
                    "Population": float(count / tot) if tot > 0 else 0.0,
                })
    else:
        overall_counts = assignments_df["Cluster"].value_counts().reindex(range(n_clusters), fill_value=0)
        tot = len(assignments_df)
        for c_id, count in overall_counts.items():
            pop_rows.append({
                "trajectory_id": "all",
                "Cluster": int(c_id),
                "Cluster_Label": f"cluster_{c_id}",
                "Frame_Count": int(count),
                "Population": float(count / tot) if tot > 0 else 0.0,
            })
    populations_df = pd.DataFrame(pop_rows)

    # 12. Transition Tables
    per_traj_trans, agg_trans = _compute_transitions(assignments_df, n_clusters)

    return DPAResult(
        assignments=assignments_df,
        cluster_centers=centers,
        n_clusters=n_clusters,
        rho=rho,
        delta=delta,
        gamma=gamma,
        z_scores=z_scores,
        cutoff_distance=d_c,
        decision_graph=decision_graph,
        populations=populations_df,
        aggregate_transitions=agg_trans,
        per_trajectory_transitions=per_traj_trans,
        training_coordinates=x,
        coordinate_columns=coord_cols,
    )
