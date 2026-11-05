"""Label-independent feature-quality diagnostics and Pareto prioritization."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


REQUIRED_MANIFEST_COLUMNS = frozenset({"trajectory_id", "replica_id", "ensemble_id"})


@dataclass(frozen=True)
class PrioritizationConfig:
    min_valid_fraction: float = 0.95
    constant_tolerance: float = 1e-12
    max_autocorrelation_lag: int = 100
    entropy_bins: int = 8
    max_correlation_frames: int = 5_000
    block_size_frames: int = 20
    pareto_bootstrap_samples: int = 0
    random_state: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


def validate_manifest(manifest: pd.DataFrame, expected_rows: int) -> None:
    missing = REQUIRED_MANIFEST_COLUMNS - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if len(manifest) != expected_rows:
        raise ValueError(f"Feature matrix has {expected_rows} rows but manifest has {len(manifest)} rows.")
    if manifest["trajectory_id"].isna().any():
        raise ValueError("Manifest trajectory_id values must be present.")


def _trajectory_series(values: np.ndarray, trajectory_ids: np.ndarray) -> Iterable[np.ndarray]:
    for trajectory_id in pd.unique(trajectory_ids):
        yield values[trajectory_ids == trajectory_id]


def _is_constant(values: np.ndarray, tolerance: float) -> bool:
    finite = values[np.isfinite(values)]
    return len(finite) > 0 and np.nanmax(finite) - np.nanmin(finite) <= tolerance


def _effective_sample_size(values: np.ndarray, max_lag: int, tolerance: float) -> float:
    """Estimate ESS using the initial positive sequence of autocorrelations."""
    finite = values[np.isfinite(values)]
    n = len(finite)
    if n < 4 or _is_constant(finite, tolerance):
        return np.nan
    centered = finite - np.mean(finite)
    denom = float(np.dot(centered, centered))
    if denom <= tolerance:
        return np.nan
    max_lag = min(max_lag, n // 2)
    autocorrelations = []
    for lag in range(1, max_lag + 1):
        rho = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if rho <= 0:
            break
        autocorrelations.append(rho)
    integrated_time = 1.0 + 2.0 * sum(autocorrelations)
    return float(np.clip(n / integrated_time, 1.0, float(n)))


def _occupancy_entropy(values: np.ndarray, bins: int, tolerance: float) -> float:
    finite = values[np.isfinite(values)]
    if len(finite) < 2 or _is_constant(finite, tolerance):
        return np.nan
    low, high = np.quantile(finite, [0.01, 0.99])
    if high - low <= tolerance:
        return np.nan
    clipped = np.clip(finite, low, high)
    counts, _ = np.histogram(clipped, bins=bins, range=(low, high))
    probabilities = counts[counts > 0] / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)) / np.log(bins))


def _block_standard_error(values: np.ndarray, block_size: int) -> tuple[float, int]:
    """Return a contiguous-block standard error for one trajectory.

    This is a sampling diagnostic, not a replacement for independent replicas.
    Incomplete final blocks are excluded so each block has the declared width.
    """
    blocks = []
    for start in range(0, len(values) - block_size + 1, block_size):
        block = values[start:start + block_size]
        if np.isfinite(block).all():
            blocks.append(float(np.mean(block)))
    if len(blocks) < 2:
        return np.nan, len(blocks)
    return float(np.std(blocks, ddof=1) / np.sqrt(len(blocks))), len(blocks)


def _feature_costs(costs: pd.DataFrame | None, features: pd.Index) -> pd.Series:
    if costs is None:
        return pd.Series(np.nan, index=features, dtype=float)
    required = {"Feature", "Elapsed_Seconds"}
    missing = required - set(costs.columns)
    if missing:
        raise ValueError(f"Feature-cost table is missing required columns: {sorted(missing)}")
    if costs["Feature"].duplicated().any():
        raise ValueError("Feature-cost table must contain at most one row per Feature.")
    values = pd.to_numeric(costs["Elapsed_Seconds"], errors="coerce")
    if (values.dropna() < 0).any():
        raise ValueError("Elapsed_Seconds must be non-negative when supplied.")
    return pd.Series(values.to_numpy(), index=costs["Feature"]).reindex(features)


def _replica_consistency(
    values: np.ndarray,
    manifest: pd.DataFrame,
    tolerance: float,
) -> float:
    """Distributional agreement across replica trajectories within each ensemble.

    The result is in (0, 1], where 1 means indistinguishable distributions
    under this normalized Wasserstein-distance diagnostic.  It is a sampling
    diagnostic, not a statistical proof that an observable has converged.
    """
    per_ensemble = []
    for _, subset in manifest.groupby("ensemble_id", sort=False):
        trajectory_ids = pd.unique(subset["trajectory_id"])
        if len(trajectory_ids) < 2:
            continue
        subset_values = values[subset.index.to_numpy()]
        subset_trajectory_ids = subset["trajectory_id"].to_numpy()
        distributions = [subset_values[subset_trajectory_ids == identifier] for identifier in trajectory_ids]
        pooled = np.concatenate([item[np.isfinite(item)] for item in distributions])
        pooled = pooled[np.isfinite(pooled)]
        if len(pooled) < 2:
            continue
        q25, q75 = np.quantile(pooled, [0.25, 0.75])
        scale = float(q75 - q25)
        distances = []
        for left, right in combinations(distributions, 2):
            left = left[np.isfinite(left)]
            right = right[np.isfinite(right)]
            if not len(left) or not len(right):
                continue
            distance = float(wasserstein_distance(left, right))
            if scale <= tolerance:
                distances.append(0.0 if distance <= tolerance else np.inf)
            else:
                distances.append(distance / scale)
        if distances:
            per_ensemble.append(1.0 / (1.0 + float(np.median(distances))))
    return float(np.median(per_ensemble)) if per_ensemble else np.nan


def _maximum_spearman_correlation(features: pd.DataFrame, max_rows: int) -> pd.Series:
    if len(features) > max_rows:
        indices = np.linspace(0, len(features) - 1, num=max_rows, dtype=int)
        features = features.iloc[indices]
    correlation = features.corr(method="spearman").abs()
    matrix = correlation.to_numpy(copy=True)
    np.fill_diagonal(matrix, 0.0)
    return pd.Series(matrix.max(axis=0), index=correlation.columns).fillna(0.0)


def _pareto_diagnostics(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return non-dominated membership and number of dominators for maximization objectives."""
    dominator_count = np.zeros(len(values), dtype=int)
    for index, candidate in enumerate(values):
        at_least_as_good = np.all(values >= candidate, axis=1)
        strictly_better = np.any(values > candidate, axis=1)
        dominator_count[index] = int(np.sum(at_least_as_good & strictly_better))
    return dominator_count == 0, dominator_count


def _resample_trajectories(
    features: pd.DataFrame, manifest: pd.DataFrame, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample complete trajectories within each ensemble.

    This estimates sensitivity to the available independent trajectories. It
    does not claim to estimate uncertainty from temporally correlated frames.
    """
    sampled_features = []
    sampled_manifest = []
    for ensemble_id, subset in manifest.groupby("ensemble_id", sort=False):
        trajectory_ids = pd.unique(subset["trajectory_id"])
        for draw_index, trajectory_id in enumerate(rng.choice(trajectory_ids, size=len(trajectory_ids), replace=True)):
            positions = subset.index[subset["trajectory_id"] == trajectory_id].to_numpy()
            copied_manifest = manifest.iloc[positions].copy()
            copied_manifest["trajectory_id"] = f"{trajectory_id}::bootstrap::{ensemble_id}::{draw_index}"
            sampled_features.append(features.iloc[positions])
            sampled_manifest.append(copied_manifest)
    return pd.concat(sampled_features, ignore_index=True), pd.concat(sampled_manifest, ignore_index=True)


def assess_feature_quality(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: PrioritizationConfig = PrioritizationConfig(),
    *,
    feature_costs: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Assess features without target labels and identify a Pareto candidate set.

    The Pareto objectives are sampling efficiency, replica consistency,
    occupancy entropy, and uniqueness.  A feature must be frame-resolved and
    have complete diagnostics to participate.  No weighted aggregate score is
    produced.
    """
    if config.block_size_frames < 1:
        raise ValueError("block_size_frames must be positive.")
    validate_manifest(manifest, len(features))
    manifest = manifest.reset_index(drop=True)
    numeric = features.apply(pd.to_numeric, errors="coerce")
    trajectory_ids = manifest["trajectory_id"].to_numpy()
    records = []

    for name in numeric.columns:
        values = numeric[name].to_numpy(dtype=float)
        valid_fraction = float(np.isfinite(values).mean())
        series = tuple(_trajectory_series(values, trajectory_ids))
        constant_by_trajectory = tuple(_is_constant(item, config.constant_tolerance) for item in series)
        resolution = (
            "unavailable" if valid_fraction == 0.0 else
            "trajectory" if all(constant_by_trajectory) else
            "frame"
        )
        ess = [_effective_sample_size(item, config.max_autocorrelation_lag, config.constant_tolerance) for item in series]
        block_errors = [_block_standard_error(item, config.block_size_frames) for item in series]
        finite_ess = np.asarray([value for value in ess if np.isfinite(value)], dtype=float)
        finite_block_errors = np.asarray([value for value, _ in block_errors if np.isfinite(value)], dtype=float)
        median_ess = float(np.median(finite_ess)) if len(finite_ess) else np.nan
        median_frames = float(np.median([len(item[np.isfinite(item)]) for item in series]))
        records.append({
            "Feature": name,
            "Valid_Fraction": valid_fraction,
            "Resolution": resolution,
            "Median_ESS": median_ess,
            "ESS_Ratio": float(median_ess / median_frames) if np.isfinite(median_ess) and median_frames else np.nan,
            "Median_Block_SE": float(np.median(finite_block_errors)) if len(finite_block_errors) else np.nan,
            "Median_Block_Count": float(np.median([count for _, count in block_errors])),
            "Replica_Consistency": _replica_consistency(values, manifest, config.constant_tolerance),
            "Occupancy_Entropy": _occupancy_entropy(values, config.entropy_bins, config.constant_tolerance),
        })

    quality = pd.DataFrame.from_records(records).set_index("Feature")
    costs = _feature_costs(feature_costs, quality.index)
    quality["Measured_Cost_Seconds"] = costs
    quality["Cost_Status"] = np.where(costs.notna(), "measured", "not_provided")
    frame_columns = quality.index[quality["Resolution"] == "frame"].tolist()
    if frame_columns:
        max_corr = _maximum_spearman_correlation(numeric[frame_columns], config.max_correlation_frames)
        quality.loc[frame_columns, "Max_Abs_Spearman"] = max_corr.reindex(frame_columns).to_numpy()
        quality.loc[frame_columns, "Uniqueness"] = 1.0 - quality.loc[frame_columns, "Max_Abs_Spearman"]
    quality["Pareto_Eligible"] = (
        (quality["Resolution"] == "frame")
        & (quality["Valid_Fraction"] >= config.min_valid_fraction)
        & quality[["ESS_Ratio", "Replica_Consistency", "Occupancy_Entropy", "Uniqueness"]].notna().all(axis=1)
    )
    quality["Pareto_Front"] = False
    quality["Dominator_Count"] = np.nan
    eligible = quality.index[quality["Pareto_Eligible"]].tolist()
    if eligible:
        objectives = quality.loc[eligible, ["ESS_Ratio", "Replica_Consistency", "Occupancy_Entropy", "Uniqueness"]].to_numpy(dtype=float)
        front, counts = _pareto_diagnostics(objectives)
        quality.loc[eligible, "Pareto_Front"] = front
        quality.loc[eligible, "Dominator_Count"] = counts
    quality["Pareto_Membership_Frequency"] = np.nan
    if config.pareto_bootstrap_samples:
        if config.pareto_bootstrap_samples < 1:
            raise ValueError("pareto_bootstrap_samples must be non-negative.")
        rng = np.random.default_rng(config.random_state)
        memberships = pd.Series(0, index=quality.index, dtype=int)
        bootstrap_config = replace(config, pareto_bootstrap_samples=0)
        for _ in range(config.pareto_bootstrap_samples):
            sampled_features, sampled_manifest = _resample_trajectories(numeric, manifest, rng)
            sampled_quality = assess_feature_quality(sampled_features, sampled_manifest, bootstrap_config)
            front = sampled_quality.loc[sampled_quality["Pareto_Front"], "Feature"]
            memberships.loc[memberships.index.intersection(front)] += 1
        quality["Pareto_Membership_Frequency"] = memberships / config.pareto_bootstrap_samples
    return quality.reset_index()
