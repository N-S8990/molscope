"""Time-lagged Independent Component Analysis (TICA) for kinetic conformational representation.

TICA identifies the slowest-relaxing dynamical coordinates in molecular dynamics
simulations by solving the generalized eigenvalue problem:
    C_tau * v = lambda * (C_0 + epsilon * I) * v

where:
    - C_0 is the mean-centered instantaneous covariance matrix
    - C_tau is the symmetrized time-lagged covariance matrix at lag time tau
    - epsilon * I is a shrinkage regularization term ensuring positive definiteness
    - lambda are eigenvalues corresponding to autocorrelation at lag tau
    - v are eigenvectors (slowest reaction coordinates / TICs)

Time-lagged frame pairs (x_t, x_{t+tau}) are strictly constructed WITHIN individual
continuous trajectories (grouped by trajectory_id), strictly preventing cross-contamination
or artificial transitions between independent replicas.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import linalg

from molscope.analysis.quality.assessment import validate_manifest


@dataclass(frozen=True)
class TICAConfig:
    """Configuration parameters for Time-lagged Independent Component Analysis (TICA)."""

    lag_time: int = 1
    n_components: int = 2
    shrinkage: float = 1e-4
    dt: float = 1.0

    def __post_init__(self) -> None:
        if self.lag_time < 1:
            raise ValueError(f"lag_time must be a positive integer, got {self.lag_time}.")
        if self.n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {self.n_components}.")
        if self.shrinkage < 0.0:
            raise ValueError(f"shrinkage must be non-negative, got {self.shrinkage}.")
        if self.dt <= 0.0:
            raise ValueError(f"dt must be positive, got {self.dt}.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TICAResult:
    """Results from fitting TICA on molecular conformation features."""

    coordinates: pd.DataFrame
    eigenvalues: np.ndarray
    implied_timescales: np.ndarray
    loadings: pd.DataFrame
    eigenvectors: np.ndarray
    lag_time: int
    n_components: int
    shrinkage: float
    dt: float
    feature_names: list[str]
    mean_: np.ndarray
    c0: np.ndarray
    c_tau: np.ndarray
    all_eigenvalues: np.ndarray
    all_implied_timescales: np.ndarray
    all_eigenvectors: np.ndarray

    @property
    def components(self) -> pd.DataFrame:
        """Summary table for the retained TICA components."""
        k = len(self.eigenvalues)
        return pd.DataFrame({
            "Component": [f"TIC_{i + 1}" for i in range(k)],
            "Eigenvalue": self.eigenvalues,
            "Implied_Timescale": self.implied_timescales,
        })

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Project new feature frames onto the fitted TIC coordinates."""
        missing = set(self.feature_names) - set(features.columns)
        if missing:
            raise KeyError(f"Features DataFrame missing required columns: {sorted(missing)}")
        sub = features[self.feature_names].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(sub.to_numpy(dtype=float)).all():
            raise ValueError("Features DataFrame contains non-finite values.")
        x_centered = sub.to_numpy(dtype=float) - self.mean_
        projected = x_centered @ self.eigenvectors
        cols = [f"TIC_{i + 1}" for i in range(self.n_components)]
        return pd.DataFrame(projected, columns=cols, index=features.index)

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "coordinates": self.coordinates,
            "eigenvalues": self.eigenvalues,
            "implied_timescales": self.implied_timescales,
            "loadings": self.loadings,
            "eigenvectors": self.eigenvectors,
            "lag_time": self.lag_time,
            "n_components": self.n_components,
            "components": self.components,
            # Aliases for compatibility
            "tica_coordinates": self.coordinates,
            "representation_coordinates": self.coordinates,
            "tica_loadings": self.loadings,
            "tica_salience": self.loadings,
            "representation_feature_salience": self.loadings,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Key {key!r} not found in TICAResult.")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [
            "coordinates",
            "eigenvalues",
            "implied_timescales",
            "loadings",
            "eigenvectors",
            "lag_time",
            "n_components",
            "components",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "eigenvalues": self.eigenvalues,
            "implied_timescales": self.implied_timescales,
            "loadings": self.loadings,
            "eigenvectors": self.eigenvectors,
            "lag_time": self.lag_time,
            "n_components": self.n_components,
            "components": self.components,
        }


def _prepare_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Validate numeric values, check for finite values, and drop constant features."""
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
    if invalid:
        raise ValueError(f"TICA requires finite feature values; invalid columns include: {invalid[:8]}")
    active = numeric.columns[numeric.std(axis=0) > 1e-12].tolist()
    if not active:
        raise ValueError("TICA requires at least one non-constant feature.")
    return numeric[active], active


def fit_tica(
    feature_matrix: pd.DataFrame,
    manifest: pd.DataFrame,
    config: TICAConfig = TICAConfig(),
) -> TICAResult:
    """Fit Time-lagged Independent Component Analysis (TICA) on frame features.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        DataFrame of shape (N, D) containing frame-resolved feature measurements.
    manifest : pd.DataFrame
        DataFrame of shape (N, ...) containing trajectory metadata. Must contain
        'trajectory_id', 'replica_id', and 'ensemble_id'.
    config : TICAConfig, default=TICAConfig()
        TICA configuration specifying lag_time, n_components, shrinkage, and dt.

    Returns
    -------
    TICAResult
        Result containing projected coordinates, eigenvalues, implied timescales,
        eigenvectors, and feature loadings (kinetic salience).
    """
    validate_manifest(manifest, len(feature_matrix))
    if len(feature_matrix) < 2:
        raise ValueError("TICA requires at least two frames.")

    active_df, feature_names = _prepare_features(feature_matrix)
    n_frames, n_features = active_df.shape
    x_raw = active_df.to_numpy(dtype=float)

    # 1. Mean-centering across all frames
    mean_ = np.mean(x_raw, axis=0)
    x_centered = x_raw - mean_

    # 2. Instantaneous covariance C_0 = (1/N) * sum_t (x_t * x_t^T)
    c0 = (x_centered.T @ x_centered) / n_frames
    c0 = 0.5 * (c0 + c0.T)

    # Regularized C_0: C_0 + epsilon * I
    c0_reg = c0 + config.shrinkage * np.eye(n_features)

    # 3. Construct time-lagged pairs strictly within continuous individual trajectories
    tau = config.lag_time
    meta = manifest.reset_index(drop=True)

    idx_start: list[int] = []
    idx_lagged: list[int] = []

    for traj_id, group in meta.groupby("trajectory_id", sort=False):
        order_col = "frame_order" if "frame_order" in group else ("frame_index" if "frame_index" in group else None)
        if order_col is not None:
            if group[order_col].duplicated().any():
                raise ValueError(f"Duplicate {order_col} values in trajectory {traj_id!r}.")
            sorted_indices = group.sort_values(order_col).index.to_numpy(dtype=int)
        else:
            sorted_indices = group.index.to_numpy(dtype=int)

        t_len = len(sorted_indices)
        if t_len > tau:
            idx_start.extend(sorted_indices[: t_len - tau])
            idx_lagged.extend(sorted_indices[tau:])

    total_pairs = len(idx_start)
    if total_pairs == 0:
        raise ValueError(
            f"lag_time={tau} is too large: no trajectory has more than {tau} frames."
        )

    # 4. Time-lagged covariance C_tau = (1 / total_pairs) * sum_t (x_t * x_{t+tau}^T)
    x_t = x_centered[idx_start]
    x_tau = x_centered[idx_lagged]
    c_tau = (x_t.T @ x_tau) / total_pairs
    c_tau = 0.5 * (c_tau + c_tau.T)

    # 5. Solve generalized eigenvalue problem: C_tau * v = lambda * (C_0 + epsilon * I) * v
    try:
        evals, evecs = linalg.eigh(c_tau, c0_reg)
    except linalg.LinAlgError:
        # Fallback with augmented diagonal regularization if condition number is severe
        c0_reg_fallback = c0_reg + 1e-4 * np.trace(c0_reg) / max(1, n_features) * np.eye(n_features)
        evals, evecs = linalg.eigh(c_tau, c0_reg_fallback)

    # 6. Sort eigenvalues descending lambda_1 >= lambda_2 >= ... and sort eigenvectors
    sort_idx = np.argsort(evals)[::-1]
    all_evals = evals[sort_idx]
    all_evecs = evecs[:, sort_idx]

    # Enforce deterministic eigenvector sign (force largest component to be positive)
    max_abs_idx = np.argmax(np.abs(all_evecs), axis=0)
    signs = np.sign([all_evecs[max_abs_idx[i], i] for i in range(all_evecs.shape[1])])
    signs[signs == 0] = 1.0
    all_evecs = all_evecs * signs

    # 7. Compute implied timescales: t_i = - (tau * dt) / ln |lambda_i| (for lambda_i > 0)
    dt = config.dt
    all_timescales = np.full(len(all_evals), np.nan, dtype=float)
    for i, val in enumerate(all_evals):
        if val > 0:
            if val < 1.0:
                all_timescales[i] = - (tau * dt) / np.log(val)
            elif np.isclose(val, 1.0, atol=1e-12):
                all_timescales[i] = np.inf
            else:
                # Finite sample / regularization artifact where val > 1.0
                all_timescales[i] = np.nan

    # 8. Retain top n_components
    k = min(config.n_components, len(all_evals))
    retained_evals = all_evals[:k]
    retained_timescales = all_timescales[:k]
    retained_evecs = all_evecs[:, :k]

    # 9. Project all frames onto top n_components coordinates (TIC_1, TIC_2, ...)
    projected = x_centered @ retained_evecs
    coord_table = meta.copy()
    for i in range(k):
        coord_table[f"TIC_{i + 1}"] = projected[:, i]
        coord_table[f"TIC{i + 1}"] = projected[:, i]

    # 10. Feature loadings and kinetic salience
    pos_evals = np.maximum(retained_evals, 0.0)
    sum_pos = np.sum(pos_evals)
    if sum_pos > 0:
        weights = pos_evals / sum_pos
    else:
        weights = np.ones(k) / k

    kinetic_salience = np.abs(retained_evecs) @ weights

    loadings_dict: dict[str, Any] = {
        "Feature": feature_names,
        "Kinetic_Salience": kinetic_salience,
        "TICA_Salience": kinetic_salience,
    }
    for i in range(k):
        loadings_dict[f"TIC_{i + 1}"] = retained_evecs[:, i]
        loadings_dict[f"TIC{i + 1}"] = retained_evecs[:, i]

    loadings_table = (
        pd.DataFrame(loadings_dict)
        .sort_values(["Kinetic_Salience", "Feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    loadings_table["Rank"] = loadings_table.index + 1

    return TICAResult(
        coordinates=coord_table,
        eigenvalues=retained_evals,
        implied_timescales=retained_timescales,
        loadings=loadings_table,
        eigenvectors=retained_evecs,
        lag_time=tau,
        n_components=k,
        shrinkage=config.shrinkage,
        dt=dt,
        feature_names=feature_names,
        mean_=mean_,
        c0=c0,
        c_tau=c_tau,
        all_eigenvalues=all_evals,
        all_implied_timescales=all_timescales,
        all_eigenvectors=all_evecs,
    )
