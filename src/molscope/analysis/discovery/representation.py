"""Target-free machine-learning discovery for frame-resolved MD features."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from molscope.analysis.quality.assessment import validate_manifest


@dataclass(frozen=True)
class RepresentationDiscoveryConfig:
    n_components: int = 5

    def __post_init__(self):
        if self.n_components < 1:
            raise ValueError("n_components must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


def _prepare(features: pd.DataFrame) -> pd.DataFrame:
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
    if invalid:
        raise ValueError(f"Representation discovery requires finite feature values; invalid columns include: {invalid[:8]}")
    active = numeric.columns[numeric.std(axis=0) > 1e-12].tolist()
    if not active:
        raise ValueError("Representation discovery requires at least one non-constant feature.")
    return numeric[active]


def _salience(values: pd.DataFrame, n_components: int) -> tuple[PCA, np.ndarray, np.ndarray]:
    scaled = StandardScaler().fit_transform(values)
    components = min(n_components, scaled.shape[0], scaled.shape[1])
    model = PCA(n_components=components, svd_solver="full")
    coordinates = model.fit_transform(scaled)
    salience = np.abs(model.components_).T @ model.explained_variance_ratio_
    return model, coordinates, salience


def discover_representation_features(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    config: RepresentationDiscoveryConfig = RepresentationDiscoveryConfig(),
) -> dict[str, pd.DataFrame]:
    """Rank features by their contribution to an unsupervised PCA representation.

    ``PCA_Salience`` is a representation-specific loading summary, not a
    causal or predictive importance claim.  The leave-one-trajectory-out ranks
    report whether that summary changes when an independent trajectory is
    withheld.
    """
    validate_manifest(manifest, len(features))
    values = _prepare(features).reset_index(drop=True)
    metadata = manifest.reset_index(drop=True)
    trajectory_ids = pd.unique(metadata["trajectory_id"])
    if len(trajectory_ids) < 2:
        raise ValueError("Representation stability requires at least two independent trajectories.")

    model, coordinates, salience = _salience(values, config.n_components)
    full_ranks = rankdata(-salience, method="average")
    fold_ranks = []
    for trajectory_id in trajectory_ids:
        train = metadata["trajectory_id"] != trajectory_id
        if train.sum() < 2:
            continue
        _, _, fold_salience = _salience(values.loc[train], config.n_components)
        fold_ranks.append(rankdata(-fold_salience, method="average"))
    ranks = np.asarray(fold_ranks, dtype=float)
    salience_table = pd.DataFrame({
        "Feature": values.columns,
        "PCA_Salience": salience,
        "PCA_Rank": full_ranks,
        "LOTO_Rank_Median": np.median(ranks, axis=0) if len(ranks) else np.nan,
        "LOTO_Rank_Std": np.std(ranks, axis=0, ddof=1) if len(ranks) > 1 else np.nan,
        "LOTO_Folds": len(ranks),
    }).sort_values(["PCA_Salience", "Feature"], ascending=[False, True]).reset_index(drop=True)
    # Express rank variability as a fraction of the available rank range. This
    # keeps stability comparable when different feature catalogues are used.
    salience_table["LOTO_Rank_Std_Normalized"] = salience_table["LOTO_Rank_Std"] / max(1, len(values.columns) - 1)
    salience_table["Rank"] = salience_table.index + 1

    coordinate_table = metadata.copy()
    for index in range(coordinates.shape[1]):
        coordinate_table[f"PC{index + 1}"] = coordinates[:, index]
    component_table = pd.DataFrame({
        "Component": [f"PC{index + 1}" for index in range(len(model.explained_variance_ratio_))],
        "Explained_Variance_Ratio": model.explained_variance_ratio_,
    })
    return {
        "representation_feature_salience": salience_table,
        "representation_coordinates": coordinate_table,
        "representation_components": component_table,
    }
