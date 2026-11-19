"""Contact Kernel PCA (CKPCA) for non-linear conformational representation of molecular contacts.

Contact Kernel PCA projects high-dimensional residue-residue contact matrices
(e.g., C-alpha distances, minimum heavy-atom distances, or binary contact presence)
into low-dimensional non-linear conformational reaction coordinates.

Mathematical Formulation:
--------------------------
1. Kernel Matrix:
   Given contact feature vectors c_i in R^D across N frames, an RBF (Gaussian) or
   exponential kernel matrix K in R^(N x N) is computed:
       K_ij = exp(-gamma * ||c_i - c_j||^2)
   where gamma is either user-specified or estimated via the median heuristic:
       gamma = 1 / (2 * sigma^2) with sigma^2 = median(||c_i - c_j||^2).

2. Kernel Centering:
   Centering in the implicit high-dimensional reproducing kernel Hilbert space (RKHS):
       K_c = K - 1_N K - K 1_N + 1_N K 1_N
   where 1_N is the N x N matrix with entries 1/N.

3. Eigenvalue Decomposition:
   Solving the symmetric eigenvalue problem:
       K_c * v_k = lambda_k * v_k
   Eigenvalues are sorted in descending order: lambda_1 >= lambda_2 >= ... >= lambda_N >= 0.
   For each retained non-zero component k (where lambda_k > 0):
       alpha_k = v_k / sqrt(lambda_k)
   The low-dimensional non-linear coordinate embeddings for training frames are:
       y_k = K_c * alpha_k = sqrt(lambda_k) * v_k
   yielding CKPCA_1, CKPCA_2, ..., CKPCA_d.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from scipy.spatial.distance import cdist, pdist

from molscope.analysis.quality.assessment import validate_manifest


@dataclass(frozen=True)
class ContactKPCAConfig:
    """Configuration parameters for Contact Kernel PCA."""

    n_components: int = 4
    gamma: float | str | None = None
    kernel: str = "rbf"
    center: bool = True
    drop_constant: bool = True

    def __post_init__(self) -> None:
        if self.n_components < 1:
            raise ValueError(f"n_components must be a positive integer, got {self.n_components}.")
        if self.gamma is not None:
            if isinstance(self.gamma, (int, float)):
                if self.gamma <= 0.0:
                    raise ValueError(f"gamma must be positive, got {self.gamma}.")
            elif isinstance(self.gamma, str):
                if self.gamma.lower() not in ("median", "scale"):
                    raise ValueError(f"gamma heuristic must be 'median' or 'scale', got {self.gamma!r}.")
            else:
                raise TypeError(f"gamma must be float, str, or None, got {type(self.gamma).__name__}.")
        if self.kernel.lower() not in ("rbf", "gaussian", "exponential"):
            raise ValueError(f"Unsupported kernel {self.kernel!r}; expected 'rbf' or 'exponential'.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ContactKPCAResult:
    """Results from fitting Contact Kernel PCA on molecular contact features."""

    coordinates: pd.DataFrame
    eigenvalues: np.ndarray
    explained_variance_ratio: np.ndarray
    eigenvectors: np.ndarray
    alphas: np.ndarray
    gamma: float
    n_components: int
    kernel: str
    feature_names: list[str]
    training_features: np.ndarray
    kernel_row_means: np.ndarray
    kernel_grand_mean: float
    salience: pd.DataFrame

    @property
    def components(self) -> pd.DataFrame:
        """Summary table for retained Contact Kernel PCA components."""
        k = len(self.eigenvalues)
        cum_var = np.cumsum(self.explained_variance_ratio)
        return pd.DataFrame({
            "Component": [f"CKPCA_{i + 1}" for i in range(k)],
            "Eigenvalue": self.eigenvalues,
            "Explained_Variance_Ratio": self.explained_variance_ratio,
            "Cumulative_Variance_Ratio": cum_var,
        })

    def transform(self, features: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        """Project out-of-sample feature frames onto fitted Contact Kernel PCA coordinates.

        Parameters
        ----------
        features : pd.DataFrame | np.ndarray
            Frame feature table matching training feature columns.

        Returns
        -------
        pd.DataFrame
            Projected coordinates with columns CKPCA_1, CKPCA_2, ...
        """
        if isinstance(features, pd.DataFrame):
            missing = set(self.feature_names) - set(features.columns)
            if missing:
                raise KeyError(f"Features DataFrame missing required columns: {sorted(missing)}")
            sub = features[self.feature_names].apply(pd.to_numeric, errors="coerce")
            x_new = sub.to_numpy(dtype=float)
            index = features.index
        else:
            x_new = np.asarray(features, dtype=float)
            if x_new.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Expected {len(self.feature_names)} features, got {x_new.shape[1]}."
                )
            index = pd.RangeIndex(len(x_new))

        if not np.isfinite(x_new).all():
            raise ValueError("Features contain non-finite (NaN or Inf) values.")

        # Compute kernel between new frames and training frames: shape (M, N)
        sq_dists = cdist(x_new, self.training_features, metric="sqeuclidean")
        if self.kernel in ("rbf", "gaussian"):
            k_new = np.exp(-self.gamma * sq_dists)
        else:  # exponential
            dists = np.sqrt(np.maximum(sq_dists, 0.0))
            k_new = np.exp(-self.gamma * dists)

        # Center out-of-sample kernel:
        # K_new_c = K_new - row_means(K_new)[:, None] - train_row_means[None, :] + grand_mean
        k_new_row_means = np.mean(k_new, axis=1, keepdims=True)
        k_new_c = (
            k_new
            - k_new_row_means
            - self.kernel_row_means[np.newaxis, :]
            + self.kernel_grand_mean
        )

        # Out-of-sample projection: y_new = K_new_c @ alphas
        projected = k_new_c @ self.alphas
        k = projected.shape[1]
        cols = [f"CKPCA_{i + 1}" for i in range(k)]
        return pd.DataFrame(projected, columns=cols, index=index)

    def __getitem__(self, key: str) -> Any:
        mapping = {
            "coordinates": self.coordinates,
            "eigenvalues": self.eigenvalues,
            "explained_variance_ratio": self.explained_variance_ratio,
            "eigenvectors": self.eigenvectors,
            "alphas": self.alphas,
            "gamma": self.gamma,
            "n_components": self.n_components,
            "kernel": self.kernel,
            "components": self.components,
            "salience": self.salience,
            # Aliases for compatibility
            "ckpca_coordinates": self.coordinates,
            "representation_coordinates": self.coordinates,
            "ckpca_salience": self.salience,
            "representation_feature_salience": self.salience,
        }
        if key in mapping:
            return mapping[key]
        raise KeyError(f"Key {key!r} not found in ContactKPCAResult.")

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> list[str]:
        return [
            "coordinates",
            "eigenvalues",
            "explained_variance_ratio",
            "eigenvectors",
            "alphas",
            "gamma",
            "n_components",
            "kernel",
            "components",
            "salience",
        ]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinates": self.coordinates,
            "eigenvalues": self.eigenvalues,
            "explained_variance_ratio": self.explained_variance_ratio,
            "eigenvectors": self.eigenvectors,
            "alphas": self.alphas,
            "gamma": self.gamma,
            "n_components": self.n_components,
            "kernel": self.kernel,
            "components": self.components,
            "salience": self.salience,
        }


def _prepare_contact_matrix(
    features: pd.DataFrame | np.ndarray,
    drop_constant: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Validate numeric contact features, ensure finite values, and optionally filter constant columns."""
    if isinstance(features, pd.DataFrame):
        numeric = features.apply(pd.to_numeric, errors="coerce")
        col_names = list(numeric.columns)
        values = numeric.to_numpy(dtype=float)
    else:
        values = np.asarray(features, dtype=float)
        col_names = [f"contact_{i + 1}" for i in range(values.shape[1])]

    if values.size == 0 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("Contact feature matrix is empty.")

    if not np.isfinite(values).all():
        raise ValueError("Contact feature matrix contains non-finite (NaN or Inf) values.")

    if drop_constant and values.shape[0] > 1:
        stds = np.std(values, axis=0)
        active_mask = stds > 1e-12
        if not np.any(active_mask):
            raise ValueError("All contact features have zero variance.")
        values = values[:, active_mask]
        col_names = [name for name, active in zip(col_names, active_mask) if active]

    return values, col_names


def _estimate_gamma(
    x: np.ndarray,
    gamma: float | str | None,
    kernel: str = "rbf",
) -> float:
    """Compute gamma parameter via specified value or median / scale heuristic."""
    if isinstance(gamma, (int, float)):
        return float(gamma)

    n_samples, n_features = x.shape
    if n_samples < 2:
        return 1.0 / max(1, n_features)

    # Median heuristic: gamma = 1 / (2 * sigma^2) where sigma^2 = median(||c_i - c_j||^2)
    if gamma is None or (isinstance(gamma, str) and gamma.lower() == "median"):
        sq_dists = pdist(x, metric="sqeuclidean")
        med_sq = float(np.median(sq_dists))
        if med_sq > 1e-12:
            if kernel in ("rbf", "gaussian"):
                return 1.0 / (2.0 * med_sq)
            else:  # exponential
                return 1.0 / np.sqrt(med_sq)
        # Fall back if median squared distance is zero
        var_x = float(np.var(x))
        if var_x > 1e-12:
            return 1.0 / (n_features * var_x)
        return 1.0 / max(1, n_features)

    if isinstance(gamma, str) and gamma.lower() == "scale":
        var_x = float(np.var(x))
        if var_x > 1e-12:
            return 1.0 / (n_features * var_x)
        return 1.0 / max(1, n_features)

    raise ValueError(f"Unknown gamma setting: {gamma!r}.")


def _compute_feature_salience(
    x: np.ndarray,
    embeddings: np.ndarray,
    exp_var_ratio: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Compute feature salience for Contact Kernel PCA.

    Measures the correlation between individual contact features and non-linear CKPCA embeddings,
    weighted by the explained variance ratio of each component.
    """
    n_features = x.shape[1]
    n_comp = embeddings.shape[1]

    # Standardize X and embeddings along columns for fast correlation computation
    x_std = np.std(x, axis=0)
    emb_std = np.std(embeddings, axis=0)

    # Replace zero std with 1.0 to avoid division by zero
    x_safe_std = np.where(x_std > 1e-12, x_std, 1.0)
    emb_safe_std = np.where(emb_std > 1e-12, emb_std, 1.0)

    x_norm = (x - np.mean(x, axis=0)) / x_safe_std
    emb_norm = (embeddings - np.mean(embeddings, axis=0)) / emb_safe_std

    # Correlation matrix: shape (n_features, n_comp)
    n_frames = x.shape[0]
    corr = (x_norm.T @ emb_norm) / max(1, n_frames - 1)
    abs_corr = np.abs(corr)

    # Salience is variance-weighted sum of absolute correlations
    salience = abs_corr @ exp_var_ratio[:n_comp]

    # Rank features in descending order of salience
    salience_df = pd.DataFrame({
        "Feature": feature_names,
        "CKPCA_Salience": salience,
    })
    for c in range(n_comp):
        salience_df[f"CKPCA_{c + 1}_Correlation"] = corr[:, c]

    salience_df = salience_df.sort_values("CKPCA_Salience", ascending=False).reset_index(drop=True)
    salience_df["Rank"] = salience_df.index + 1
    return salience_df


def fit_contact_kpca(
    features: pd.DataFrame | np.ndarray,
    manifest: pd.DataFrame | None = None,
    config: ContactKPCAConfig | None = None,
    **kwargs: Any,
) -> ContactKPCAResult:
    """Fit Contact Kernel PCA on molecular contact features.

    Parameters
    ----------
    features : pd.DataFrame | np.ndarray
        Contact feature matrix (e.g., residue-residue distances, contacts, or PLIF).
    manifest : pd.DataFrame | None, default=None
        Optional frame manifest containing trajectory identity metadata.
    config : ContactKPCAConfig | None, default=None
        Configuration options. If None, built from keyword arguments.
    **kwargs : Any
        Passed to ContactKPCAConfig if config is None.

    Returns
    -------
    ContactKPCAResult
        Fitted Contact Kernel PCA model and non-linear embeddings.
    """
    if config is None:
        config = ContactKPCAConfig(**kwargs)

    x, feature_names = _prepare_contact_matrix(features, drop_constant=config.drop_constant)
    n_samples, n_features = x.shape

    if manifest is not None:
        validate_manifest(manifest, n_samples)

    if n_samples < 2:
        raise ValueError(f"Contact Kernel PCA requires at least 2 frames, got {n_samples}.")

    kernel_type = config.kernel.lower()
    gamma = _estimate_gamma(x, config.gamma, kernel=kernel_type)

    # 1. Compute Pairwise Kernel Matrix K_ij = exp(-gamma * ||c_i - c_j||^2)
    sq_dists = cdist(x, x, metric="sqeuclidean")
    if kernel_type in ("rbf", "gaussian"):
        k_mat = np.exp(-gamma * sq_dists)
    else:  # exponential
        dists = np.sqrt(np.maximum(sq_dists, 0.0))
        k_mat = np.exp(-gamma * dists)

    # 2. Kernel Centering: K_c = K - 1_N K - K 1_N + 1_N K 1_N
    if config.center:
        kernel_row_means = np.mean(k_mat, axis=1)
        kernel_grand_mean = float(np.mean(k_mat))
        k_c = (
            k_mat
            - kernel_row_means[:, np.newaxis]
            - kernel_row_means[np.newaxis, :]
            + kernel_grand_mean
        )
        # Ensure numerical symmetry
        k_c = 0.5 * (k_c + k_c.T)
    else:
        kernel_row_means = np.zeros(n_samples)
        kernel_grand_mean = 0.0
        k_c = k_mat

    # 3. Solve Eigenvalue Decomposition: K_c v_k = lambda_k v_k
    eigenvalues, eigenvectors = linalg.eigh(k_c)

    # Sort descending
    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    eigenvectors = eigenvectors[:, sort_idx]

    # Filter positive eigenvalues (tolerance for numerical precision)
    positive_mask = eigenvalues > 1e-10
    if not np.any(positive_mask):
        raise ValueError("Kernel matrix has no positive eigenvalues.")

    total_positive_variance = float(np.sum(eigenvalues[positive_mask]))

    # Determine number of components to retain
    n_pos = int(np.sum(positive_mask))
    k = min(config.n_components, n_pos)

    retained_eigenvalues = eigenvalues[:k]
    retained_eigenvectors = eigenvectors[:, :k]
    explained_variance_ratio = retained_eigenvalues / total_positive_variance

    # Embedding projections: y_k = sqrt(lambda_k) * v_k
    # and alpha expansion coefficients: alpha_k = v_k / sqrt(lambda_k)
    sqrt_lambdas = np.sqrt(retained_eigenvalues)
    embeddings = retained_eigenvectors * sqrt_lambdas[np.newaxis, :]
    alphas = retained_eigenvectors / sqrt_lambdas[np.newaxis, :]

    # Coordinate DataFrame
    coord_cols = [f"CKPCA_{i + 1}" for i in range(k)]
    if manifest is not None:
        coords_df = manifest.reset_index(drop=True).copy()
        for idx, col in enumerate(coord_cols):
            coords_df[col] = embeddings[:, idx]
            coords_df[f"CKPCA{idx + 1}"] = embeddings[:, idx]
    elif isinstance(features, pd.DataFrame):
        coords_df = pd.DataFrame(embeddings, columns=coord_cols, index=features.index)
        for idx in range(k):
            coords_df[f"CKPCA{idx + 1}"] = embeddings[:, idx]
    else:
        coords_df = pd.DataFrame(embeddings, columns=coord_cols)
        for idx in range(k):
            coords_df[f"CKPCA{idx + 1}"] = embeddings[:, idx]

    # Compute feature salience
    salience_df = _compute_feature_salience(x, embeddings, explained_variance_ratio, feature_names)

    return ContactKPCAResult(
        coordinates=coords_df,
        eigenvalues=retained_eigenvalues,
        explained_variance_ratio=explained_variance_ratio,
        eigenvectors=retained_eigenvectors,
        alphas=alphas,
        gamma=gamma,
        n_components=k,
        kernel=kernel_type,
        feature_names=feature_names,
        training_features=x,
        kernel_row_means=kernel_row_means,
        kernel_grand_mean=kernel_grand_mean,
        salience=salience_df,
    )
