"""Comparative supervised baselines and benchmark utilities for feature prioritization."""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def _extract_feature_matrix_and_names(
    X: pd.DataFrame | np.ndarray,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Validate and extract numpy matrix and feature names."""
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        mat = X.to_numpy(dtype=float)
    elif isinstance(X, np.ndarray):
        if X.ndim != 2:
            raise ValueError(f"Feature matrix X must be 2D, got shape {X.shape}")
        if feature_names is not None:
            if len(feature_names) != X.shape[1]:
                raise ValueError(
                    f"Length of feature_names ({len(feature_names)}) does not match columns ({X.shape[1]})"
                )
            names = list(feature_names)
        else:
            names = [f"feature_{i}" for i in range(X.shape[1])]
        mat = np.asarray(X, dtype=float)
    else:
        raise TypeError(f"X must be a pandas DataFrame or numpy ndarray, got {type(X)}")

    if not np.isfinite(mat).all():
        raise ValueError("Feature matrix contains non-finite values (NaN or Inf).")
    return mat, names


def _infer_task(y: np.ndarray, task: str = "auto") -> str:
    """Infer whether task is classification or regression."""
    if task not in ("auto", "classification", "regression"):
        raise ValueError(f"task must be 'auto', 'classification', or 'regression', got {task!r}")
    if task != "auto":
        return task

    # Check for discrete values
    if np.issubdtype(y.dtype, np.integer) or y.dtype == bool or np.issubdtype(y.dtype, np.str_):
        return "classification"
    unique_vals = len(np.unique(y[np.isfinite(y)] if np.issubdtype(y.dtype, np.floating) else y))
    if unique_vals <= 10:
        return "classification"
    return "regression"


def compute_rf_importance(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence,
    *,
    task: str = "auto",
    n_estimators: int = 100,
    max_depth: int | None = None,
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Random Forest Gini/MDI feature importances.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    y : array-like of shape (n_samples,)
        Target label or continuous activity / coordinate.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Task type. If 'auto', inferred from target dtype and unique count.
    n_estimators : int, default=100
        Number of trees in forest.
    max_depth : int or None, default=None
        Maximum tree depth.
    random_state : int, default=42
        Random seed for reproducibility.
    feature_names : list of str, optional
        Names of features if X is a numpy array.

    Returns
    -------
    pd.DataFrame
        Table containing 'Feature', 'RF_Importance', and 'RF_Rank', sorted descending.
    """
    X_mat, names = _extract_feature_matrix_and_names(X, feature_names)
    y_arr = np.asarray(y)
    resolved_task = _infer_task(y_arr, task)

    if resolved_task == "classification":
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )
    else:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
        )

    model.fit(X_mat, y_arr)
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": names,
        "RF_Importance": importances,
    }).sort_values(["RF_Importance", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["RF_Rank"] = df.index + 1
    return df


def compute_decision_tree_importance(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence,
    *,
    task: str = "auto",
    max_depth: int | None = None,
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Decision Tree feature importances from recursive split purity.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
        Input feature matrix.
    y : array-like
        Target values.
    task : {'auto', 'classification', 'regression'}, default='auto'
    max_depth : int or None, default=None
    random_state : int, default=42
    feature_names : list of str, optional

    Returns
    -------
    pd.DataFrame
        Table containing 'Feature', 'DT_Importance', and 'DT_Rank'.
    """
    X_mat, names = _extract_feature_matrix_and_names(X, feature_names)
    y_arr = np.asarray(y)
    resolved_task = _infer_task(y_arr, task)

    if resolved_task == "classification":
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
    else:
        model = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)

    model.fit(X_mat, y_arr)
    importances = model.feature_importances_

    df = pd.DataFrame({
        "Feature": names,
        "DT_Importance": importances,
    }).sort_values(["DT_Importance", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["DT_Rank"] = df.index + 1
    return df


def compute_mutual_info(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence,
    *,
    task: str = "auto",
    n_neighbors: int = 3,
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Mutual Information between features and target.

    Mutual information I(X; Y) quantifies non-linear dependency without
    assuming parametric distributions or linear correlations.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
    y : array-like
    task : {'auto', 'classification', 'regression'}, default='auto'
    n_neighbors : int, default=3
        Number of neighbors for continuous mutual information estimation.
    random_state : int, default=42
    feature_names : list of str, optional

    Returns
    -------
    pd.DataFrame
        Table containing 'Feature', 'Mutual_Info', and 'MI_Rank'.
    """
    X_mat, names = _extract_feature_matrix_and_names(X, feature_names)
    y_arr = np.asarray(y)
    resolved_task = _infer_task(y_arr, task)

    if resolved_task == "classification":
        mi = mutual_info_classif(
            X_mat, y_arr, n_neighbors=n_neighbors, random_state=random_state
        )
    else:
        mi = mutual_info_regression(
            X_mat, y_arr, n_neighbors=n_neighbors, random_state=random_state
        )

    df = pd.DataFrame({
        "Feature": names,
        "Mutual_Info": mi,
    }).sort_values(["Mutual_Info", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["MI_Rank"] = df.index + 1
    return df


def compute_permutation_importance(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence,
    *,
    estimator: Any | None = None,
    task: str = "auto",
    n_repeats: int = 5,
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute Permutation Importance by randomly shuffling feature values.

    Permutation importance measures the drop in model score when feature
    values are shuffled, breaking the relationship between feature and target.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray
    y : array-like
    estimator : fitted estimator or None
        If None, a default Random Forest estimator is trained on (X, y).
    task : {'auto', 'classification', 'regression'}, default='auto'
    n_repeats : int, default=5
        Number of permutation shuffles.
    random_state : int, default=42
    feature_names : list of str, optional

    Returns
    -------
    pd.DataFrame
        Table with 'Feature', 'Permutation_Importance_Mean',
        'Permutation_Importance_Std', and 'Permutation_Rank'.
    """
    X_mat, names = _extract_feature_matrix_and_names(X, feature_names)
    y_arr = np.asarray(y)
    resolved_task = _infer_task(y_arr, task)

    if estimator is None:
        if resolved_task == "classification":
            estimator = RandomForestClassifier(n_estimators=50, random_state=random_state)
        else:
            estimator = RandomForestRegressor(n_estimators=50, random_state=random_state)
        estimator.fit(X_mat, y_arr)

    perm = permutation_importance(
        estimator, X_mat, y_arr, n_repeats=n_repeats, random_state=random_state
    )

    df = pd.DataFrame({
        "Feature": names,
        "Permutation_Importance_Mean": perm.importances_mean,
        "Permutation_Importance_Std": perm.importances_std,
    }).sort_values(["Permutation_Importance_Mean", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["Permutation_Rank"] = df.index + 1
    return df


def compute_variance_importance(
    X: pd.DataFrame | np.ndarray,
    *,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute empirical feature variance baseline.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    feature_names : list of str, optional
        Feature names if X is ndarray.

    Returns
    -------
    pd.DataFrame
        Table containing 'Feature', 'Variance', and 'Variance_Rank', sorted descending.
    """
    X_mat, names = _extract_feature_matrix_and_names(X, feature_names)
    variances = np.var(X_mat, axis=0, ddof=1) if X_mat.shape[0] > 1 else np.zeros(X_mat.shape[1])
    df = pd.DataFrame({
        "Feature": names,
        "Variance": variances,
    }).sort_values(["Variance", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["Variance_Rank"] = df.index + 1
    return df


def compute_random_ranking(
    X: pd.DataFrame | np.ndarray,
    *,
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Compute random ranking baseline for null-hypothesis benchmarking.

    Parameters
    ----------
    X : pd.DataFrame or np.ndarray of shape (n_samples, n_features)
        Input feature matrix.
    random_state : int, default=42
        Random seed for reproducibility.
    feature_names : list of str, optional
        Feature names if X is ndarray.

    Returns
    -------
    pd.DataFrame
        Table containing 'Feature', 'Random_Score', and 'Random_Rank'.
    """
    _, names = _extract_feature_matrix_and_names(X, feature_names)
    rng = np.random.default_rng(random_state)
    scores = rng.uniform(0.0, 1.0, size=len(names))
    df = pd.DataFrame({
        "Feature": names,
        "Random_Score": scores,
    }).sort_values(["Random_Score", "Feature"], ascending=[False, True]).reset_index(drop=True)
    df["Random_Rank"] = df.index + 1
    return df


def compare_prioritization(
    sam_mfp_df: pd.DataFrame,
    baseline_df: pd.DataFrame,
    *,
    top_k: int = 10,
    baseline_score_col: str | None = None,
    baseline_rank_col: str | None = None,
) -> dict[str, Any]:
    """Compare multi-criteria SAM-MFP Pareto candidates against a baseline ranking.

    Parameters
    ----------
    sam_mfp_df : pd.DataFrame
        Output of prioritize_sam_mfp, containing 'Feature', 'SAM_MFP_Pareto_Front',
        and optionally 'SAM_MFP_Dominator_Count', 'PCA_Salience', etc.
    baseline_df : pd.DataFrame
        Output of a baseline importance method, containing 'Feature' and a score column.
    top_k : int, default=10
        Number of top baseline features to compare against the Pareto front.
    baseline_score_col : str, optional
        Name of baseline score column. If None, auto-detected.
    baseline_rank_col : str, optional
        Name of baseline rank column. If None, auto-detected.

    Returns
    -------
    dict[str, Any]
        Dictionary with comparison diagnostics:
        - 'merged_table': DataFrame merging both views
        - 'pareto_count': count of SAM-MFP Pareto front features
        - 'top_k': k evaluated
        - 'overlap_count': count of Pareto front features in top_k baseline
        - 'overlap_features': list of overlapping feature names
        - 'jaccard_index': Jaccard similarity between Pareto front and top_k baseline
        - 'top_k_recall': fraction of top-k baseline features captured in Pareto front
        - 'pareto_precision': fraction of Pareto features present in top-k baseline
        - 'spearman_rho': rank correlation between baseline score and representation salience
        - 'spearman_pvalue': p-value of rank correlation
        - 'divergent_baseline_top': features top-ranked in baseline but dominated in SAM-MFP
        - 'divergent_pareto_front': features in SAM-MFP front but low-ranked in baseline
        - 'summary_notes': list of explanatory observations
    """
    if "Feature" not in sam_mfp_df.columns:
        raise ValueError("sam_mfp_df must contain 'Feature' column.")
    if "Feature" not in baseline_df.columns:
        raise ValueError("baseline_df must contain 'Feature' column.")

    # Auto-detect baseline score column
    candidate_score_cols = [
        "RF_Importance",
        "Mutual_Info",
        "Permutation_Importance_Mean",
        "DT_Importance",
        "Variance",
        "Random_Score",
        "Baseline_Score",
        "Importance",
        "Score",
    ]
    if baseline_score_col is None:
        for col in candidate_score_cols:
            if col in baseline_df.columns:
                baseline_score_col = col
                break
        if baseline_score_col is None:
            # Pick first numeric column that is not 'Feature' or rank
            numeric_cols = [
                c for c in baseline_df.columns
                if c != "Feature" and not c.lower().endswith("rank") and pd.api.types.is_numeric_dtype(baseline_df[c])
            ]
            if numeric_cols:
                baseline_score_col = numeric_cols[0]
            else:
                raise ValueError("Could not auto-detect a baseline score column in baseline_df.")

    # Auto-detect rank column
    candidate_rank_cols = [
        "RF_Rank",
        "MI_Rank",
        "Permutation_Rank",
        "DT_Rank",
        "Variance_Rank",
        "Random_Rank",
        "Baseline_Rank",
        "Rank",
    ]
    if baseline_rank_col is None:
        for col in candidate_rank_cols:
            if col in baseline_df.columns:
                baseline_rank_col = col
                break
        if baseline_rank_col is None:
            baseline_df = baseline_df.copy()
            baseline_df["Auto_Rank"] = baseline_df[baseline_score_col].rank(ascending=False, method="min")
            baseline_rank_col = "Auto_Rank"

    # Merge tables
    merged = sam_mfp_df.merge(baseline_df, on="Feature", how="inner")
    if merged.empty:
        raise ValueError("No common features found between sam_mfp_df and baseline_df.")

    # Pareto set
    if "SAM_MFP_Pareto_Front" in merged.columns:
        pareto_mask = merged["SAM_MFP_Pareto_Front"].fillna(False).astype(bool)
    elif "Pareto_Front" in merged.columns:
        pareto_mask = merged["Pareto_Front"].fillna(False).astype(bool)
    else:
        raise ValueError("Neither 'SAM_MFP_Pareto_Front' nor 'Pareto_Front' found in sam_mfp_df.")
    pareto_features = set(merged.loc[pareto_mask, "Feature"])

    # Top-k baseline set
    sorted_baseline = merged.sort_values(baseline_score_col, ascending=False).reset_index(drop=True)
    actual_k = min(top_k, len(sorted_baseline))
    top_k_features = set(sorted_baseline.iloc[:actual_k]["Feature"])

    # Overlaps and set metrics
    overlap_features = sorted(pareto_features & top_k_features)
    overlap_count = len(overlap_features)
    union_count = len(pareto_features | top_k_features)
    jaccard_index = overlap_count / union_count if union_count > 0 else 0.0
    top_k_recall = overlap_count / actual_k if actual_k > 0 else 0.0
    pareto_precision = overlap_count / len(pareto_features) if pareto_features else 0.0

    # Rank correlation: baseline score vs salience
    salience_col = None
    for s_col in ("PCA_Salience", "TICA_Salience", "Kinetic_Salience", "State_Explained_Variance_Eta_Squared"):
        if s_col in merged.columns and merged[s_col].notna().any():
            salience_col = s_col
            break

    if salience_col is not None:
        valid_corr = merged[[baseline_score_col, salience_col]].dropna()
        if (
            len(valid_corr) >= 3
            and valid_corr[salience_col].std() > 1e-12
            and valid_corr[baseline_score_col].std() > 1e-12
        ):
            res = spearmanr(valid_corr[baseline_score_col], valid_corr[salience_col])
            spearman_rho = float(res.statistic)
            spearman_pvalue = float(res.pvalue)
        else:
            spearman_rho = float("nan")
            spearman_pvalue = float("nan")
    else:
        spearman_rho = float("nan")
        spearman_pvalue = float("nan")

    # Divergent features
    divergent_baseline_top = sorted(top_k_features - pareto_features)
    divergent_pareto_front = sorted(pareto_features - top_k_features)

    notes = [
        f"Compared {len(pareto_features)} SAM-MFP Pareto candidates against top-{actual_k} baseline features.",
        f"Overlap count: {overlap_count}/{actual_k} (top_k_recall: {top_k_recall:.1%}, Jaccard: {jaccard_index:.3f}).",
    ]
    if salience_col:
        notes.append(
            f"Spearman correlation between baseline {baseline_score_col} and {salience_col}: "
            f"rho={spearman_rho:.3f} (p={spearman_pvalue:.2e})."
        )
    if divergent_baseline_top:
        notes.append(
            f"{len(divergent_baseline_top)} top baseline features were not selected in SAM-MFP Pareto front "
            f"(likely due to replica inconsistency, poor ESS, or high redundancy)."
        )
    if divergent_pareto_front:
        notes.append(
            f"{len(divergent_pareto_front)} SAM-MFP Pareto features were outside baseline top-{actual_k} "
            f"(high-quality intrinsic dynamics uncoupled to current supervised target)."
        )

    return {
        "merged_table": merged,
        "pareto_count": len(pareto_features),
        "top_k": actual_k,
        "overlap_count": overlap_count,
        "overlap_features": overlap_features,
        "jaccard_index": jaccard_index,
        "top_k_recall": top_k_recall,
        "pareto_precision": pareto_precision,
        "spearman_rho": spearman_rho,
        "spearman_pvalue": spearman_pvalue,
        "divergent_baseline_top": divergent_baseline_top,
        "divergent_pareto_front": divergent_pareto_front,
        "summary_notes": notes,
    }


def compare_sam_mfp_with_baselines(
    sam_mfp_df: pd.DataFrame,
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray | Sequence | None = None,
    *,
    top_k: int = 10,
    task: str = "auto",
    baselines: Sequence[str] = ("rf", "dt", "mi", "permutation"),
    random_state: int = 42,
    feature_names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run multiple comparative baselines against SAM-MFP Pareto candidates.

    Parameters
    ----------
    sam_mfp_df : pd.DataFrame
        Output of prioritize_sam_mfp.
    X : pd.DataFrame or np.ndarray
        Feature matrix.
    y : array-like or None, default=None
        Target vector. Required for supervised baselines ('rf', 'dt', 'mi', 'permutation').
    top_k : int, default=10
        Top-k cut-off for each baseline comparator.
    task : {'auto', 'classification', 'regression'}, default='auto'
        Task type for supervised baselines.
    baselines : sequence of str, default=('rf', 'dt', 'mi', 'permutation')
        Subset of baselines to compute ('rf', 'dt', 'mi', 'permutation', 'variance', 'random').
    random_state : int, default=42
        Random seed for reproducibility.
    feature_names : list of str, optional
        Feature names if X is ndarray.

    Returns
    -------
    dict[str, Any]
        Dictionary with:
        - 'baseline_tables': dict of individual baseline DataFrames
        - 'comparisons': dict of compare_prioritization results per baseline
        - 'summary_table': pd.DataFrame comparing set overlap, recall, Jaccard, Spearman rho
        - 'consensus_top_k': features appearing in top-k across all evaluated baselines
        - 'consensus_pareto_overlap': consensus features captured in SAM-MFP Pareto front
        - 'summary_notes': diagnostic synthesis notes
    """
    baseline_tables: dict[str, pd.DataFrame] = {}
    comparisons: dict[str, dict[str, Any]] = {}
    summary_rows = []

    for b in baselines:
        b_clean = b.strip().lower()
        if b_clean in ("rf", "random_forest"):
            if y is None:
                raise ValueError("Target 'y' is required for Random Forest baseline.")
            b_df = compute_rf_importance(
                X, y, task=task, random_state=random_state, feature_names=feature_names
            )
            key = "rf"
        elif b_clean in ("dt", "decision_tree"):
            if y is None:
                raise ValueError("Target 'y' is required for Decision Tree baseline.")
            b_df = compute_decision_tree_importance(
                X, y, task=task, random_state=random_state, feature_names=feature_names
            )
            key = "dt"
        elif b_clean in ("mi", "mutual_info", "mutual_information"):
            if y is None:
                raise ValueError("Target 'y' is required for Mutual Information baseline.")
            b_df = compute_mutual_info(
                X, y, task=task, random_state=random_state, feature_names=feature_names
            )
            key = "mi"
        elif b_clean in ("permutation", "perm", "permutation_importance"):
            if y is None:
                raise ValueError("Target 'y' is required for Permutation Importance baseline.")
            b_df = compute_permutation_importance(
                X, y, task=task, random_state=random_state, feature_names=feature_names
            )
            key = "permutation"
        elif b_clean in ("variance", "var"):
            b_df = compute_variance_importance(X, feature_names=feature_names)
            key = "variance"
        elif b_clean in ("random", "rand"):
            b_df = compute_random_ranking(X, random_state=random_state, feature_names=feature_names)
            key = "random"
        else:
            raise ValueError(
                f"Unknown baseline name '{b}'. Choose from 'rf', 'dt', 'mi', 'permutation', 'variance', 'random'."
            )

        baseline_tables[key] = b_df
        comp = compare_prioritization(sam_mfp_df, b_df, top_k=top_k)
        comparisons[key] = comp

        summary_rows.append({
            "Baseline": key,
            "Top_K": comp["top_k"],
            "Pareto_Count": comp["pareto_count"],
            "Overlap_Count": comp["overlap_count"],
            "Top_K_Recall": comp["top_k_recall"],
            "Pareto_Precision": comp["pareto_precision"],
            "Jaccard_Index": comp["jaccard_index"],
            "Spearman_Rho": comp["spearman_rho"],
            "Spearman_PValue": comp["spearman_pvalue"],
        })

    summary_df = pd.DataFrame(summary_rows)

    # Find consensus features across top-k of all evaluated baselines
    top_sets = []
    for key, b_df in baseline_tables.items():
        actual_k = min(top_k, len(b_df))
        top_sets.append(set(b_df.iloc[:actual_k]["Feature"]))

    consensus_top_k = sorted(set.intersection(*top_sets)) if top_sets else []

    # Pareto set
    if "SAM_MFP_Pareto_Front" in sam_mfp_df.columns:
        pareto_mask = sam_mfp_df["SAM_MFP_Pareto_Front"].fillna(False).astype(bool)
    elif "Pareto_Front" in sam_mfp_df.columns:
        pareto_mask = sam_mfp_df["Pareto_Front"].fillna(False).astype(bool)
    else:
        pareto_mask = pd.Series(False, index=sam_mfp_df.index)
    pareto_features = set(sam_mfp_df.loc[pareto_mask, "Feature"])

    consensus_pareto_overlap = sorted(set(consensus_top_k) & pareto_features)

    notes = [
        f"Evaluated {len(baselines)} supervised baselines against {len(pareto_features)} SAM-MFP Pareto candidates.",
        f"Consensus baseline top-{top_k} features: {len(consensus_top_k)} ({consensus_top_k}).",
        f"Consensus features recovered on Pareto front: {len(consensus_pareto_overlap)}/{len(consensus_top_k)} "
        f"({len(consensus_pareto_overlap) / len(consensus_top_k):.1%})" if consensus_top_k else "No consensus features.",
    ]

    return {
        "baseline_tables": baseline_tables,
        "comparisons": comparisons,
        "summary_table": summary_df,
        "consensus_top_k": consensus_top_k,
        "consensus_pareto_overlap": consensus_pareto_overlap,
        "summary_notes": notes,
    }
