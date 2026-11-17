"""Leakage-aware, trajectory-grouped ML benchmarking and LOFO importance."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import warnings

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class MLWorkflowConfig:
    n_splits: int = 3
    random_state: int = 42
    rf_estimators: int = 300
    bootstrap_samples: int = 1_000
    include_baseline_dummy: bool = False
    rf_max_depth: int | None = 6
    rf_min_samples_leaf: int = 5
    lr_c: float = 0.5
    block_chunk_size: int = 30
    block_embargo: int = 2
    allow_inapplicable: bool = False

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if self.rf_estimators < 1 or self.bootstrap_samples < 1:
            raise ValueError("rf_estimators and bootstrap_samples must be positive.")
        if self.block_chunk_size < 1:
            raise ValueError("block_chunk_size must be at least 1.")
        if self.block_embargo < 0:
            raise ValueError("block_embargo cannot be negative.")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_ml_inputs(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    target_column: str,
    *,
    allow_inapplicable: bool = False,
) -> list[str]:
    required = {"trajectory_id", target_column}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")
    if len(features) != len(manifest):
        raise ValueError("Feature matrix and manifest must have the same row count.")
    if manifest["trajectory_id"].isna().any():
        raise ValueError("trajectory_id contains missing values.")
    if features.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")
    if manifest[target_column].isna().any():
        raise ValueError(f"Target column {target_column!r} contains missing values.")
    if features.empty:
        raise ValueError("Feature matrix has no columns.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    bad = numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)].tolist()
    if bad and not allow_inapplicable:
        raise ValueError(
            "ML workflow does not silently impute invalid values. Filter unavailable features first; "
            f"invalid columns include: {bad[:8]}"
        )
    return bad


def _models(config: MLWorkflowConfig) -> dict[str, Any]:
    models: dict[str, Any] = {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2_000,
                random_state=config.random_state,
                class_weight="balanced",
                C=getattr(config, "lr_c", 0.5),
            )),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            random_state=config.random_state,
            n_jobs=-1,
            class_weight="balanced",
            max_depth=getattr(config, "rf_max_depth", 6),
            min_samples_leaf=getattr(config, "rf_min_samples_leaf", 5),
            max_features="sqrt",
        ),
    }
    if getattr(config, "include_baseline_dummy", False):
        models["dummy"] = DummyClassifier(strategy="most_frequent")
    return models


def _trajectory_mean_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([
        accuracy_score(y_true[groups == group], y_pred[groups == group])
        for group in pd.unique(groups)
    ]))


def _trajectory_sample_weights(groups: np.ndarray) -> np.ndarray:
    """Give each independent trajectory equal total influence during fitting."""
    counts = pd.Series(groups).value_counts()
    weights = np.asarray([1.0 / counts[group] for group in groups], dtype=float)
    return weights / weights.mean()


def _trajectory_vote_metrics(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> dict[str, float]:
    """Score one majority-vote prediction per held-out trajectory when target is trajectory-level."""
    rows = []
    for group in pd.unique(groups):
        mask = groups == group
        true_values = pd.unique(y_true[mask])
        if len(true_values) != 1:
            return {
                "Trajectory_Vote_Accuracy": np.nan,
                "Trajectory_Vote_Balanced_Accuracy": np.nan,
            }
        predictions = pd.Series(y_pred[mask]).value_counts()
        # ``value_counts`` preserves first-occurrence order for equal counts,
        # making ties deterministic without looking at the target.
        rows.append((true_values[0], predictions.index[0]))
    trajectory_true, trajectory_pred = map(np.asarray, zip(*rows, strict=True))
    return {
        "Trajectory_Vote_Accuracy": float(accuracy_score(trajectory_true, trajectory_pred)),
        "Trajectory_Vote_Balanced_Accuracy": float(balanced_accuracy_score(trajectory_true, trajectory_pred)),
    }


def _evaluate(model, x_train, y_train, groups_train, x_test, y_test, groups_test) -> tuple[dict[str, float], np.ndarray, np.ndarray | None]:
    weights = _trajectory_sample_weights(groups_train)
    if isinstance(model, Pipeline):
        model.fit(x_train, y_train, model__sample_weight=weights)
    else:
        model.fit(x_train, y_train, sample_weight=weights)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    result = {
        "Frame_Accuracy": float(accuracy_score(y_test, predictions)),
        "Trajectory_Mean_Accuracy": _trajectory_mean_accuracy(y_test, predictions, groups_test),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "F1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }
    result.update(_trajectory_vote_metrics(y_test, predictions, groups_test))
    if probabilities is not None and len(np.unique(y_test)) == 2:
        result["ROC_AUC"] = float(roc_auc_score(y_test, probabilities))
    else:
        result["ROC_AUC"] = np.nan
    return result, predictions, probabilities


def _fold_resampling_interval(values: np.ndarray, samples: int, random_state: int) -> tuple[float, float]:
    """Describe variation across CV folds; this is not a confidence interval."""
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def purged_temporal_block_splits(
    metadata: pd.DataFrame,
    y: np.ndarray,
    n_splits: int,
    random_state: int,
    chunk_size: int = 30,
    embargo: int = 2,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Partition frames into contiguous temporal chunks with purged embargo buffer to eliminate autocorrelation leakage."""
    n_frames = len(metadata)
    chunk_len = max(5, min(chunk_size, n_frames // (n_splits * 2))) if n_frames >= n_splits * 10 else max(1, n_frames // n_splits)
    n_chunks = max(n_splits, n_frames // chunk_len)
    chunk_indices = np.array_split(np.arange(n_frames), n_chunks)
    chunk_labels = [pd.Series(y[chunk]).mode()[0] for chunk in chunk_indices]

    val_counts = pd.Series(chunk_labels).value_counts()
    if len(val_counts) < 2 or val_counts.min() < n_splits:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return list(skf.split(np.zeros(n_frames), y))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = []
    traj_ids = metadata["trajectory_id"].to_numpy() if "trajectory_id" in metadata.columns else None
    for train_chunk_idx, test_chunk_idx in skf.split(chunk_indices, chunk_labels):
        test_frames = np.concatenate([chunk_indices[i] for i in test_chunk_idx])
        train_raw = np.concatenate([chunk_indices[i] for i in train_chunk_idx])
        if embargo > 0:
            embargoed = set()
            for tf in test_frames:
                current_traj = traj_ids[tf] if traj_ids is not None else None
                for offset in range(-embargo, embargo + 1):
                    neighbor = tf + offset
                    if 0 <= neighbor < n_frames:
                        if traj_ids is None or traj_ids[neighbor] == current_traj:
                            embargoed.add(neighbor)
            train_frames = np.array([idx for idx in train_raw if idx not in embargoed])
            if len(train_frames) < n_splits or len(np.unique(y[train_frames])) < 2:
                train_frames = train_raw
        else:
            train_frames = train_raw
        splits.append((train_frames, test_frames))
    return splits


_purged_block_splits = purged_temporal_block_splits


def run_grouped_ml_workflow(
    features: pd.DataFrame,
    manifest: pd.DataFrame,
    target_column: str,
    config: MLWorkflowConfig = MLWorkflowConfig(),
    *,
    run_lofo: bool = True,
    feature_groups: dict[str, str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run outer trajectory-grouped evaluation and optional LOFO/group ablation.

    The primary importance endpoint is drop in trajectory-mean accuracy. This
    avoids presenting correlated frame count as the effective sample size.
    """
    bad_columns = validate_ml_inputs(features, manifest, target_column, allow_inapplicable=config.allow_inapplicable)
    x = features.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    if bad_columns:
        x = x.drop(columns=bad_columns)
        if x.empty:
            raise ValueError("All features in matrix contain non-finite values; no valid features remaining for ML.")
    metadata = manifest.reset_index(drop=True)
    y = metadata[target_column].to_numpy()
    groups = metadata["trajectory_id"].to_numpy()
    group_labels = metadata.groupby("trajectory_id", sort=False)[target_column].nunique()
    is_trajectory_level = bool((group_labels == 1).all())
    class_group_counts = (
        metadata.drop_duplicates("trajectory_id").groupby(target_column)["trajectory_id"].count()
        if is_trajectory_level
        else pd.Series(dtype=int)
    )
    can_group_by_trajectory = (
        is_trajectory_level
        and len(class_group_counts) >= 2
        and class_group_counts.min() >= config.n_splits
    )

    splits = []
    assignment_rows = []
    if can_group_by_trajectory:
        # Stratify independent trajectories, so long trajectories cannot dominate
        # class balancing or leave a fold without a class.
        trajectories = metadata.drop_duplicates("trajectory_id").reset_index(drop=True)
        if len(trajectories) < 10:
            warnings.warn(
                f"Insufficient independent samples: only {len(trajectories)} independent trajectories available across classes. "
                "Feature importance and generalization estimates carry high variance in trajectory-grouped cross-validation.",
                UserWarning,
                stacklevel=2,
            )
        splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
        for fold, (train_groups, test_groups) in enumerate(
            splitter.split(trajectories, trajectories[target_column]), start=1
        ):
            test_ids = trajectories.iloc[test_groups]["trajectory_id"]
            test_mask = metadata["trajectory_id"].isin(test_ids).to_numpy()
            splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
            for _, row in trajectories.iloc[test_groups].iterrows():
                assignment_rows.append({"trajectory_id": row["trajectory_id"], "Target": row[target_column], "Test_Fold": fold})
    else:
        # Purged temporal block cross-validation for conformational states or within-trajectory classifications
        val_counts = pd.Series(y).value_counts()
        if len(val_counts) < 2 or val_counts.min() < config.n_splits:
            raise ValueError(
                f"Target column {target_column!r} requires at least 2 distinct classes with at least {config.n_splits} frames each."
            )
        block_splits = _purged_block_splits(
            metadata,
            y,
            config.n_splits,
            config.random_state,
            chunk_size=config.block_chunk_size,
            embargo=config.block_embargo,
        )
        for fold, (train_index, test_index) in enumerate(block_splits, start=1):
            splits.append((train_index, test_index))
            for idx in test_index:
                assignment_rows.append({
                    "trajectory_id": metadata.loc[idx, "trajectory_id"],
                    "frame_index": metadata.loc[idx, "frame_index"] if "frame_index" in metadata.columns else idx,
                    "Target": str(y[idx]),
                    "Test_Fold": fold,
                })
    metric_rows = []
    prediction_rows = []
    baseline_rows = []
    for model_name, model in _models(config).items():
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            metrics, predictions, probabilities = _evaluate(
                model, x.iloc[train_index], y[train_index], groups[train_index],
                x.iloc[test_index], y[test_index], groups[test_index],
            )
            metric_rows.append({"Model": model_name, "Fold": fold, "Train_Trajectories": len(pd.unique(groups[train_index])), "Test_Trajectories": len(pd.unique(groups[test_index])), **metrics})
            for position, row_index in enumerate(test_index):
                prediction_rows.append({
                    "Model": model_name, "Fold": fold, "Frame_Row": int(row_index),
                    "trajectory_id": groups[row_index], "Target": y[row_index],
                    "Predicted_Target": predictions[position],
                    "Positive_Class_Probability": float(probabilities[position]) if probabilities is not None else np.nan,
                })
            if model_name == "random_forest":
                baseline_rows.append({"Fold": fold, **metrics})
    fold_metrics = pd.DataFrame(metric_rows)
    summary_rows = []
    for model_name, subset in fold_metrics.groupby("Model", sort=False):
        record = {"Model": model_name, "Outer_Folds": len(subset)}
        for metric in ["Frame_Accuracy", "Trajectory_Mean_Accuracy", "Trajectory_Vote_Accuracy", "Trajectory_Vote_Balanced_Accuracy", "Balanced_Accuracy", "F1", "ROC_AUC"]:
            values = subset[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_Mean"] = float(np.mean(values))
            record[f"{metric}_Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        summary_rows.append(record)
    outputs = {
        "outer_fold_metrics": fold_metrics,
        "model_summary": pd.DataFrame(summary_rows),
        "fold_assignments": pd.DataFrame(assignment_rows),
        "out_of_fold_predictions": pd.DataFrame(prediction_rows),
    }
    if not run_lofo:
        return outputs

    baseline = pd.DataFrame(baseline_rows).set_index("Fold")
    groups_to_features: dict[str, list[str]] = {}
    for feature in x.columns:
        group = feature_groups.get(feature, feature) if feature_groups else feature
        groups_to_features.setdefault(group, []).append(feature)
    lofo_rows = []
    for feature_group, removed_features in groups_to_features.items():
        keep = [column for column in x.columns if column not in removed_features]
        if not keep:
            continue
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            metrics, _, _ = _evaluate(
                _models(config)["random_forest"],
                x.loc[train_index, keep], y[train_index], groups[train_index],
                x.loc[test_index, keep], y[test_index], groups[test_index],
            )
            lofo_rows.append({
                "Feature_Group": feature_group,
                "Feature_Count": len(removed_features),
                "Fold": fold,
                "Baseline_Trajectory_Mean_Accuracy": float(baseline.loc[fold, "Trajectory_Mean_Accuracy"]),
                "Without_Group_Trajectory_Mean_Accuracy": metrics["Trajectory_Mean_Accuracy"],
                "Trajectory_Mean_Accuracy_Drop": float(baseline.loc[fold, "Trajectory_Mean_Accuracy"] - metrics["Trajectory_Mean_Accuracy"]),
                "Frame_Accuracy_Drop": float(baseline.loc[fold, "Frame_Accuracy"] - metrics["Frame_Accuracy"]),
                "Balanced_Accuracy_Drop": float(baseline.loc[fold, "Balanced_Accuracy"] - metrics["Balanced_Accuracy"]),
            })
    lofo_folds = pd.DataFrame(lofo_rows)
    if lofo_folds.empty:
        outputs["lofo_fold_drops"] = lofo_folds
        outputs["lofo_importance"] = lofo_folds
        outputs["predictive_feature_ranking"] = lofo_folds
        return outputs
    lofo_folds["Fold_Rank"] = lofo_folds.groupby("Fold")["Trajectory_Mean_Accuracy_Drop"].rank(method="average", ascending=False)
    summary = []
    for feature_group, subset in lofo_folds.groupby("Feature_Group", sort=False):
        drops = subset["Trajectory_Mean_Accuracy_Drop"].to_numpy(dtype=float)
        low, high = _fold_resampling_interval(drops, config.bootstrap_samples, config.random_state)
        summary.append({
            "Feature_Group": feature_group,
            "Feature_Count": int(subset["Feature_Count"].iloc[0]),
            "Outer_Folds": len(subset),
            "Baseline_Trajectory_Mean_Accuracy": float(np.mean(subset["Baseline_Trajectory_Mean_Accuracy"])),
            "Without_Group_Trajectory_Mean_Accuracy": float(np.mean(subset["Without_Group_Trajectory_Mean_Accuracy"])),
            "Trajectory_Mean_Accuracy_Drop_Mean": float(np.mean(drops)),
            "Trajectory_Mean_Accuracy_Gain_Without_Group": float(max(0.0, -np.mean(drops))),
            "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Lower": low,
            "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Upper": high,
            "Frame_Accuracy_Drop_Mean": float(np.mean(subset["Frame_Accuracy_Drop"])),
            "Mean_Rank": float(np.mean(subset["Fold_Rank"])),
            "Rank_Std": float(np.std(subset["Fold_Rank"], ddof=1)) if len(subset) > 1 else np.nan,
            "Positive_Drop_Frequency": float(np.mean(drops > 0)),
        })
    importance = pd.DataFrame(summary).sort_values(
        ["Trajectory_Mean_Accuracy_Drop_Mean", "Mean_Rank"], ascending=[False, True]
    ).reset_index(drop=True)
    importance["Rank"] = importance.index + 1
    outputs["lofo_fold_drops"] = lofo_folds
    outputs["lofo_importance"] = importance
    predictive_ranking = importance.copy()
    predictive_ranking.insert(0, "Ranking_Evidence", "trajectory_grouped_LOFO_random_forest")
    predictive_ranking.insert(1, "Importance_Endpoint", "trajectory_mean_accuracy_drop")
    predictive_ranking.insert(2, "Interpretation", "predictive_association_not_causality")
    predictive_ranking.insert(3, "Accuracy_Change_Definition", "baseline_minus_without_group;positive_drop_means_group_helped_prediction")
    predictive_ranking.insert(4, "Uncertainty_Qualification", "fold_resampling_is_descriptive_not_independent_replica_confidence_interval")
    predictive_ranking["Ranking_Status"] = "ranked"
    predictive_ranking["Status_Reason"] = "Evaluated via trajectory-grouped LOFO cross-validation."
    if bad_columns:
        inapplicable_rows = []
        for col in sorted(bad_columns):
            group_name = feature_groups.get(col, col) if feature_groups else col
            inapplicable_rows.append({
                "Ranking_Evidence": "inapplicable_or_missing_observations",
                "Importance_Endpoint": "trajectory_mean_accuracy_drop",
                "Interpretation": "excluded_from_ranking_without_silent_imputation",
                "Accuracy_Change_Definition": "baseline_minus_without_group;positive_drop_means_group_helped_prediction",
                "Uncertainty_Qualification": "not_applicable_excluded_feature",
                "Feature_Group": group_name,
                "Feature_Count": 1,
                "Outer_Folds": 0,
                "Baseline_Trajectory_Mean_Accuracy": np.nan,
                "Without_Group_Trajectory_Mean_Accuracy": np.nan,
                "Trajectory_Mean_Accuracy_Drop_Mean": np.nan,
                "Trajectory_Mean_Accuracy_Gain_Without_Group": np.nan,
                "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Lower": np.nan,
                "Trajectory_Mean_Accuracy_Drop_Fold_Resample_Upper": np.nan,
                "Frame_Accuracy_Drop_Mean": np.nan,
                "Mean_Rank": np.nan,
                "Rank_Std": np.nan,
                "Positive_Drop_Frequency": np.nan,
                "Rank": np.nan,
                "Ranking_Status": "inapplicable",
                "Status_Reason": "Feature contains non-finite or missing values; excluded without silent imputation.",
            })
        predictive_ranking = pd.concat([predictive_ranking, pd.DataFrame(inapplicable_rows)], ignore_index=True)
    outputs["predictive_feature_ranking"] = predictive_ranking
    return outputs
