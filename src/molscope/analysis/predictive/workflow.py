"""Leakage-aware, trajectory-grouped ML benchmarking and LOFO importance."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
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

    def __post_init__(self):
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if self.rf_estimators < 1 or self.bootstrap_samples < 1:
            raise ValueError("rf_estimators and bootstrap_samples must be positive.")

    def to_dict(self) -> dict:
        return asdict(self)


def validate_ml_inputs(features: pd.DataFrame, manifest: pd.DataFrame, target_column: str) -> None:
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
    if bad:
        raise ValueError(
            "ML workflow does not silently impute invalid values. Filter unavailable features first; "
            f"invalid columns include: {bad[:8]}"
        )


def _models(config: MLWorkflowConfig) -> dict[str, Pipeline | RandomForestClassifier]:
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=2_000, random_state=config.random_state)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=config.rf_estimators,
            random_state=config.random_state,
            n_jobs=-1,
        ),
    }


def _trajectory_mean_accuracy(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    return float(np.mean([
        accuracy_score(y_true[groups == group], y_pred[groups == group])
        for group in pd.unique(groups)
    ]))


def _evaluate(model, x_train, y_train, x_test, y_test, groups_test) -> dict[str, float]:
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1] if len(np.unique(y_train)) == 2 else None
    result = {
        "Frame_Accuracy": float(accuracy_score(y_test, predictions)),
        "Trajectory_Mean_Accuracy": _trajectory_mean_accuracy(y_test, predictions, groups_test),
        "Balanced_Accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "F1": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
    }
    if probabilities is not None and len(np.unique(y_test)) == 2:
        result["ROC_AUC"] = float(roc_auc_score(y_test, probabilities))
    else:
        result["ROC_AUC"] = np.nan
    return result


def _bootstrap_ci(values: np.ndarray, samples: int, random_state: int) -> tuple[float, float]:
    if len(values) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(random_state)
    means = np.array([np.mean(rng.choice(values, size=len(values), replace=True)) for _ in range(samples)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


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
    validate_ml_inputs(features, manifest, target_column)
    x = features.apply(pd.to_numeric, errors="coerce").reset_index(drop=True)
    metadata = manifest.reset_index(drop=True)
    y = metadata[target_column].to_numpy()
    groups = metadata["trajectory_id"].to_numpy()
    group_labels = metadata.groupby("trajectory_id", sort=False)[target_column].nunique()
    if (group_labels != 1).any():
        raise ValueError("Each trajectory_id must have exactly one target value for grouped classification.")
    class_group_counts = metadata.drop_duplicates("trajectory_id").groupby(target_column)["trajectory_id"].count()
    if len(class_group_counts) < 2 or class_group_counts.min() < config.n_splits:
        raise ValueError("Each target class needs at least n_splits independent trajectories.")

    # Stratify independent trajectories, so long trajectories cannot dominate
    # class balancing or leave a fold without a class.
    trajectories = metadata.drop_duplicates("trajectory_id").reset_index(drop=True)
    splitter = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.random_state)
    splits = []
    assignment_rows = []
    for fold, (train_groups, test_groups) in enumerate(
        splitter.split(trajectories, trajectories[target_column]), start=1
    ):
        test_ids = trajectories.iloc[test_groups]["trajectory_id"]
        test_mask = metadata["trajectory_id"].isin(test_ids).to_numpy()
        splits.append((np.flatnonzero(~test_mask), np.flatnonzero(test_mask)))
        for _, row in trajectories.iloc[test_groups].iterrows():
            assignment_rows.append({"trajectory_id": row["trajectory_id"], "Target": row[target_column], "Test_Fold": fold})
    metric_rows = []
    baseline_rows = []
    for model_name, model in _models(config).items():
        for fold, (train_index, test_index) in enumerate(splits, start=1):
            metrics = _evaluate(model, x.iloc[train_index], y[train_index], x.iloc[test_index], y[test_index], groups[test_index])
            metric_rows.append({"Model": model_name, "Fold": fold, "Train_Trajectories": len(pd.unique(groups[train_index])), "Test_Trajectories": len(pd.unique(groups[test_index])), **metrics})
            if model_name == "random_forest":
                baseline_rows.append({"Fold": fold, **metrics})
    fold_metrics = pd.DataFrame(metric_rows)
    summary_rows = []
    for model_name, subset in fold_metrics.groupby("Model", sort=False):
        record = {"Model": model_name, "Outer_Folds": len(subset)}
        for metric in ["Frame_Accuracy", "Trajectory_Mean_Accuracy", "Balanced_Accuracy", "F1", "ROC_AUC"]:
            values = subset[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_Mean"] = float(np.mean(values))
            record[f"{metric}_Std"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
        summary_rows.append(record)
    outputs = {"outer_fold_metrics": fold_metrics, "model_summary": pd.DataFrame(summary_rows), "fold_assignments": pd.DataFrame(assignment_rows)}
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
            metrics = _evaluate(
                _models(config)["random_forest"],
                x.loc[train_index, keep], y[train_index], x.loc[test_index, keep], y[test_index], groups[test_index],
            )
            lofo_rows.append({
                "Feature_Group": feature_group,
                "Feature_Count": len(removed_features),
                "Fold": fold,
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
        low, high = _bootstrap_ci(drops, config.bootstrap_samples, config.random_state)
        summary.append({
            "Feature_Group": feature_group,
            "Feature_Count": int(subset["Feature_Count"].iloc[0]),
            "Outer_Folds": len(subset),
            "Trajectory_Mean_Accuracy_Drop_Mean": float(np.mean(drops)),
            "Trajectory_Mean_Accuracy_Drop_CI_Lower": low,
            "Trajectory_Mean_Accuracy_Drop_CI_Upper": high,
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
    outputs["predictive_feature_ranking"] = predictive_ranking
    return outputs
