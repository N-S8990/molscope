"""Unsupervised conformational-state discovery from ML coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

from molscope.analysis.quality.assessment import validate_manifest


@dataclass(frozen=True)
class StateDiscoveryConfig:
    n_states: int = 2
    n_restarts: int = 10
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be at least 2.")
        if self.n_restarts < 2:
            raise ValueError("n_restarts must be at least 2.")

    def to_dict(self) -> dict:
        return asdict(self)


def _coordinate_columns(coordinates: pd.DataFrame) -> list[str]:
    ckpca_underscore_cols = [c for c in coordinates if c.startswith("CKPCA_") and c[6:].isdigit()]
    if ckpca_underscore_cols:
        return sorted(ckpca_underscore_cols, key=lambda value: int(value[6:]))
    ckpca_cols = [c for c in coordinates if c.startswith("CKPCA") and not c.startswith("CKPCA_") and c[5:].isdigit()]
    if ckpca_cols:
        return sorted(ckpca_cols, key=lambda value: int(value[5:]))
    tic_underscore_cols = [c for c in coordinates if c.startswith("TIC_") and c[4:].isdigit()]
    if tic_underscore_cols:
        return sorted(tic_underscore_cols, key=lambda value: int(value[4:]))
    tic_cols = [c for c in coordinates if c.startswith("TIC") and not c.startswith("TIC_") and c[3:].isdigit()]
    if tic_cols:
        return sorted(tic_cols, key=lambda value: int(value[3:]))
    pc_cols = [column for column in coordinates if column.startswith("PC") and column[2:].isdigit()]
    if pc_cols:
        return sorted(pc_cols, key=lambda value: int(value[2:]))
    raise ValueError("State discovery requires at least one PC, TIC, or CKPCA coordinate column.")


def _validate_coordinate_identity(coordinates: pd.DataFrame, manifest: pd.DataFrame) -> None:
    identity = ["trajectory_id", "replica_id", "ensemble_id"]
    missing = set(identity) - set(coordinates.columns)
    if missing:
        raise ValueError(f"Coordinate table is missing identity columns: {sorted(missing)}")
    if not coordinates[identity].reset_index(drop=True).equals(manifest[identity].reset_index(drop=True)):
        raise ValueError("Coordinate and manifest identities do not match in row order.")
    if not ({"frame_order", "frame_index"} & set(coordinates.columns)):
        raise ValueError("Coordinate table needs frame_order or frame_index for transition analysis.")


def _transition_tables(assignments: pd.DataFrame, n_states: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count successive state changes within trajectories; do not infer rates."""
    rows = []
    for trajectory_id, subset in assignments.groupby("trajectory_id", sort=False):
        order_column = "frame_order" if "frame_order" in subset else "frame_index"
        if subset[order_column].duplicated().any():
            raise ValueError(f"Duplicate {order_column} values in trajectory {trajectory_id!r}.")
        subset = subset.sort_values(order_column)
        labels = subset["State"].to_numpy(dtype=int)
        counts = np.zeros((n_states, n_states), dtype=int)
        for source, target in zip(labels[:-1], labels[1:]):
            counts[source, target] += 1
        for source in range(n_states):
            total = counts[source].sum()
            for target in range(n_states):
                rows.append({
                    "trajectory_id": trajectory_id,
                    "Source_State": source,
                    "Target_State": target,
                    "Transition_Count": int(counts[source, target]),
                    "Conditional_Probability": float(counts[source, target] / total) if total else np.nan,
                })
    per_trajectory = pd.DataFrame(rows)
    aggregate = per_trajectory.groupby(["Source_State", "Target_State"], as_index=False)["Transition_Count"].sum()
    aggregate["Conditional_Probability"] = aggregate.groupby("Source_State")["Transition_Count"].transform(
        lambda values: values / values.sum() if values.sum() else np.nan
    )
    return per_trajectory, aggregate


def discover_conformational_states(
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    config: StateDiscoveryConfig = StateDiscoveryConfig(),
) -> dict[str, pd.DataFrame]:
    """Cluster ML coordinates and report state populations and stability.

    Frames are assignments, not independent replicas. Cluster reproducibility is
    measured across initializations; populations are reported per trajectory.
    """
    validate_manifest(manifest, len(coordinates))
    metadata = manifest.reset_index(drop=True)
    _validate_coordinate_identity(coordinates, metadata)
    pcs = _coordinate_columns(coordinates)
    values = coordinates.loc[:, pcs].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("State discovery requires finite coordinate values.")
    if len(values) <= config.n_states:
        raise ValueError("State discovery needs more frames than requested states.")
    labels_by_restart, models = [], []
    for restart in range(config.n_restarts):
        model = KMeans(n_clusters=config.n_states, n_init=1, random_state=config.random_state + restart)
        labels_by_restart.append(model.fit_predict(values))
        models.append(model)
    inertias = np.asarray([model.inertia_ for model in models])
    best_index = int(np.argmin(inertias))
    labels, best_model = labels_by_restart[best_index], models[best_index]
    agreements = [adjusted_rand_score(labels, candidate) for index, candidate in enumerate(labels_by_restart) if index != best_index]
    assignments = metadata.copy()
    assignments["State"] = labels
    assignments["State_Label"] = [f"state_{item}" for item in labels]
    population_rows = []
    for trajectory_id, subset in assignments.groupby("trajectory_id", sort=False):
        counts = subset["State"].value_counts().reindex(range(config.n_states), fill_value=0)
        for state, count in counts.items():
            population_rows.append({"trajectory_id": trajectory_id, "replica_id": subset["replica_id"].iloc[0], "ensemble_id": subset["ensemble_id"].iloc[0], "State": state, "Frame_Count": int(count), "Population": float(count / len(subset))})
    populations = pd.DataFrame(population_rows)
    distributions = {
        trajectory_id: subset.sort_values("State")["Population"].to_numpy()
        for trajectory_id, subset in populations.groupby("trajectory_id", sort=False)
    }
    trajectory_ensembles = assignments.groupby("trajectory_id", sort=False)["ensemble_id"].first().to_dict()
    within_distances, cross_distances = [], []
    trajectory_ids = list(distributions)
    for index, left_id in enumerate(trajectory_ids):
        for right_id in trajectory_ids[index + 1:]:
            distance = float(jensenshannon(distributions[left_id], distributions[right_id]))
            (within_distances if trajectory_ensembles[left_id] == trajectory_ensembles[right_id] else cross_distances).append(distance)
    diagnostics = pd.DataFrame([{
        "State_Count": config.n_states, "Frame_Count": len(values), "Trajectory_Count": metadata["trajectory_id"].nunique(),
        "Silhouette": float(silhouette_score(values, labels)), "KMeans_Inertia": float(best_model.inertia_),
        "Initialization_ARI_Median": float(np.median(agreements)), "Initialization_ARI_Min": float(np.min(agreements)),
        "Within_Ensemble_State_Population_JSD_Median": float(np.median(within_distances)) if within_distances else np.nan,
        "Cross_Ensemble_State_Population_JSD_Median": float(np.median(cross_distances)) if cross_distances else np.nan,
    }])
    centers = pd.DataFrame(best_model.cluster_centers_, columns=pcs)
    centers.insert(0, "State", range(config.n_states))
    trajectory_transitions, aggregate_transitions = _transition_tables(assignments, config.n_states)
    return {
        "state_assignments": assignments,
        "state_populations": populations,
        "state_centers": centers,
        "state_diagnostics": diagnostics,
        "trajectory_state_transitions": trajectory_transitions,
        "aggregate_state_transitions": aggregate_transitions,
    }


def rank_state_explanatory_features(
    features: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    allow_inapplicable: bool = True,
) -> pd.DataFrame:
    """Describe measured feature separation among *derived* candidate states.

    This is deliberately an effect-size table, not supervised validation: state
    labels were inferred from the same trajectory data, frames are correlated,
    and the output cannot establish physical states, mechanisms, or causality.

    When ``allow_inapplicable`` is True, features with missing, non-finite, or
    inapplicable observations propagate transparently into the returned table
    with explicit ``Ranking_Status`` and ``Status_Reason`` fields without silent
    imputation or zero-filling.
    """
    if len(features) != len(assignments):
        raise ValueError("Feature matrix and state assignments must have the same row count.")
    if "State" not in assignments:
        raise ValueError("State assignments require a State column.")
    states = assignments["State"]
    if states.isna().any() or states.nunique() < 2:
        raise ValueError("State-explanatory ranking requires at least two assigned states.")
    numeric = features.apply(pd.to_numeric, errors="coerce")
    invalid = set(numeric.columns[~np.isfinite(numeric.to_numpy(dtype=float)).all(axis=0)])
    if invalid and not allow_inapplicable:
        sorted_invalid = sorted(invalid)
        raise ValueError(
            "State-explanatory ranking does not impute invalid features; "
            f"invalid columns include: {sorted_invalid[:8]}"
        )
    rows = []
    inapplicable_rows = []
    for feature in numeric:
        if feature in invalid:
            values = numeric[feature]
            valid_frac = float(np.isfinite(values.to_numpy(dtype=float)).mean())
            inapplicable_rows.append({
                "Feature": feature,
                "State_Explained_Variance_Eta_Squared": np.nan,
                "Max_State_Mean_Difference": np.nan,
                "Candidate_State_Count": int(states.nunique()),
                "Ranking_Evidence": "inapplicable_or_missing_observations",
                "Interpretation": "excluded_from_ranking_without_silent_imputation",
                "Ranking_Status": "inapplicable" if valid_frac == 0.0 else "missing_values",
                "Status_Reason": (
                    f"Feature contains non-finite or missing values (valid fraction {valid_frac:.1%}); "
                    "excluded from ranking without silent imputation."
                ),
            })
            continue

        values = numeric[feature]
        overall = float(values.mean())
        grouped = values.groupby(states, sort=True)
        means = grouped.mean()
        counts = grouped.size()
        between = float(np.sum(counts.to_numpy(dtype=float) * (means.to_numpy(dtype=float) - overall) ** 2))
        total = float(np.sum((values.to_numpy(dtype=float) - overall) ** 2))
        rows.append({
            "Feature": feature,
            "State_Explained_Variance_Eta_Squared": between / total if total > 0 else 0.0,
            "Max_State_Mean_Difference": float(means.max() - means.min()),
            "Candidate_State_Count": int(states.nunique()),
            "Ranking_Evidence": "derived_state_descriptive_effect_size",
            "Interpretation": "descriptive_candidate_state_separation_not_predictive_or_causal",
            "Ranking_Status": "ranked",
            "Status_Reason": "Complete finite observations across all frames.",
        })
    ranked_valid = pd.DataFrame(rows)
    if not ranked_valid.empty:
        ranked_valid = ranked_valid.sort_values(
            ["State_Explained_Variance_Eta_Squared", "Max_State_Mean_Difference", "Feature"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranked_valid["Rank"] = ranked_valid.index + 1
    else:
        ranked_valid["Rank"] = pd.Series(dtype=float)

    if inapplicable_rows:
        inapplicable_df = pd.DataFrame(inapplicable_rows).sort_values("Feature").reset_index(drop=True)
        inapplicable_df["Rank"] = np.nan
        ranked = pd.concat([ranked_valid, inapplicable_df], ignore_index=True)
    else:
        ranked = ranked_valid

    return ranked


def evaluate_state_count_sensitivity(
    coordinates: pd.DataFrame,
    manifest: pd.DataFrame,
    state_counts: tuple[int, ...] = (2, 3, 4, 5),
    *,
    n_restarts: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Report diagnostics across candidate state counts without auto-selecting one.

    Silhouette and initialization agreement are screening diagnostics only. The
    output deliberately does not declare an objectively correct state count.
    """
    unique = tuple(dict.fromkeys(state_counts))
    if not unique or any(value < 2 for value in unique):
        raise ValueError("state_counts must contain one or more integers of at least 2.")
    records = []
    for count in unique:
        if count >= len(coordinates):
            continue
        outputs = discover_conformational_states(
            coordinates, manifest, StateDiscoveryConfig(n_states=count, n_restarts=n_restarts, random_state=random_state),
        )
        records.append(outputs["state_diagnostics"].iloc[0].to_dict())
    if not records:
        raise ValueError("No requested state count is smaller than the number of frames.")
    return pd.DataFrame.from_records(records).sort_values("State_Count").reset_index(drop=True)
