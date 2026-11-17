"""System-Aware Multi-Criteria Molecular Feature Prioritization (SAM-MFP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from molscope.analysis.quality.assessment import _pareto_diagnostics


@dataclass(frozen=True)
class SAMMFPConfig:
    """Eligibility constraints and objective configuration for SAM-MFP assessment.

    Parameters
    ----------
    min_loto_folds : int
        Minimum number of leave-one-trajectory-out folds required for representation
        stability eligibility. Defaults to 2.
    salience_metric : str
        Name of the representation salience column to maximize. Defaults to 'PCA_Salience'.
        Can also be 'TICA_Salience' or 'Kinetic_Salience'.
    include_state_separation : bool
        Whether to include state separation ('State_Explained_Variance_Eta_Squared')
        as an optimization objective if state explanatory features are provided.
    excluded_objectives : tuple[str, ...]
        Names of objectives to exclude from the Pareto optimization (useful for ablations).
    custom_objectives : tuple[str, ...] | None
        Explicit list of objectives to optimize. If specified, overrides default objective selection.
    """

    min_loto_folds: int = 2
    salience_metric: str = "PCA_Salience"
    include_state_separation: bool = False
    excluded_objectives: tuple[str, ...] = ()
    custom_objectives: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.min_loto_folds < 2:
            raise ValueError("min_loto_folds must be at least 2.")

    def to_dict(self) -> dict:
        return asdict(self)


_QUALITY_OBJECTIVES = ("ESS_Ratio", "Replica_Consistency", "Occupancy_Entropy", "Uniqueness")
_BASE_REQUIRED_QUALITY_COLUMNS = {"Feature", "Resolution", "Valid_Fraction"}


def prioritize_sam_mfp(
    quality: pd.DataFrame,
    representation_salience: pd.DataFrame,
    config: SAMMFPConfig = SAMMFPConfig(),
    state_explanatory: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Create target-free multi-criteria Pareto feature candidates.

    The output is deliberately not a scalar importance score. It combines
    measurement-quality, representation-discovery evidence, and optional
    derived-state separation into a Pareto front of non-dominated candidates.
    Supervised predictive importance belongs to a later, target-specific analysis.
    """
    if "Feature" not in quality.columns:
        raise ValueError("Quality table is missing required 'Feature' column.")
    if "Feature" not in representation_salience.columns:
        raise ValueError("Representation table is missing required 'Feature' column.")
    if quality["Feature"].duplicated().any() or representation_salience["Feature"].duplicated().any():
        raise ValueError("Feature names must be unique in both SAM-MFP inputs.")

    # Determine salience column to use
    salience_col = config.salience_metric
    if salience_col not in representation_salience.columns:
        if "PCA_Salience" in representation_salience.columns:
            salience_col = "PCA_Salience"
        elif "TICA_Salience" in representation_salience.columns:
            salience_col = "TICA_Salience"
        elif "Kinetic_Salience" in representation_salience.columns:
            salience_col = "Kinetic_Salience"

    # Validate quality columns
    required_quality = set(_BASE_REQUIRED_QUALITY_COLUMNS)
    if config.custom_objectives is not None:
        for obj in config.custom_objectives:
            if obj in _QUALITY_OBJECTIVES:
                required_quality.add(obj)
    else:
        for obj in _QUALITY_OBJECTIVES:
            if obj not in config.excluded_objectives:
                required_quality.add(obj)
    missing_quality = required_quality - set(quality.columns)
    if missing_quality:
        raise ValueError(f"Quality table is missing required columns: {sorted(missing_quality)}")

    # Merge quality and discovery tables
    table = quality.merge(representation_salience, on="Feature", how="left")

    # Merge state explanatory if provided
    if state_explanatory is not None:
        if "Feature" not in state_explanatory.columns:
            raise ValueError("State explanatory table is missing required 'Feature' column.")
        if state_explanatory["Feature"].duplicated().any():
            raise ValueError("Feature names must be unique in state_explanatory table.")
        cols_to_merge = ["Feature"]
        if "State_Explained_Variance_Eta_Squared" in state_explanatory.columns:
            cols_to_merge.append("State_Explained_Variance_Eta_Squared")
        for col in state_explanatory.columns:
            if col not in cols_to_merge and col not in table.columns:
                cols_to_merge.append(col)
        table = table.merge(state_explanatory[cols_to_merge], on="Feature", how="left")

    notes = [
        "SAM-MFP candidates are Pareto trade-offs, not a scalar importance ranking.",
        "Representation salience is target-free evidence, not causal or predictive importance.",
    ]

    # Compute Representation_Stability if LOTO columns are present
    if "LOTO_Rank_Std_Normalized" in table.columns:
        table["Representation_Stability"] = (1.0 - table["LOTO_Rank_Std_Normalized"]).clip(lower=0.0, upper=1.0)

    # Determine objectives
    if config.custom_objectives is not None:
        objective_columns = list(config.custom_objectives)
    else:
        objective_columns = [col for col in _QUALITY_OBJECTIVES if col not in config.excluded_objectives]
        if salience_col in table.columns and salience_col not in config.excluded_objectives:
            objective_columns.append(salience_col)
        if "Representation_Stability" in table.columns and "Representation_Stability" not in config.excluded_objectives:
            objective_columns.append("Representation_Stability")
        if (
            (config.include_state_separation or state_explanatory is not None)
            and "State_Explained_Variance_Eta_Squared" in table.columns
            and "State_Explained_Variance_Eta_Squared" not in config.excluded_objectives
        ):
            objective_columns.append("State_Explained_Variance_Eta_Squared")

    # Base eligibility check
    base_eligible = table["Resolution"].eq("frame")
    if salience_col in objective_columns:
        base_eligible = base_eligible & table[salience_col].notna()
    if "Representation_Stability" in objective_columns and "LOTO_Folds" in table.columns:
        if table["LOTO_Folds"].max() >= config.min_loto_folds:
            base_eligible = base_eligible & table["LOTO_Folds"].ge(config.min_loto_folds)
        else:
            notes.append("Single trajectory available; evaluated Pareto candidates without cross-replica LOTO thresholding.")
    base_eligible = base_eligible & table[objective_columns].notna().all(axis=1)

    # Cost objective
    cost_available = (
        bool(base_eligible.any())
        and "Measured_Cost_Seconds" in table.columns
        and bool(table.loc[base_eligible, "Measured_Cost_Seconds"].notna().all())
    )
    if "Cost_Efficiency" in config.excluded_objectives:
        table["Cost_Objective_Status"] = "excluded_by_configuration"
    elif cost_available:
        table["Cost_Efficiency"] = -table["Measured_Cost_Seconds"]
        if config.custom_objectives is None or "Cost_Efficiency" in config.custom_objectives:
            if "Cost_Efficiency" not in objective_columns:
                objective_columns.append("Cost_Efficiency")
        table["Cost_Objective_Status"] = "included_measured_cost"
    else:
        table["Cost_Objective_Status"] = "excluded_missing_measurements"
        notes.append("Cost was excluded because complete measured feature costs were not supplied.")

    table["SAM_MFP_Eligible"] = base_eligible & table[objective_columns].notna().all(axis=1)
    table["SAM_MFP_Pareto_Front"] = False
    table["SAM_MFP_Dominator_Count"] = np.nan
    eligible = table["SAM_MFP_Eligible"]
    if eligible.any():
        front, counts = _pareto_diagnostics(table.loc[eligible, objective_columns].to_numpy(dtype=float))
        table.loc[eligible, "SAM_MFP_Pareto_Front"] = front
        table.loc[eligible, "SAM_MFP_Dominator_Count"] = counts
    table["SAM_MFP_Objectives"] = ",".join(objective_columns)

    sort_cols = ["SAM_MFP_Pareto_Front"]
    ascending = [False]
    if salience_col in table.columns:
        sort_cols.append(salience_col)
        ascending.append(False)
    sort_cols.append("Feature")
    ascending.append(True)

    return table.sort_values(sort_cols, ascending=ascending).reset_index(drop=True), notes
