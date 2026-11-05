"""System-Aware Multi-Criteria Molecular Feature Prioritization (SAM-MFP)."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from molscope.analysis.quality.assessment import _pareto_diagnostics


@dataclass(frozen=True)
class SAMMFPConfig:
    """Eligibility constraints for an unsupervised SAM-MFP assessment."""

    min_loto_folds: int = 2

    def __post_init__(self) -> None:
        if self.min_loto_folds < 2:
            raise ValueError("min_loto_folds must be at least 2.")

    def to_dict(self) -> dict:
        return asdict(self)


_QUALITY_OBJECTIVES = ("ESS_Ratio", "Replica_Consistency", "Occupancy_Entropy", "Uniqueness")
_REQUIRED_QUALITY_COLUMNS = {"Feature", "Resolution", "Valid_Fraction", *_QUALITY_OBJECTIVES}
_REQUIRED_DISCOVERY_COLUMNS = {"Feature", "PCA_Salience", "LOTO_Rank_Std_Normalized", "LOTO_Folds"}


def prioritize_sam_mfp(
    quality: pd.DataFrame,
    representation_salience: pd.DataFrame,
    config: SAMMFPConfig = SAMMFPConfig(),
) -> tuple[pd.DataFrame, list[str]]:
    """Create target-free multi-criteria Pareto feature candidates.

    The output is deliberately not a scalar importance score. It combines
    measurement-quality and representation-discovery evidence; supervised
    predictive importance belongs to a later, target-specific analysis.
    """
    missing_quality = _REQUIRED_QUALITY_COLUMNS - set(quality.columns)
    missing_discovery = _REQUIRED_DISCOVERY_COLUMNS - set(representation_salience.columns)
    if missing_quality:
        raise ValueError(f"Quality table is missing required columns: {sorted(missing_quality)}")
    if missing_discovery:
        raise ValueError(f"Representation table is missing required columns: {sorted(missing_discovery)}")
    if quality["Feature"].duplicated().any() or representation_salience["Feature"].duplicated().any():
        raise ValueError("Feature names must be unique in both SAM-MFP inputs.")

    table = quality.merge(representation_salience, on="Feature", how="left", validate="one_to_one")
    objective_columns = [*_QUALITY_OBJECTIVES, "PCA_Salience"]
    table["Representation_Stability"] = (1.0 - table["LOTO_Rank_Std_Normalized"]).clip(lower=0.0, upper=1.0)
    objective_columns.append("Representation_Stability")
    notes = [
        "SAM-MFP candidates are Pareto trade-offs, not a scalar importance ranking.",
        "PCA salience is target-free representation evidence, not causal or predictive importance.",
    ]
    base_eligible = (
        table["Resolution"].eq("frame")
        & table["PCA_Salience"].notna()
        & table["LOTO_Folds"].ge(config.min_loto_folds)
        & table[objective_columns].notna().all(axis=1)
    )
    cost_available = "Measured_Cost_Seconds" in table.columns and table.loc[base_eligible, "Measured_Cost_Seconds"].notna().all()
    table["Cost_Objective_Status"] = "excluded_missing_measurements"
    if cost_available:
        table["Cost_Efficiency"] = -table["Measured_Cost_Seconds"]
        objective_columns.append("Cost_Efficiency")
        table["Cost_Objective_Status"] = "included_measured_cost"
    else:
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
    return table.sort_values(["SAM_MFP_Pareto_Front", "PCA_Salience", "Feature"], ascending=[False, False, True]).reset_index(drop=True), notes
