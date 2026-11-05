"""External MD energy ingestion with explicit frame alignment."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_ENERGY_COLUMNS = frozenset({"trajectory_id", "time_ps"})


def validate_energy_table(energies: pd.DataFrame) -> pd.DataFrame:
    """Validate an externally produced energy table without altering values."""
    missing = REQUIRED_ENERGY_COLUMNS - set(energies.columns)
    if missing:
        raise ValueError(f"Energy table is missing required columns: {sorted(missing)}")
    if energies["trajectory_id"].isna().any() or energies["time_ps"].isna().any():
        raise ValueError("Energy trajectory_id and time_ps values must be present.")
    if energies.duplicated(["trajectory_id", "time_ps"]).any():
        raise ValueError("Energy table contains duplicate trajectory_id/time_ps records.")
    value_columns = [column for column in energies if column not in REQUIRED_ENERGY_COLUMNS]
    if not value_columns:
        raise ValueError("Energy table must contain at least one energy observable column.")
    numeric = energies[value_columns].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Energy observables must be finite numeric values; unavailable values must remain absent, not zero-filled.")
    return pd.concat([energies.loc[:, ["trajectory_id", "time_ps"]].reset_index(drop=True), numeric.reset_index(drop=True)], axis=1)


def load_energy_table(path: str | Path) -> pd.DataFrame:
    """Load a CSV energy table with explicit trajectory and simulation time."""
    return validate_energy_table(pd.read_csv(path))


def align_energy_to_frames(manifest: pd.DataFrame, energies: pd.DataFrame) -> pd.DataFrame:
    """Exactly align supplied energy terms to frame metadata.

    Exact matching is deliberate: nearest-time interpolation can create energy
    values for frames that were never written by the simulation engine. Users
    must resample externally before using a different cadence.
    """
    required_manifest = {"trajectory_id", "time_ps"}
    missing_manifest = required_manifest - set(manifest.columns)
    if missing_manifest:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_manifest)}")
    energy_table = validate_energy_table(energies)
    aligned = manifest.loc[:, ["trajectory_id", "time_ps"]].merge(
        energy_table, on=["trajectory_id", "time_ps"], how="left", validate="one_to_one",
    )
    value_columns = [column for column in energy_table if column not in REQUIRED_ENERGY_COLUMNS]
    if aligned[value_columns].isna().any().any():
        missing = int(aligned[value_columns].isna().any(axis=1).sum())
        raise ValueError(f"No exactly matching energy record exists for {missing} frame rows; values were not imputed.")
    return aligned.loc[:, value_columns]
