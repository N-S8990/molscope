"""Frame-level identity metadata used by every later analysis stage."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import MDAnalysis as mda
import pandas as pd


@dataclass(frozen=True)
class TrajectoryInput:
    """A trajectory with user-provided scientific identity, not ML labels."""

    topology: str | Path
    trajectory: str | Path
    trajectory_id: str
    replica_id: str
    ensemble_id: str
    system_id: str = "unspecified"
    metadata: dict = field(default_factory=dict)


def build_frame_manifest(
    trajectories: Iterable[TrajectoryInput],
    *,
    frame_stride: int = 1,
    max_frames: int | None = None,
) -> pd.DataFrame:
    """Return one identity record per selected frame.

    ``ensemble_id`` may describe a temperature, mutation, ligand state, or a
    user-defined collection.  It is metadata, not a required prediction label.
    """
    if frame_stride < 1:
        raise ValueError("frame_stride must be positive.")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be positive when provided.")

    records = []
    for item in trajectories:
        universe = mda.Universe(str(item.topology), str(item.trajectory))
        selected = list(range(0, len(universe.trajectory), frame_stride))
        if max_frames is not None:
            selected = selected[:max_frames]
        for output_index, frame_index in enumerate(selected):
            timestep = universe.trajectory[frame_index]
            records.append({
                "trajectory_id": item.trajectory_id,
                "replica_id": item.replica_id,
                "ensemble_id": item.ensemble_id,
                "system_id": item.system_id,
                "metadata_json": json.dumps(item.metadata, sort_keys=True),
                "frame_index": frame_index,
                "frame_order": output_index,
                "time_ps": float(timestep.time),
                "topology_path": str(item.topology),
                "trajectory_path": str(item.trajectory),
            })
    return pd.DataFrame.from_records(records)
