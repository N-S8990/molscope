"""Optional, capability-gated domain feature plugins."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Protocol

import MDAnalysis as mda
from MDAnalysis.analysis.distances import capped_distance
from MDAnalysis.lib.distances import distance_array, calc_bonds
import networkx as nx
import numpy as np
import pandas as pd

from molscope.core.contracts import ApplicabilityStatus, FeatureAssessment, FeatureProvenance
from molscope.profiling.system import SystemProfile, discover_component_selections, profile_system


@dataclass(frozen=True)
class PluginResult:
    frame_features: pd.DataFrame
    trajectory_features: pd.DataFrame
    assessment: FeatureAssessment
    provenance: FeatureProvenance | None


class FeaturePlugin(Protocol):
    plugin_id: str
    version: str

    def assess(self, profile: SystemProfile) -> FeatureAssessment:
        ...

    def extract(self, topology: str | Path, trajectory: str | Path, *, max_frames: int | None = None, frame_stride: int = 1) -> PluginResult:
        ...


def _safe_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value).strip("_")
