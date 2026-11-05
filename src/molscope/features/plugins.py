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

from molscope.features.base import PluginResult, FeaturePlugin, _safe_id
from molscope.features.protein.plugins import ProteinBackboneTorsionPlugin, ProteinCADistanceContactPlugin, ProteinDSSPPlugin, ProteinSaltBridgePlugin, ProteinHydrophobicContactPlugin, ProteinHydrogenBondPlugin, ProteinAromaticStackingPlugin, ProteinContactNetworkPlugin
from molscope.features.ligand.plugins import ProteinLigandContactPlugin
from molscope.features.environment.plugins import SASAPlugin, SolvationShellPlugin, MolecularDipolePlugin
from molscope.features.nucleic_acid.plugins import NucleicAcidBaseProximityPlugin

PLUGIN_REGISTRY: dict[str, FeaturePlugin] = {
    ProteinBackboneTorsionPlugin.plugin_id: ProteinBackboneTorsionPlugin(),
    ProteinCADistanceContactPlugin.plugin_id: ProteinCADistanceContactPlugin(),
    ProteinDSSPPlugin.plugin_id: ProteinDSSPPlugin(),
    ProteinLigandContactPlugin.plugin_id: ProteinLigandContactPlugin(),
    ProteinSaltBridgePlugin.plugin_id: ProteinSaltBridgePlugin(),
    ProteinHydrophobicContactPlugin.plugin_id: ProteinHydrophobicContactPlugin(),
    SASAPlugin.plugin_id: SASAPlugin(),
    ProteinHydrogenBondPlugin.plugin_id: ProteinHydrogenBondPlugin(),
    ProteinAromaticStackingPlugin.plugin_id: ProteinAromaticStackingPlugin(),
    SolvationShellPlugin.plugin_id: SolvationShellPlugin(),
    MolecularDipolePlugin.plugin_id: MolecularDipolePlugin(),
    ProteinContactNetworkPlugin.plugin_id: ProteinContactNetworkPlugin(),
    NucleicAcidBaseProximityPlugin.plugin_id: NucleicAcidBaseProximityPlugin(),
}


def get_plugin(plugin_id: str, parameters: dict | None = None) -> FeaturePlugin:
    try:
        prototype = PLUGIN_REGISTRY[plugin_id]
    except KeyError as exc:
        raise KeyError(f"Unknown plugin {plugin_id!r}. Available plugins: {sorted(PLUGIN_REGISTRY)}") from exc
    if parameters is not None and not isinstance(parameters, dict):
        raise ValueError("Plugin parameters must be an object.")
    try:
        return type(prototype)(**(parameters or {}))
    except TypeError as exc:
        raise ValueError(f"Unsupported parameters for plugin {plugin_id!r}: {parameters}") from exc
