"""Generic, explicit project specification for system-independent workflows."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from molscope.core.identity import TrajectoryInput, build_frame_manifest


SCHEMA_VERSION = "1"


@dataclass(frozen=True)
class PredictionTask:
    """An optional ML task with an explicit trajectory-level target."""

    task_id: str
    task_type: str
    target_metadata_key: str | None = None
    target_label_id: str | None = None
    description: str = ""


@dataclass(frozen=True)
class LabelSpec:
    """Researcher-supplied trajectory labels with declared provenance.

    Labels are intentionally scalar and trajectory-level. Frame labels and
    labels derived from the same state-discovery run cannot enter predictive
    modelling through this project contract, preventing a common leakage path.
    """

    label_id: str
    label_type: str
    level: str
    source: str
    assignments: dict[str, str | int | float | bool]
    description: str = ""


@dataclass(frozen=True)
class AnalysisSpec:
    """Reproducible analysis choices shared by all project trajectories."""

    generic_feature_parameters: dict[str, Any] = field(default_factory=dict)
    plugins: tuple[str, ...] = ()
    plugin_parameters: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class OpenMMSimulationSpec:
    """A conservative, declarative OpenMM production protocol.

    This records enough information to generate a reproducible execution plan;
    it does not imply that a force field is valid for a particular system.
    """

    forcefield_files: tuple[str, ...] = ("amber14-all.xml", "implicit/gbn2.xml")
    nonbonded_method: str = "NoCutoff"
    constraints: str = "HBonds"
    temperature_kelvin: float = 300.0
    friction_per_picosecond: float = 1.0
    timestep_femtoseconds: float = 1.0
    minimization_max_iterations: int = 0
    equilibration_steps: int = 1000
    production_steps: int = 200000
    report_interval_steps: int = 1000
    platform: str | None = None
    seed: int | None = None
    ph: float = 7.0
    keep_heterogens: bool = False


@dataclass(frozen=True)
class ReplicaSpec:
    replica_id: str
    trajectory: Path
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EnsembleSpec:
    ensemble_id: str
    replicas: tuple[ReplicaSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SystemSpec:
    system_id: str
    topology: Path
    ensembles: tuple[EnsembleSpec, ...]
    metadata: dict[str, Any] = field(default_factory=dict)
    simulation_protocol: OpenMMSimulationSpec | None = None


@dataclass(frozen=True)
class ProjectSpec:
    project_id: str
    systems: tuple[SystemSpec, ...]
    labels: tuple[LabelSpec, ...] = ()
    prediction_tasks: tuple[PredictionTask, ...] = ()
    analysis: AnalysisSpec = AnalysisSpec()
    schema_version: str = SCHEMA_VERSION
    source_path: Path | None = None

    def trajectory_inputs(self) -> tuple[TrajectoryInput, ...]:
        """Build trajectory identities with declared labels kept distinct from metadata."""
        items = []
        for system in self.systems:
            for ensemble in system.ensembles:
                for replica in ensemble.replicas:
                    trajectory_id = f"{system.system_id}:{ensemble.ensemble_id}:{replica.replica_id}"
                    metadata = _trajectory_metadata(system, ensemble, replica)
                    for label in self.labels:
                        if trajectory_id in label.assignments:
                            metadata[f"label_{label.label_id}"] = label.assignments[trajectory_id]
                    items.append(TrajectoryInput(
                        topology=system.topology,
                        trajectory=replica.trajectory,
                        trajectory_id=trajectory_id,
                        replica_id=replica.replica_id,
                        ensemble_id=f"{system.system_id}:{ensemble.ensemble_id}",
                        system_id=system.system_id,
                        metadata=metadata,
                    ))
        return tuple(items)

    def validate_paths(self) -> list[str]:
        """Return missing input paths without changing any project state."""
        missing = []
        for system in self.systems:
            if not system.topology.is_file():
                missing.append(str(system.topology))
            for ensemble in system.ensembles:
                for replica in ensemble.replicas:
                    if not replica.trajectory.is_file():
                        missing.append(str(replica.trajectory))
        return missing


def _trajectory_metadata(
    system: SystemSpec,
    ensemble: EnsembleSpec,
    replica: ReplicaSpec,
) -> dict[str, Any]:
    """Flatten scalar provenance for frame manifests without inventing targets.

    The unprefixed form makes a declared ``target_metadata_key`` usable by the
    project runner. Namespaced duplicates retain provenance and prevent a
    similarly named key at another hierarchy level from being silently lost.
    Complex metadata remain in the project specification rather than being
    coerced into a frame-level ML column.
    """
    flattened: dict[str, Any] = {}
    for scope, source in (
        ("system", system.metadata),
        ("ensemble", ensemble.metadata),
        ("replica", replica.metadata),
    ):
        for key, value in source.items():
            if not isinstance(value, (str, int, float, bool)) and value is not None:
                continue
            flattened[f"{scope}_{key}"] = value
            if key not in flattened:
                flattened[key] = value
            elif flattened[key] != value:
                raise ValueError(
                    f"Metadata key {key!r} has conflicting scalar values across "
                    f"system/ensemble/replica scope for trajectory "
                    f"{system.system_id}:{ensemble.ensemble_id}:{replica.replica_id}. "
                    "Use distinct metadata keys for an unambiguous ML target."
                )
    return flattened


def _required_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string.")
    return value


def _optional_string(payload: dict[str, Any], key: str, context: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context}.{key} must be a non-empty string when supplied.")
    return value


def _metadata(payload: dict[str, Any], context: str) -> dict[str, Any]:
    value = payload.get("metadata", {})
    if not isinstance(value, dict):
        raise ValueError(f"{context}.metadata must be an object when supplied.")
    return value


def _resolve(path_value: str, base: Path) -> Path:
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()


def _number(payload: dict[str, Any], key: str, context: str, default: float, *, positive: bool = True) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{context}.{key} must be a finite number.")
    value = float(value)
    if (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context}.{key} must be {qualifier}.")
    return value


def _integer(payload: dict[str, Any], key: str, context: str, default: int, *, positive: bool = False) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or (positive and value <= 0) or (not positive and value < 0):
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{context}.{key} must be a {qualifier} integer.")
    return value


def _openmm_protocol(payload: dict[str, Any], context: str) -> OpenMMSimulationSpec:
    payload = dict(payload)
    timestep = _number(payload, "timestep_femtoseconds", context, 1.0)
    for physical_key, step_key, scale in (
        ("production_duration_ns", "production_steps", 1_000_000.0),
        ("save_interval_ps", "report_interval_steps", 1_000.0),
    ):
        if physical_key in payload:
            if step_key in payload:
                raise ValueError(f"{context}: declare {physical_key} or {step_key}, not both.")
            value = _number(payload, physical_key, context, 1.0)
            steps = value * scale / timestep
            if not math.isfinite(steps) or not math.isclose(steps, round(steps), rel_tol=0, abs_tol=1e-6) or steps < 1:
                raise ValueError(f"{context}.{physical_key} must be an integer number of timesteps.")
            payload[step_key] = int(round(steps))
    engine = payload.get("engine", "openmm")
    if engine != "openmm":
        raise ValueError(f"{context}.engine must be 'openmm' for configurable simulation generation.")
    raw_forcefields = payload.get("forcefield_files", ["amber14-all.xml", "implicit/gbn2.xml"])
    if not isinstance(raw_forcefields, list) or not raw_forcefields or not all(isinstance(item, str) and item.strip() for item in raw_forcefields):
        raise ValueError(f"{context}.forcefield_files must be a non-empty list of non-empty strings.")
    nonbonded_method = payload.get("nonbonded_method", "NoCutoff")
    if nonbonded_method not in {"NoCutoff", "CutoffNonPeriodic", "CutoffPeriodic", "PME", "Ewald", "LJPME"}:
        raise ValueError(f"{context}.nonbonded_method is not a supported OpenMM method.")
    constraints = payload.get("constraints", "HBonds")
    if constraints not in {"None", "HBonds", "AllBonds", "HAngles"}:
        raise ValueError(f"{context}.constraints must be one of None, HBonds, AllBonds, or HAngles.")
    platform = payload.get("platform")
    if platform is not None and (not isinstance(platform, str) or not platform.strip()):
        raise ValueError(f"{context}.platform must be a non-empty string when supplied.")
    seed = payload.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise ValueError(f"{context}.seed must be an integer when supplied.")
    keep_heterogens = payload.get("keep_heterogens", False)
    if not isinstance(keep_heterogens, bool):
        raise ValueError(f"{context}.keep_heterogens must be boolean when supplied.")
    return OpenMMSimulationSpec(
        forcefield_files=tuple(raw_forcefields),
        nonbonded_method=nonbonded_method,
        constraints=constraints,
        temperature_kelvin=_number(payload, "temperature_kelvin", context, 300.0),
        friction_per_picosecond=_number(payload, "friction_per_picosecond", context, 1.0),
        timestep_femtoseconds=_number(payload, "timestep_femtoseconds", context, 1.0),
        minimization_max_iterations=_integer(payload, "minimization_max_iterations", context, 0),
        equilibration_steps=_integer(payload, "equilibration_steps", context, 1000),
        production_steps=_integer(payload, "production_steps", context, 200000, positive=True),
        report_interval_steps=_integer(payload, "report_interval_steps", context, 1000, positive=True),
        platform=platform,
        seed=seed,
        ph=_number(payload, "ph", context, 7.0),
        keep_heterogens=keep_heterogens,
    )


def load_project_spec(path: str | Path) -> ProjectSpec:
    """Load and validate the simple JSON project manifest.

    JSON is used initially to avoid adding a YAML dependency.  Paths are resolved
    relative to the project-spec file, not the caller's current directory.
    """
    source = Path(path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Project manifest root must be a JSON object.")
    version = str(payload.get("schema_version", SCHEMA_VERSION))
    if version != SCHEMA_VERSION:
        raise ValueError(f"Unsupported project manifest schema_version {version!r}.")
    project_id = _required_string(payload, "project_id", "project")
    raw_systems = payload.get("systems")
    if not isinstance(raw_systems, list) or not raw_systems:
        raise ValueError("project.systems must be a non-empty array.")

    systems = []
    system_ids = set()
    trajectory_ids = set()
    for system_index, raw_system in enumerate(raw_systems):
        context = f"systems[{system_index}]"
        if not isinstance(raw_system, dict):
            raise ValueError(f"{context} must be an object.")
        system_id = _required_string(raw_system, "system_id", context)
        if system_id in system_ids:
            raise ValueError(f"Duplicate system_id: {system_id!r}.")
        system_ids.add(system_id)
        topology = _resolve(_required_string(raw_system, "topology", context), source.parent)
        raw_ensembles = raw_system.get("ensembles")
        if not isinstance(raw_ensembles, list) or not raw_ensembles:
            raise ValueError(f"{context}.ensembles must be a non-empty array.")
        ensembles = []
        ensemble_ids = set()
        for ensemble_index, raw_ensemble in enumerate(raw_ensembles):
            ensemble_context = f"{context}.ensembles[{ensemble_index}]"
            if not isinstance(raw_ensemble, dict):
                raise ValueError(f"{ensemble_context} must be an object.")
            ensemble_id = _required_string(raw_ensemble, "ensemble_id", ensemble_context)
            if ensemble_id in ensemble_ids:
                raise ValueError(f"Duplicate ensemble_id within {system_id!r}: {ensemble_id!r}.")
            ensemble_ids.add(ensemble_id)
            raw_replicas = raw_ensemble.get("replicas")
            if not isinstance(raw_replicas, list) or not raw_replicas:
                raise ValueError(f"{ensemble_context}.replicas must be a non-empty array.")
            replicas = []
            replica_ids = set()
            for replica_index, raw_replica in enumerate(raw_replicas):
                replica_context = f"{ensemble_context}.replicas[{replica_index}]"
                if not isinstance(raw_replica, dict):
                    raise ValueError(f"{replica_context} must be an object.")
                replica_id = _required_string(raw_replica, "replica_id", replica_context)
                if replica_id in replica_ids:
                    raise ValueError(f"Duplicate replica_id within {ensemble_id!r}: {replica_id!r}.")
                replica_ids.add(replica_id)
                trajectory_id = f"{system_id}:{ensemble_id}:{replica_id}"
                if trajectory_id in trajectory_ids:
                    raise ValueError(f"Duplicate trajectory identity: {trajectory_id!r}.")
                trajectory_ids.add(trajectory_id)
                replicas.append(ReplicaSpec(
                    replica_id=replica_id,
                    trajectory=_resolve(_required_string(raw_replica, "trajectory", replica_context), source.parent),
                    metadata=_metadata(raw_replica, replica_context),
                ))
            ensembles.append(EnsembleSpec(
                ensemble_id=ensemble_id,
                replicas=tuple(replicas),
                metadata=_metadata(raw_ensemble, ensemble_context),
            ))
        protocol = raw_system.get("simulation_protocol")
        if protocol is not None and not isinstance(protocol, dict):
            raise ValueError(f"{context}.simulation_protocol must be an object when supplied.")
        systems.append(SystemSpec(
            system_id=system_id,
            topology=topology,
            ensembles=tuple(ensembles),
            metadata=_metadata(raw_system, context),
            simulation_protocol=_openmm_protocol(protocol, f"{context}.simulation_protocol") if protocol is not None else None,
        ))

    labels = []
    label_ids = set()
    for label_index, raw_label in enumerate(payload.get("labels", [])):
        context = f"labels[{label_index}]"
        if not isinstance(raw_label, dict):
            raise ValueError(f"{context} must be an object.")
        label_id = _required_string(raw_label, "label_id", context)
        if label_id in label_ids:
            raise ValueError(f"Duplicate label_id: {label_id!r}.")
        label_ids.add(label_id)
        label_type = _required_string(raw_label, "label_type", context)
        if label_type not in {"categorical", "continuous"}:
            raise ValueError(f"{context}.label_type must be categorical or continuous.")
        level = _required_string(raw_label, "level", context)
        if level != "trajectory":
            raise ValueError(
                f"{context}.level must be 'trajectory'; frame-level labels are not accepted "
                "because they can leak temporal information into grouped ML."
            )
        source_kind = _required_string(raw_label, "source", context)
        if source_kind == "derived_state":
            raise ValueError(
                f"{context}.source must not be 'derived_state'. Candidate state labels are "
                "reported separately and cannot be used as project predictive targets."
            )
        if source_kind not in {"researcher_provided", "external_annotation"}:
            raise ValueError(f"{context}.source must be researcher_provided or external_annotation.")
        assignments = raw_label.get("assignments")
        if not isinstance(assignments, dict) or not assignments:
            raise ValueError(f"{context}.assignments must be a non-empty object keyed by trajectory_id.")
        unknown_trajectories = sorted(set(assignments) - trajectory_ids)
        if unknown_trajectories:
            raise ValueError(f"{context}.assignments references unknown trajectory IDs: {unknown_trajectories[:4]}")
        parsed_assignments: dict[str, str | int | float | bool] = {}
        for trajectory_id, value in assignments.items():
            if not isinstance(trajectory_id, str):
                raise ValueError(f"{context}.assignments keys must be trajectory IDs.")
            if isinstance(value, bool) or isinstance(value, str) or isinstance(value, int):
                parsed_assignments[trajectory_id] = value
            elif isinstance(value, float) and math.isfinite(value):
                parsed_assignments[trajectory_id] = value
            else:
                raise ValueError(f"{context}.assignments[{trajectory_id!r}] must be a finite scalar value.")
        labels.append(LabelSpec(
            label_id=label_id,
            label_type=label_type,
            level=level,
            source=source_kind,
            assignments=parsed_assignments,
            description=str(raw_label.get("description", "")),
        ))

    tasks = []
    task_ids = set()
    for task_index, raw_task in enumerate(payload.get("prediction_tasks", [])):
        context = f"prediction_tasks[{task_index}]"
        if not isinstance(raw_task, dict):
            raise ValueError(f"{context} must be an object.")
        task_id = _required_string(raw_task, "task_id", context)
        if task_id in task_ids:
            raise ValueError(f"Duplicate prediction task_id: {task_id!r}.")
        task_ids.add(task_id)
        target_metadata_key = _optional_string(raw_task, "target_metadata_key", context)
        target_label_id = _optional_string(raw_task, "target_label_id", context)
        if (target_metadata_key is None) == (target_label_id is None):
            raise ValueError(f"{context} must declare exactly one of target_label_id or target_metadata_key.")
        if target_label_id is not None:
            label = next((item for item in labels if item.label_id == target_label_id), None)
            if label is None:
                raise ValueError(f"{context}.target_label_id references unknown label {target_label_id!r}.")
            if label.label_type != "categorical":
                raise ValueError(f"{context}.target_label_id must reference a categorical label; regression is not implemented.")
            missing_assignments = sorted(trajectory_ids - set(label.assignments))
            if missing_assignments:
                raise ValueError(
                    f"{context}.target_label_id requires a value for every trajectory; missing {missing_assignments[:4]}"
                )
        tasks.append(PredictionTask(
            task_id=task_id,
            task_type=_required_string(raw_task, "task_type", context),
            target_metadata_key=target_metadata_key,
            target_label_id=target_label_id,
            description=str(raw_task.get("description", "")),
        ))
    raw_analysis = payload.get("analysis", {})
    if not isinstance(raw_analysis, dict):
        raise ValueError("project.analysis must be an object when supplied.")
    generic_parameters = raw_analysis.get("generic_feature_parameters", {})
    if not isinstance(generic_parameters, dict):
        raise ValueError("project.analysis.generic_feature_parameters must be an object when supplied.")
    raw_plugins = raw_analysis.get("plugins", [])
    if not isinstance(raw_plugins, list) or not all(isinstance(item, str) and item.strip() for item in raw_plugins):
        raise ValueError("project.analysis.plugins must be a list of non-empty plugin IDs.")
    plugins = tuple(raw_plugins)
    if len(set(plugins)) != len(plugins):
        raise ValueError("project.analysis.plugins must not contain duplicate plugin IDs.")
    plugin_parameters = raw_analysis.get("plugin_parameters", {})
    if not isinstance(plugin_parameters, dict) or not all(isinstance(value, dict) for value in plugin_parameters.values()):
        raise ValueError("analysis.plugin_parameters must map plugin IDs to parameter objects.")
    if set(plugin_parameters) - set(plugins):
        raise ValueError("analysis.plugin_parameters may only configure enabled plugins.")
    from molscope.features.plugins import get_plugin
    for plugin_id in plugins:
        get_plugin(plugin_id, plugin_parameters.get(plugin_id))
    analysis = AnalysisSpec(generic_feature_parameters=generic_parameters, plugins=plugins, plugin_parameters=plugin_parameters)
    return ProjectSpec(project_id=project_id, systems=tuple(systems), labels=tuple(labels), prediction_tasks=tuple(tasks), analysis=analysis, source_path=source)


def project_summary(project: ProjectSpec) -> dict[str, Any]:
    return {
        "project_id": project.project_id,
        "systems": len(project.systems),
        "trajectories": len(project.trajectory_inputs()),
        "labels": [asdict(label) for label in project.labels],
        "prediction_tasks": [asdict(task) for task in project.prediction_tasks],
        "analysis": asdict(project.analysis),
        "missing_paths": project.validate_paths(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a generic MolScope JSON project specification.")
    parser.add_argument("--project", required=True, help="Path to a project JSON file.")
    parser.add_argument("--write-manifest", default=None, help="Optional output CSV for frame/replica identity metadata.")
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    args = parser.parse_args()

    project = load_project_spec(args.project)
    summary = project_summary(project)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["missing_paths"]:
        raise SystemExit("Project manifest is valid, but input files are missing.")
    if args.write_manifest:
        frame_manifest = build_frame_manifest(project.trajectory_inputs(), frame_stride=args.frame_stride, max_frames=args.max_frames)
        frame_manifest.to_csv(args.write_manifest, index=False)
        print(f"Wrote {len(frame_manifest)} frame records to {args.write_manifest}")


if __name__ == "__main__":
    main()
