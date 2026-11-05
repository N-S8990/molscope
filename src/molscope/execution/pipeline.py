"""High-level, explicit workflow interface for MolScope."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from molscope.execution.discovery import DiscoveryPipelineConfig, run_target_free_discovery_pipeline
from molscope.core.contracts import ApplicabilityStatus
from molscope.features.energetics import align_energy_to_frames
from molscope.features.geometry.generic import GenericFeatureConfig, GenericFeatureExtractor
from molscope.core.identity import TrajectoryInput
from molscope.io.validation import inspect_md_inputs, InputIntegrityStatus
from molscope.profiling.system import profile_system
from molscope.features.plugins import get_plugin
from molscope.core.project import ProjectSpec, load_project_spec
from molscope.reporting.research import render_report_figures
from molscope.io.storage import FeatureCache, extraction_cache_key, read_table, write_table


class MDContactMLPipeline:
    """Stateful convenience API for the current feature-matrix workflow.

    It accepts either an extracted feature matrix or a homogeneous collection
    of trajectories for generic feature extraction.
    """

    def __init__(self, config: DiscoveryPipelineConfig = DiscoveryPipelineConfig()):
        self.config = config
        self.features: pd.DataFrame | None = None
        self.manifest: pd.DataFrame | None = None
        self.outputs: dict[str, pd.DataFrame | str | dict] | None = None
        self.extraction_evidence: list[dict] = []

    def load_feature_matrix(self, features: pd.DataFrame, manifest: pd.DataFrame) -> "MDContactMLPipeline":
        self.features = features.copy()
        self.manifest = manifest.copy()
        self.outputs = None
        self.extraction_evidence = []
        return self

    def load_trajectories(
        self,
        trajectories: Iterable[TrajectoryInput],
        config: GenericFeatureConfig = GenericFeatureConfig(),
        *,
        plugins: Iterable[str] = (),
        plugin_parameters: dict[str, dict] | None = None,
        cache_dir: str | Path | None = None,
    ) -> "MDContactMLPipeline":
        """Extract generic frame features and preserve supplied trajectory identity.

        All inputs must yield the same feature columns. Heterogeneous systems
        are intentionally rejected here: treating inapplicable quantities as
        missing values in a common ML matrix needs a separate, validated policy.
        """
        inputs = tuple(trajectories)
        if not inputs:
            raise ValueError("At least one trajectory input is required.")
        extractor = GenericFeatureExtractor(config)
        plugin_parameters = plugin_parameters or {}
        plugin_ids = tuple(plugins)
        unknown_parameters = set(plugin_parameters) - set(plugin_ids)
        if unknown_parameters:
            raise ValueError(f"Parameters supplied for disabled plugins: {sorted(unknown_parameters)}")
        plugin_instances = tuple(get_plugin(plugin_id, plugin_parameters.get(plugin_id)) for plugin_id in plugin_ids)
        cache = FeatureCache(cache_dir) if cache_dir is not None else None
        plugin_versions = {plugin.plugin_id: plugin.version for plugin in plugin_instances}
        if plugin_parameters:
            plugin_versions = {plugin.plugin_id: {"version": plugin.version, "parameters": plugin_parameters.get(plugin.plugin_id, {})} for plugin in plugin_instances}
        feature_tables, manifest_rows = [], []
        expected_columns: tuple[str, ...] | None = None
        metadata_columns: set[str] = set()
        evidence_records = []
        # Validate every input before performing any extraction or cache writes.
        preflight = []
        for item in inputs:
            integrity = inspect_md_inputs(item.topology, item.trajectory, include_checksum=True)
            if integrity.status is InputIntegrityStatus.INVALID:
                raise ValueError(f"Input preflight failed for {item.trajectory_id!r}: {integrity.errors}")
            preflight.append(integrity.to_dict())
        for item, integrity in zip(inputs, preflight, strict=True):
            cache_key = extraction_cache_key(
                item.topology, item.trajectory, generic_config=config, plugins=plugin_versions,
            ) if cache is not None else None
            frame_table = cache.get(cache_key) if cache is not None and cache_key is not None else None
            evidence = cache.get_evidence(cache_key) if cache is not None and cache_key is not None else None
            if evidence is None:
                frame_table = None
            if frame_table is None:
                result = extractor.extract(item.topology, item.trajectory)
                evidence = {
                    "evidence_version": 1,
                    "assessments": result.assessments_payload(),
                    "provenance": result.provenance_payload(),
                    "trajectory_features": json.loads(result.trajectory_features.to_json(orient="records")),
                    "system_profile": profile_system(item.topology, item.trajectory).to_dict(),
                }
                frame_table = result.frame_features.copy()
                for plugin in plugin_instances:
                    plugin_result = plugin.extract(item.topology, item.trajectory, max_frames=config.max_frames, frame_stride=config.frame_stride)
                    evidence["assessments"].append(plugin_result.assessment.to_dict())
                    if plugin_result.provenance is not None:
                        evidence["provenance"].append(plugin_result.provenance.to_dict())
                    evidence["trajectory_features"].extend(json.loads(plugin_result.trajectory_features.to_json(orient="records")))
                    if plugin_result.assessment.status is not ApplicabilityStatus.APPLICABLE:
                        continue
                    plugin_frames = plugin_result.frame_features
                    if not frame_table[["frame_index", "frame_order"]].equals(plugin_frames[["frame_index", "frame_order"]]):
                        raise ValueError(f"Plugin {plugin.plugin_id!r} frame identity does not match generic extraction.")
                    values = plugin_frames.drop(columns=["frame_index", "frame_order", "time_ps"], errors="ignore")
                    overlap = set(values.columns) & set(frame_table.columns)
                    if overlap:
                        raise ValueError(f"Plugin {plugin.plugin_id!r} produced duplicate feature columns: {sorted(overlap)[:4]}")
                    frame_table = pd.concat([frame_table, values], axis=1)
                if cache is not None and cache_key is not None:
                    cache.put(cache_key, frame_table)
                    cache.put_evidence(cache_key, evidence)
            evidence_records.append(json.loads(json.dumps({
                **evidence, "trajectory_id": item.trajectory_id, "system_id": item.system_id,
                "ensemble_id": item.ensemble_id, "replica_id": item.replica_id,
                "input_integrity": integrity,
            })))
            columns = tuple(column for column in frame_table.columns if column not in {"frame_index", "frame_order", "time_ps"})
            if expected_columns is None:
                expected_columns = columns
            elif columns != expected_columns:
                raise ValueError("Trajectory inputs produced different feature schemas; analyze heterogeneous systems separately for now.")
            feature_tables.append(frame_table.loc[:, list(columns)])
            for _, row in frame_table.iterrows():
                record = {
                    "trajectory_id": item.trajectory_id, "replica_id": item.replica_id, "ensemble_id": item.ensemble_id,
                    "system_id": item.system_id, "frame_index": int(row["frame_index"]), "frame_order": int(row["frame_order"]),
                    "time_ps": float(row["time_ps"]), "topology_path": str(item.topology), "trajectory_path": str(item.trajectory),
                }
                for key, value in item.metadata.items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        record[f"metadata_{key}"] = value
                        metadata_columns.add(f"metadata_{key}")
                manifest_rows.append(record)
        manifest = pd.DataFrame.from_records(manifest_rows)
        for column in metadata_columns:
            if column not in manifest:
                manifest[column] = None
        self.load_feature_matrix(pd.concat(feature_tables, ignore_index=True), manifest)
        self.extraction_evidence = evidence_records
        return self

    def load_project(
        self,
        project: ProjectSpec | str | Path,
        config: GenericFeatureConfig | None = None,
        *,
        plugins: Iterable[str] | None = None,
        cache_dir: str | Path | None = None,
    ) -> "MDContactMLPipeline":
        """Load all declared trajectories from a validated project specification."""
        specification = load_project_spec(project) if isinstance(project, (str, Path)) else project
        missing = specification.validate_paths()
        if missing:
            raise FileNotFoundError(f"Project specification references missing paths: {missing[:4]}")
        resolved_config = config or GenericFeatureConfig(**specification.analysis.generic_feature_parameters)
        resolved_plugins = tuple(plugins) if plugins is not None else specification.analysis.plugins
        return self.load_trajectories(
            specification.trajectory_inputs(), resolved_config, plugins=resolved_plugins, cache_dir=cache_dir,
            plugin_parameters={key: value for key, value in specification.analysis.plugin_parameters.items() if key in resolved_plugins},
        )

    def add_external_energies(self, energies: pd.DataFrame) -> "MDContactMLPipeline":
        """Attach exactly aligned engine-produced energy terms to frame features."""
        if self.features is None or self.manifest is None:
            raise RuntimeError("Load trajectories or a feature matrix before adding energy terms.")
        values = align_energy_to_frames(self.manifest, energies)
        overlap = set(values.columns) & set(self.features.columns)
        if overlap:
            raise ValueError(f"Energy columns duplicate existing feature names: {sorted(overlap)}")
        self.features = pd.concat([self.features.reset_index(drop=True), values.reset_index(drop=True)], axis=1)
        self.outputs = None
        return self

    def run(self, *, target_column: str | None = None, feature_costs: pd.DataFrame | None = None) -> "MDContactMLPipeline":
        if self.features is None or self.manifest is None:
            raise RuntimeError("Load a feature matrix and manifest before running the pipeline.")
        self.outputs = run_target_free_discovery_pipeline(
            self.features, self.manifest, self.config, feature_costs=feature_costs, target_column=target_column,
        )
        if self.extraction_evidence:
            self.outputs["extraction_evidence"] = {"trajectories": self.extraction_evidence}
        return self

    def table(self, name: str) -> pd.DataFrame:
        if self.outputs is None:
            raise RuntimeError("Run the pipeline before requesting results.")
        value = self.outputs[name]
        if not isinstance(value, pd.DataFrame):
            raise TypeError(f"Output {name!r} is not a table.")
        return value.copy()

    def write_results(self, output_dir: str | Path) -> Path:
        if self.outputs is None:
            raise RuntimeError("Run the pipeline before writing results.")
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        for name, value in self.outputs.items():
            if isinstance(value, pd.DataFrame):
                value.to_csv(destination / f"{name}.csv", index=False)
        if self.extraction_evidence:
            (destination / "extraction_evidence.json").write_text(json.dumps({"trajectories": self.extraction_evidence}, indent=2) + "\n", encoding="utf-8")
        (destination / "research_report.md").write_text(str(self.outputs["research_report_markdown"]) + "\n", encoding="utf-8")
        (destination / "research_report.json").write_text(json.dumps(self.outputs["research_report_summary"], indent=2) + "\n", encoding="utf-8")
        render_report_figures(
            self.table("sam_mfp_candidates"), self.table("state_populations"), destination,
            quality=self.table("feature_quality_metrics"),
            representation_salience=self.table("representation_feature_salience"),
            predictive_ranking=self.table("supervised_predictive_feature_ranking") if "supervised_predictive_feature_ranking" in self.outputs else None,
            state_explanatory_ranking=self.table("state_explanatory_feature_ranking"),
        )
        return destination

    def save_session(self, session_dir: str | Path) -> Path:
        """Persist the current analysis inputs, configuration, and outputs."""
        if self.features is None or self.manifest is None:
            raise RuntimeError("Load analysis inputs before saving a session.")
        destination = Path(session_dir)
        destination.mkdir(parents=True, exist_ok=True)
        feature_path = write_table(self.features, destination / "frame_features")
        manifest_path = write_table(self.manifest, destination / "frame_manifest")
        (destination / "session_storage.json").write_text(json.dumps({
            "schema_version": 1,
            "features": feature_path.name,
            "manifest": manifest_path.name,
        }, indent=2) + "\n", encoding="utf-8")
        (destination / "session_config.json").write_text(json.dumps(self.config.to_dict(), indent=2) + "\n", encoding="utf-8")
        (destination / "extraction_evidence.json").write_text(json.dumps({"trajectories": self.extraction_evidence}, indent=2) + "\n", encoding="utf-8")
        if self.outputs is not None:
            self.write_results(destination / "results")
        return destination

    @classmethod
    def load_session(cls, session_dir: str | Path) -> "MDContactMLPipeline":
        """Restore saved inputs and configuration; results can be rerun reproducibly."""
        source = Path(session_dir)
        config_path = source / "session_config.json"
        storage_path = source / "session_storage.json"
        if storage_path.is_file():
            storage = json.loads(storage_path.read_text(encoding="utf-8"))
            feature_path = source / storage["features"]
            manifest_path = source / storage["manifest"]
        else:  # compatibility with sessions created before persistent-table storage
            feature_path = source / "frame_features.csv"
            manifest_path = source / "frame_manifest.csv"
        missing = [str(path.name) for path in (config_path, feature_path, manifest_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Session is missing required files: {missing}")
        config = DiscoveryPipelineConfig(**json.loads(config_path.read_text(encoding="utf-8")))
        pipeline = cls(config).load_feature_matrix(read_table(feature_path), read_table(manifest_path))
        evidence_path = source / "extraction_evidence.json"
        if evidence_path.is_file():
            pipeline.extraction_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))["trajectories"]
        return pipeline
