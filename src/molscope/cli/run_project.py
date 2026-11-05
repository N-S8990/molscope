"""Run a reproducible MolScope project declaration end to end."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from molscope.execution.discovery import DiscoveryPipelineConfig
from molscope.execution.pipeline import MDContactMLPipeline
from molscope.core.project import load_project_spec


def _target_column(project, task_id: str | None) -> str | None:
    """Resolve an explicitly declared predictive task to a manifest column."""
    if task_id is None:
        return None
    tasks = {task.task_id: task for task in project.prediction_tasks}
    if task_id not in tasks:
        raise ValueError(f"Unknown prediction task {task_id!r}; declared tasks: {sorted(tasks)}")
    task = tasks[task_id]
    if task.target_label_id is not None:
        return f"metadata_label_{task.target_label_id}"
    return f"metadata_{task.target_metadata_key}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run configured extraction, target-free ML discovery, SAM-MFP, state discovery, "
            "and optionally leakage-aware predictive ML from an MolScope project JSON."
        )
    )
    parser.add_argument("--project", required=True, help="Validated MolScope project JSON.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--supervised-folds", type=int, default=3)
    parser.add_argument("--feature-costs", help="Optional measured feature-cost CSV.")
    parser.add_argument("--prediction-task", help="Declared task ID; enables grouped predictive ML.")
    parser.add_argument("--extract-only", action="store_true", help="Validate, extract, and write an extraction report, including single-trajectory studies; skip ML discovery.")
    args = parser.parse_args()

    project = load_project_spec(args.project)
    if args.extract_only and args.prediction_task:
        parser.error("--extract-only cannot be combined with --prediction-task")
    target_column = _target_column(project, args.prediction_task)
    costs = pd.read_csv(args.feature_costs) if args.feature_costs else None
    pipeline = MDContactMLPipeline(
        DiscoveryPipelineConfig(
            pca_components=args.components,
            state_count=args.states,
            supervised_splits=args.supervised_folds,
        )
    ).load_project(project)
    if args.extract_only:
        destination = pipeline.save_session(args.out_dir)
        from molscope.reporting.extraction import write_extraction_report
        write_extraction_report(destination)
        (destination / "label_definitions.json").write_text(
            json.dumps([asdict(label) for label in project.labels], indent=2) + "\n", encoding="utf-8",
        )
        print(f"Validated and extracted {len(pipeline.manifest)} frames to {destination}; ML was not requested.")
        return
    if target_column is not None and target_column not in pipeline.manifest.columns:
        raise ValueError(
            f"Prediction target {target_column!r} is absent from the trajectory manifest. "
            "Declare a complete trajectory-level label or add scalar project metadata."
        )
    destination = pipeline.run(target_column=target_column, feature_costs=costs).write_results(args.out_dir)
    (destination / "label_definitions.json").write_text(
        json.dumps([asdict(label) for label in project.labels], indent=2) + "\n", encoding="utf-8",
    )
    print(f"Wrote MolScope project results to {destination}")


if __name__ == "__main__":
    main()
