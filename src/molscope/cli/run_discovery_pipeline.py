"""CLI for the target-free MolScope prototype pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molscope.execution.discovery import DiscoveryPipelineConfig, run_target_free_discovery_pipeline
from molscope.reporting.research import render_report_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Run feature quality, ML discovery, SAM-MFP, state discovery, and reporting.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--feature-costs")
    parser.add_argument("--target-column", help="Optional valid trajectory-level target for leakage-aware predictive ML.")
    parser.add_argument("--supervised-folds", type=int, default=3)
    args = parser.parse_args()
    costs = pd.read_csv(args.feature_costs) if args.feature_costs else None
    outputs = run_target_free_discovery_pipeline(
        pd.read_csv(args.features), pd.read_csv(args.manifest),
        DiscoveryPipelineConfig(pca_components=args.components, state_count=args.states, supervised_splits=args.supervised_folds),
        feature_costs=costs, target_column=args.target_column,
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in outputs.items():
        if isinstance(value, pd.DataFrame):
            value.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "research_report.md").write_text(str(outputs["research_report_markdown"]) + "\n", encoding="utf-8")
    (out_dir / "research_report.json").write_text(json.dumps(outputs["research_report_summary"], indent=2) + "\n", encoding="utf-8")
    render_report_figures(
        outputs["sam_mfp_candidates"], outputs["state_populations"], out_dir,
        quality=outputs["feature_quality_metrics"], representation_salience=outputs["representation_feature_salience"],
    )
    print(f"Wrote target-free discovery pipeline outputs to {out_dir}")


if __name__ == "__main__":
    main()
