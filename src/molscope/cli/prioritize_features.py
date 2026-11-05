"""CLI for the target-free SAM-MFP feature-prioritization workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molscope.analysis.discovery.representation import RepresentationDiscoveryConfig, discover_representation_features
from molscope.analysis.quality.assessment import PrioritizationConfig, assess_feature_quality
from molscope.analysis.prioritization.sam_mfp import SAMMFPConfig, prioritize_sam_mfp


def main() -> None:
    parser = argparse.ArgumentParser(description="Run target-free SAM-MFP on a feature matrix and frame manifest.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--feature-costs", help="Optional measured Feature,Elapsed_Seconds table.")
    args = parser.parse_args()
    features, manifest = pd.read_csv(args.features), pd.read_csv(args.manifest)
    costs = pd.read_csv(args.feature_costs) if args.feature_costs else None
    quality = assess_feature_quality(features, manifest, PrioritizationConfig(), feature_costs=costs)
    discovery = discover_representation_features(features, manifest, RepresentationDiscoveryConfig(args.components))
    prioritized, notes = prioritize_sam_mfp(quality, discovery["representation_feature_salience"], SAMMFPConfig())
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(out_dir / "feature_quality_metrics.csv", index=False)
    for name, table in discovery.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    prioritized.to_csv(out_dir / "sam_mfp_candidates.csv", index=False)
    prioritized[prioritized["SAM_MFP_Pareto_Front"]].to_csv(out_dir / "sam_mfp_pareto_candidates.csv", index=False)
    (out_dir / "sam_mfp_report.json").write_text(json.dumps({
        "configuration": {"pca_components": args.components},
        "eligible_features": int(prioritized["SAM_MFP_Eligible"].sum()),
        "pareto_candidates": int(prioritized["SAM_MFP_Pareto_Front"].sum()),
        "notes": notes,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote SAM-MFP results to {out_dir}")


if __name__ == "__main__":
    main()
