"""CLI for label-independent feature quality assessment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from molscope.core.identity import TrajectoryInput, build_frame_manifest
from molscope.analysis.quality.assessment import PrioritizationConfig, assess_feature_quality, validate_manifest


def legacy_manifest_from_run_metadata(path: str | Path) -> pd.DataFrame:
    """Adapt the legacy class0/class1 row order into explicit identity metadata.

    This adapter is only for existing MolScope output.  New workflows should
    write a manifest during extraction instead of relying on row order.
    """
    metadata_path = Path(path)
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    arguments = payload["arguments"]
    base = metadata_path.parent.parent if metadata_path.parent.name == "results" else metadata_path.parent
    inputs = []
    for ensemble_index, (topology_key, trajectory_key) in enumerate((("class0_top", "class0_traj"), ("class1_top", "class1_traj"))):
        topology = Path(arguments[topology_key])
        if not topology.is_absolute():
            topology = base / topology
        for replica_number, trajectory in enumerate(arguments[trajectory_key], start=1):
            trajectory_path = Path(trajectory)
            if not trajectory_path.is_absolute():
                trajectory_path = base / trajectory_path
            inputs.append(TrajectoryInput(
                topology=topology,
                trajectory=trajectory_path,
                trajectory_id=f"condition_{ensemble_index}:replica_{replica_number}",
                replica_id=f"replica_{replica_number}",
                ensemble_id=f"condition_{ensemble_index}",
            ))
    return build_frame_manifest(
        inputs,
        frame_stride=int(arguments.get("frame_stride", 1)),
        max_frames=arguments.get("max_frames"),
    )


def _plot_quality(quality: pd.DataFrame, output_path: Path) -> None:
    plotted = quality[quality["Pareto_Eligible"]].copy()
    if plotted.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    non_front = plotted[~plotted["Pareto_Front"]]
    front = plotted[plotted["Pareto_Front"]]
    ax.scatter(non_front["ESS_Ratio"], non_front["Replica_Consistency"], s=18, alpha=0.35, label="Other eligible features")
    ax.scatter(front["ESS_Ratio"], front["Replica_Consistency"], s=36, color="#d95f02", label="Pareto candidates")
    ax.set(xlabel="Effective sample-size ratio", ylabel="Replica consistency", title="Label-independent feature-quality trade-offs")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Assess MD feature quality without ML labels.")
    parser.add_argument("--features", required=True, help="CSV feature matrix; rows must align with the manifest.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="CSV containing trajectory_id, replica_id, and ensemble_id per feature row.")
    source.add_argument("--legacy-run-metadata", help="Existing MolScope run_metadata.json; reconstructs legacy row order.")
    parser.add_argument("--out-dir", default="feature_quality_results", help="Directory for tables, report, and plot.")
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--max-correlation-frames", type=int, default=5_000)
    parser.add_argument("--block-size-frames", type=int, default=20, help="Contiguous block width for within-trajectory uncertainty diagnostics.")
    parser.add_argument("--feature-costs", default=None, help="Optional CSV with Feature and Elapsed_Seconds columns.")
    parser.add_argument("--pareto-bootstrap-samples", type=int, default=0, help="Resample complete trajectories to estimate Pareto-front membership stability.")
    args = parser.parse_args()

    if not 0.0 < args.min_valid_fraction <= 1.0:
        parser.error("--min-valid-fraction must be in (0, 1].")
    features = pd.read_csv(args.features)
    manifest = pd.read_csv(args.manifest) if args.manifest else legacy_manifest_from_run_metadata(args.legacy_run_metadata)
    validate_manifest(manifest, len(features))
    config = PrioritizationConfig(
        min_valid_fraction=args.min_valid_fraction,
        max_correlation_frames=args.max_correlation_frames,
        block_size_frames=args.block_size_frames,
        pareto_bootstrap_samples=args.pareto_bootstrap_samples,
    )
    costs = pd.read_csv(args.feature_costs) if args.feature_costs else None
    quality = assess_feature_quality(features, manifest, config, feature_costs=costs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    quality.to_csv(out_dir / "feature_quality_metrics.csv", index=False)
    quality[quality["Resolution"] == "trajectory"].to_csv(out_dir / "trajectory_level_features.csv", index=False)
    quality[quality["Pareto_Front"]].sort_values("Feature").to_csv(out_dir / "pareto_feature_candidates.csv", index=False)
    _plot_quality(quality, out_dir / "pareto_quality_tradeoffs.png")
    report = {
        "feature_count": int(len(quality)),
        "frame_resolved_features": int((quality["Resolution"] == "frame").sum()),
        "trajectory_level_features": int((quality["Resolution"] == "trajectory").sum()),
        "unavailable_features": int((quality["Resolution"] == "unavailable").sum()),
        "pareto_candidates": int(quality["Pareto_Front"].sum()),
        "configuration": config.to_dict(),
        "notes": [
            "No target labels were used to calculate this report.",
            "Pareto candidates are trade-offs, not a single importance ranking.",
            "Legacy manifests are reconstructed from historical row order; future extraction must write the manifest directly.",
            "Cost is reported only when a measured per-feature cost table is supplied; rank-bootstrap uncertainty is not yet implemented.",
            "Block standard errors are within-trajectory diagnostics and do not replace independent-replica uncertainty.",
            "Pareto membership frequency, when requested, resamples complete trajectories within ensembles and measures sensitivity to the available replicas.",
        ],
    }
    (out_dir / "feature_quality_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote feature-quality audit to {out_dir}")


if __name__ == "__main__":
    main()
