"""CLI for running one optional capability-gated feature plugin."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molscope.features.plugins import PLUGIN_REGISTRY, get_plugin


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract optional domain-specific MD features.")
    parser.add_argument("--plugin", required=True, choices=sorted(PLUGIN_REGISTRY))
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--parameters", default="{}", help="JSON object of plugin parameters.")
    args = parser.parse_args()

    result = get_plugin(args.plugin, json.loads(args.parameters)).extract(args.topology, args.trajectory, max_frames=args.max_frames, frame_stride=args.frame_stride)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.frame_features.to_csv(out_dir / "frame_features.csv", index=False)
    result.trajectory_features.to_csv(out_dir / "trajectory_features.csv", index=False)
    (out_dir / "feature_assessment.json").write_text(json.dumps(result.assessment.to_dict(), indent=2) + "\n", encoding="utf-8")
    (out_dir / "feature_provenance.json").write_text(json.dumps(result.provenance.to_dict() if result.provenance else None, indent=2) + "\n", encoding="utf-8")
    print(f"Plugin status: {result.assessment.status.value}; wrote {len(result.frame_features)} frame rows to {out_dir}")


if __name__ == "__main__":
    main()
