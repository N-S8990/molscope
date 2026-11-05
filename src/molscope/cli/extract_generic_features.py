"""CLI for generic, capability-aware coordinate feature extraction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molscope.features.geometry.generic import GenericFeatureConfig, GenericFeatureExtractor


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract generic MD features without protein-specific assumptions.")
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--contact-cutoff", type=float, default=4.5, help="Explicit all-atom proximity cutoff in angstroms.")
    parser.add_argument("--use-pbc", action="store_true", help="Use the trajectory cell for contacts only after PBC validation.")
    parser.add_argument("--include-environment-in-global-features", action="store_true", help="Include solvent and ions in global shape, displacement, and RMSF descriptors.")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    args = parser.parse_args()

    config = GenericFeatureConfig(
        contact_cutoff_angstrom=args.contact_cutoff,
        use_pbc=args.use_pbc,
        include_environment_in_global_features=args.include_environment_in_global_features,
        max_frames=args.max_frames,
        frame_stride=args.frame_stride,
    )
    result = GenericFeatureExtractor(config).extract(args.topology, args.trajectory)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result.frame_features.to_csv(out_dir / "frame_features.csv", index=False)
    result.trajectory_features.to_csv(out_dir / "trajectory_features.csv", index=False)
    (out_dir / "feature_assessments.json").write_text(json.dumps(result.assessments_payload(), indent=2) + "\n", encoding="utf-8")
    (out_dir / "feature_provenance.json").write_text(json.dumps(result.provenance_payload(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result.frame_features)} frame rows and {len(result.trajectory_features)} trajectory rows to {out_dir}")


if __name__ == "__main__":
    main()
