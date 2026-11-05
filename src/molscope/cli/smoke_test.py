"""Generate a synthetic analysis report to check output generation.

Synthetic trajectories test software only; they are not molecular evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from molscope.execution.discovery import DiscoveryPipelineConfig
from molscope.execution.pipeline import MDContactMLPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    destination = Path(args.out_dir)
    if destination.exists():
        raise FileExistsError("Choose a new output directory for this smoke test.")
    rng = np.random.default_rng(42)
    features, manifest = [], []
    for trajectory in range(6):
        target = trajectory // 3
        for frame in range(24):
            features.append({
                "synthetic_signal": target * 2 + np.sin(frame / 4) + rng.normal(0, .1),
                "synthetic_noise": rng.normal(),
                "synthetic_motion": np.cos(frame / 3) + rng.normal(0, .1),
            })
            manifest.append({"trajectory_id": f"synthetic_{trajectory}",
                             "replica_id": f"r{trajectory}", "ensemble_id": f"e{target}",
                             "system_id": "synthetic_software_test", "frame_order": frame,
                             "condition": target})
    pipeline = MDContactMLPipeline(DiscoveryPipelineConfig(
        pca_components=2, state_count=2, supervised_splits=3,
        supervised_bootstrap_samples=20,
    )).load_feature_matrix(pd.DataFrame(features), pd.DataFrame(manifest))
    pipeline.run(target_column="condition").write_results(destination)
    expected = ["research_report.md", "research_report.json", "sam_mfp_tradeoff.png",
                "state_populations.png", "feature_quality_tradeoff.png",
                "representation_salience.png", "predictive_feature_ranking.png",
                "state_explanatory_feature_ranking.png"]
    missing = [name for name in expected if not (destination / name).is_file()]
    result = {"scope": "synthetic software smoke test; no scientific conclusions",
              "status": "failed" if missing else "passed", "missing_outputs": missing,
              "output_directory": str(destination.resolve())}
    (destination / "smoke_status.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
