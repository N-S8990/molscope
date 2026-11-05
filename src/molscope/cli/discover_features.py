"""CLI for target-free ML representation discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molscope.analysis.discovery.representation import RepresentationDiscoveryConfig, discover_representation_features


def main() -> None:
    parser = argparse.ArgumentParser(description="Discover target-free PCA feature salience with trajectory stability checks.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--components", type=int, default=5)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    outputs = discover_representation_features(
        pd.read_csv(args.features), pd.read_csv(args.manifest), RepresentationDiscoveryConfig(args.components),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "representation_report.json").write_text(json.dumps({
        "configuration": {"n_components": args.components},
        "notes": [
            "PCA salience summarizes contribution to this target-free representation.",
            "It is not causal importance or supervised predictive importance.",
            "LOTO rank stability leaves out complete trajectories.",
        ],
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote representation discovery results to {out_dir}")


if __name__ == "__main__":
    main()
