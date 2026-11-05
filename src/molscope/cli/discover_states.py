"""CLI for ML-based conformational-state discovery."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from molscope.analysis.discovery.states import StateDiscoveryConfig, discover_conformational_states, evaluate_state_count_sensitivity


def main() -> None:
    parser = argparse.ArgumentParser(description="Cluster target-free ML coordinates into candidate conformational states.")
    parser.add_argument("--coordinates", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--states", type=int, default=2)
    parser.add_argument("--state-counts", default=None, help="Comma-separated candidate counts for sensitivity diagnostics, e.g. 2,3,4,5.")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    outputs = discover_conformational_states(pd.read_csv(args.coordinates), pd.read_csv(args.manifest), StateDiscoveryConfig(n_states=args.states))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    if args.state_counts:
        counts = tuple(int(value.strip()) for value in args.state_counts.split(",") if value.strip())
        sensitivity = evaluate_state_count_sensitivity(pd.read_csv(args.coordinates), pd.read_csv(args.manifest), counts)
        sensitivity.to_csv(out_dir / "state_count_sensitivity.csv", index=False)
    (out_dir / "state_report.json").write_text(json.dumps({"state_count": args.states, "notes": ["State assignments are ML clustering results, not experimentally validated metastable states.", "Frames are not treated as independent replicas; state populations are reported per trajectory.", "Reported state transitions are successive-frame counts and conditional probabilities, not kinetic rates.", "Silhouette and initialization agreement are diagnostics, not proof that the chosen state count is physically correct."]}, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote state-discovery results to {out_dir}")


if __name__ == "__main__":
    main()
