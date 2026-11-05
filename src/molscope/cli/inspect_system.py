"""CLI for reviewing system capabilities before selecting analysis modules."""

from __future__ import annotations

import argparse
import json

from molscope.profiling.analyzers import AnalyzerCatalog
from molscope.features.registry import DEFAULT_FEATURE_REGISTRY
from molscope.profiling.system import profile_system


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile an MD system and assess generic analysis applicability.")
    parser.add_argument("--topology", required=True, help="Topology file readable by MDAnalysis.")
    parser.add_argument("--trajectory", default=None, help="Optional trajectory to inspect periodic-cell metadata.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args()

    profile = profile_system(args.topology, args.trajectory)
    payload = {
        "system_profile": profile.to_dict(),
        "analysis_applicability": [DEFAULT_FEATURE_REGISTRY.assess(
            item.feature_id, profile,
            available_inputs={"topology", "trajectory"} if args.trajectory else {"topology"},
        ).to_dict() for item in DEFAULT_FEATURE_REGISTRY.values()],
        "scientific_feature_definitions": [item.to_dict() for item in DEFAULT_FEATURE_REGISTRY.values()],
        "proposed_analysis_families": [assessment.to_dict() for assessment in AnalyzerCatalog().assess(profile)],
        "catalogue_policy": "Scientific registry is authoritative; proposed families do not imply implementation availability.",
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
