"""CLI for the unified MolScope researcher report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molscope.reporting.research import ReportInputs, build_research_report, read_optional_csv, read_optional_json, render_report_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Assemble a cautious MolScope research report from pipeline outputs.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--quality")
    parser.add_argument("--representation-salience")
    parser.add_argument("--sam-mfp")
    parser.add_argument("--state-diagnostics")
    parser.add_argument("--state-populations")
    parser.add_argument("--supervised-summary")
    parser.add_argument("--system-profile")
    parser.add_argument("--input-integrity")
    args = parser.parse_args()
    inputs = ReportInputs(
        quality=read_optional_csv(args.quality), sam_mfp=read_optional_csv(args.sam_mfp),
        state_diagnostics=read_optional_csv(args.state_diagnostics), state_populations=read_optional_csv(args.state_populations),
        supervised_summary=read_optional_csv(args.supervised_summary), system_profile=read_optional_json(args.system_profile),
        input_integrity=read_optional_json(args.input_integrity),
    )
    markdown, summary = build_research_report(inputs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "research_report.md").write_text(markdown + "\n", encoding="utf-8")
    (out_dir / "research_report.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    render_report_figures(
        inputs.sam_mfp, inputs.state_populations, out_dir,
        quality=inputs.quality, representation_salience=read_optional_csv(args.representation_salience),
    )
    print(f"Wrote researcher report to {out_dir}")


if __name__ == "__main__":
    main()
