"""CLI for read-only topology and trajectory integrity checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from molscope.io.validation import inspect_md_inputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate MD topology/trajectory compatibility and record molecular identities."
    )
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--checksum", action="store_true", help="Record SHA-256 digests; can be slow for large files.")
    parser.add_argument("--include-identities", action="store_true", help="Include atom and residue identity tables in JSON output.")
    parser.add_argument("--output", default=None, help="Write JSON to this path instead of stdout.")
    args = parser.parse_args()

    report = inspect_md_inputs(
        args.topology,
        args.trajectory,
        frame_stride=args.frame_stride,
        max_frames=args.max_frames,
        include_checksum=args.checksum,
    )
    rendered = json.dumps(report.to_dict(include_identities=args.include_identities), indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    if report.status.value == "invalid":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
