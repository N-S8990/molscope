"""Synthetic scaling benchmark; measures analysis, not MD extraction accuracy."""
import argparse
import json
from pathlib import Path
import resource
import sys
from time import perf_counter

import numpy as np
import pandas as pd

from molscope.execution.stream_analysis import StreamAnalysisConfig, analyze_stream


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.frames < 10:
        parser.error("--frames must be at least 10")
    root = Path(args.out_dir)
    root.mkdir(parents=True, exist_ok=False)
    source = root / "synthetic_chunks"
    source.mkdir()
    rng = np.random.default_rng(17)
    chunks = []
    for start in range(0, args.frames, 10000):
        index = np.arange(start, min(args.frames, start + 10000))
        table = pd.DataFrame({"frame_index": index, "trajectory_id": "synthetic_benchmark",
                              "signal": np.sin(index / 200), "noise": rng.normal(size=len(index)),
                              "contact": (index % 97 < 30).astype(float)})
        name = f"chunk_{start:09d}.csv"
        table.to_csv(source / name, index=False)
        chunks.append({"file": name})
    (source / "manifest.json").write_text(json.dumps({"status": "completed", "feature_columns": ["signal", "noise", "contact"], "chunks": chunks}))
    start = perf_counter()
    analyze_stream(source, root / "analysis", StreamAnalysisConfig(batch_rows=10000, sample_rows=2000))
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {"scope": "synthetic three-feature analysis only", "frames": args.frames,
              "analysis_wall_seconds": perf_counter() - start,
              "process_peak_rss_mb": peak / (1024 ** 2 if sys.platform == "darwin" else 1024),
              "sample_rows": 2000, "batch_rows": 10000}
    (root / "benchmark.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
