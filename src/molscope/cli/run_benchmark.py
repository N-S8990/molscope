"""CLI for matched-input benchmark collation."""

from __future__ import annotations

import argparse

from molscope.benchmarking.core import build_benchmark, load_benchmark_spec, write_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate matched benchmark inputs and collate method results.")
    parser.add_argument("--benchmark", required=True, help="Benchmark JSON declaration.")
    parser.add_argument("--out-dir", required=True, help="Directory for benchmark CSV/JSON/Markdown outputs.")
    args = parser.parse_args()
    destination = write_benchmark(build_benchmark(load_benchmark_spec(args.benchmark)), args.out_dir)
    print(f"Wrote benchmark outputs to {destination}")


if __name__ == "__main__":
    main()
