"""Bounded-batch summaries and sample-trained state discovery for chunk stores.

This deliberately does not compute replica stability or SAM-MFP from one
trajectory. All frames contribute to summaries and state populations; a seeded
uniform reservoir supplies PCA and cluster training only.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class StreamAnalysisConfig:
    batch_rows: int = 512
    sample_rows: int = 2000
    components: int = 3
    states: int = 2
    seed: int = 42
    memory_budget_mb: int = 512

    def __post_init__(self):
        for key in ("batch_rows", "sample_rows", "components", "states", "memory_budget_mb"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        if self.sample_rows < max(2, self.states):
            raise ValueError("sample_rows must be at least two and at least states")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2 ** 32:
            raise ValueError("seed must be an integer in [0, 2**32)")


class RunningMoments:
    """Parallel-Welford summary requiring O(feature count) retained memory."""
    def __init__(self, width: int):
        self.count = 0
        self.mean = np.zeros(width)
        self.m2 = np.zeros(width)
        self.minimum = np.full(width, np.inf)
        self.maximum = np.full(width, -np.inf)

    def update(self, values: np.ndarray) -> None:
        if not len(values):
            return
        count = len(values)
        mean = values.mean(axis=0)
        delta = mean - self.mean
        total = self.count + count
        self.m2 += np.sum((values - mean) ** 2, axis=0) + delta ** 2 * self.count * count / total
        self.mean += delta * count / total
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        self.count = total


class Reservoir:
    """Algorithm R uniform reservoir, deterministic regardless of batch sizes."""
    def __init__(self, capacity: int, width: int, seed: int):
        self.values = np.empty((capacity, width), dtype=np.float64)
        self.capacity = capacity
        self.seen = 0
        self.rng = np.random.default_rng(seed)

    def update(self, values: np.ndarray) -> None:
        for row in values:
            index = self.seen if self.seen < self.capacity else int(self.rng.integers(self.seen + 1))
            if index < self.capacity:
                self.values[index] = row
            self.seen += 1

    @property
    def sample(self):
        return self.values[:min(self.seen, self.capacity)]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_store(source: Path):
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if not (manifest.get("complete") is True or manifest.get("status") == "completed"):
        raise ValueError("Streaming extraction must be completed before analysis")
    columns = manifest.get("feature_columns", [])
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Store requires unique feature_columns")
    chunks = []
    for item in manifest.get("chunks", []):
        name = item["file"] if isinstance(item, dict) else item
        path = (source / name).resolve()
        if source.resolve() not in path.parents or not path.is_file():
            raise ValueError("Invalid or missing chunk path")
        digest = _sha256(path)
        if isinstance(item, dict) and item.get("sha256") and item["sha256"] != digest:
            raise ValueError(f"Chunk checksum mismatch: {name}")
        chunks.append((path, digest))
    if not chunks:
        raise ValueError("Store has no committed chunks")
    return manifest, columns, chunks


def analyze_stream(source: str | Path, output_dir: str | Path,
                   config: StreamAnalysisConfig = StreamAnalysisConfig()) -> dict:
    source, destination = Path(source), Path(output_dir)
    if destination.exists():
        raise FileExistsError("Analysis output exists; choose a fresh output directory (completed or partial outputs are never overwritten)")
    manifest, columns, chunks = _load_store(source)
    width = len(columns)
    # Includes conservative array copies/SVD workspace allowance. This is an
    # allocation guard, not an OS RSS cap; CSV parsing and libraries add overhead.
    estimate = 8 * width * (config.batch_rows * 4 + config.sample_rows * 8)
    if estimate > config.memory_budget_mb * 1024 ** 2:
        raise ValueError("Estimated analysis array workspace exceeds memory budget; reduce sample_rows, batch_rows, or feature selection")
    moments, reservoir = RunningMoments(width), Reservoir(config.sample_rows, width, config.seed)

    def batches():
        for path, expected_hash in chunks:
            if _sha256(path) != expected_hash:
                raise ValueError(f"Chunk changed during analysis: {path.name}")
            for table in pd.read_csv(path, chunksize=config.batch_rows):
                if not set(columns).issubset(table):
                    raise ValueError("Chunk feature schema mismatch")
                values = table[columns].to_numpy(dtype=np.float64)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite features found; streaming analysis does not impute unavailable values")
                yield table, values

    for _, values in batches():
        moments.update(values)
        reservoir.update(values)
    if "total_rows" in manifest and manifest["total_rows"] != moments.count:
        raise ValueError("Manifest total_rows does not match committed chunks")
    if moments.count < max(2, config.states):
        raise ValueError("Too few frames for requested states")
    sample = reservoir.sample
    active = np.ptp(sample, axis=0) > 1e-12
    if not active.any():
        raise ValueError("Training sample has no variable features")
    scaler = StandardScaler().fit(sample[:, active])
    standardized = scaler.transform(sample[:, active])
    components = min(config.components, standardized.shape[0] - 1, standardized.shape[1])
    pca = PCA(n_components=components, svd_solver="full").fit(standardized)
    training_coordinates = pca.transform(standardized)
    if len(np.unique(training_coordinates, axis=0)) < config.states:
        raise ValueError("Training sample has fewer distinct coordinates than requested states")
    model = KMeans(n_clusters=config.states, n_init=10, random_state=config.seed).fit(training_coordinates)
    destination.mkdir(parents=True)
    (destination / "assignments").mkdir()
    # All model parameters are data arrays; no executable pickle is required.
    np.savez(destination / "sample_model.npz", active_mask=active, scaler_mean=scaler.mean_,
             scaler_scale=scaler.scale_, pca_mean=pca.mean_, pca_components=pca.components_,
             pca_explained_variance_ratio=pca.explained_variance_ratio_,
             cluster_centers=model.cluster_centers_)
    counts = np.zeros(config.states, dtype=np.int64)
    assignment_files = []
    for batch_index, (table, values) in enumerate(batches()):
        coordinates = pca.transform(scaler.transform(values[:, active]))
        labels = model.predict(coordinates)
        counts += np.bincount(labels, minlength=config.states)
        identities = table.drop(columns=columns).copy()
        identities["State"] = labels
        for component in range(components):
            identities[f"PC{component + 1}"] = coordinates[:, component]
        name = f"assignments/chunk_{batch_index:08d}.csv"
        identities.to_csv(destination / name, index=False)
        assignment_files.append(name)
    summary_table = pd.DataFrame({"Feature": columns, "Count": moments.count, "Mean": moments.mean,
                                  "Std": np.sqrt(moments.m2 / max(1, moments.count - 1)),
                                  "Minimum": moments.minimum, "Maximum": moments.maximum,
                                  "Variable_In_Training_Sample": active})
    summary_table.to_csv(destination / "feature_summary.csv", index=False)
    pd.DataFrame({"State": range(config.states), "Frame_Count": counts,
                  "Population": counts / counts.sum()}).to_csv(destination / "state_populations.csv", index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.bar(range(config.states), counts / counts.sum())
    axis.set(xlabel="Sample-trained candidate state", ylabel="Fraction of all analyzed frames",
             title="Descriptive state populations")
    figure.tight_layout(); figure.savefig(destination / "state_populations.png", dpi=160); plt.close(figure)
    report = {"status": "completed", "scope": "sample-trained descriptive state discovery",
              "configuration": asdict(config), "frames_analyzed": moments.count,
              "training_sample_rows": len(sample), "feature_count": width,
              "active_feature_count": int(active.sum()), "estimated_array_workspace_bytes": estimate,
              "assignment_files": assignment_files,
              "source_chunks": [{"file": str(path), "sha256": digest} for path, digest in chunks],
              "limitations": ["PCA and clustering are fitted to a seeded uniform reservoir sample.",
                              "States are descriptive hypotheses; no replica stability or SAM-MFP was computed.",
                              "Array budget is an estimate, not a hard process memory cap.",
                              "Million-frame capacity has not been validated by this run unless its recorded frame count reaches that size."]}
    text = f"<h1>MolScope streaming analysis</h1><p>{moments.count} frames; {width} features; {len(sample)} training samples.</p>"
    text += "<ul>" + "".join(f"<li>{html.escape(line)}</li>" for line in report["limitations"]) + "</ul>"
    text += "<img src='state_populations.png' alt='State populations'><p><a href='feature_summary.csv'>Feature summaries</a> · <a href='state_populations.csv'>State populations</a> · <a href='analysis_summary.json'>Provenance and chunk index</a></p>"
    (destination / "report.html").write_text("<!doctype html><meta charset='utf-8'><title>MolScope streaming report</title>" + text)
    # Commit the completed marker only after every artifact has been written.
    pending = destination / "analysis_summary.json.tmp"
    pending.write_text(json.dumps(report, indent=2) + "\n")
    pending.replace(destination / "analysis_summary.json")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    for name, default in (("batch-rows", 512), ("sample-rows", 2000), ("components", 3), ("states", 2), ("seed", 42), ("memory-budget-mb", 512)):
        parser.add_argument(f"--{name}", type=int, default=default)
    args = parser.parse_args()
    report = analyze_stream(args.source, args.out_dir, StreamAnalysisConfig(
        batch_rows=args.batch_rows, sample_rows=args.sample_rows, components=args.components,
        states=args.states, seed=args.seed, memory_budget_mb=args.memory_budget_mb))
    print(json.dumps({"status": report["status"], "frames_analyzed": report["frames_analyzed"]}))


if __name__ == "__main__":
    main()
