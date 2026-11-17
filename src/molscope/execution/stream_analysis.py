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
    state_rows = [{"state": int(i), "frames": int(count), "fraction": float(count / moments.count)}
                  for i, count in enumerate(counts)]
    state_rows.sort(key=lambda row: row["frames"], reverse=True)
    largest_state = state_rows[0] if state_rows else None
    variable = summary_table.loc[summary_table["Variable_In_Training_Sample"]].nlargest(8, "Std")
    feature_rows = "".join(
        f"<tr><td>{html.escape(str(row.Feature))}</td><td>{row.Mean:.4g}</td><td>{row.Std:.4g}</td><td>{row.Minimum:.4g}</td><td>{row.Maximum:.4g}</td></tr>"
        for row in variable.itertuples(index=False)
    ) or "<tr><td colspan='5'>No variable feature measurements in the training sample.</td></tr>"
    population_rows = "".join(
        f"<tr><td>State {row['state']}</td><td>{row['frames']:,}</td><td>{row['fraction']:.1%}</td></tr>"
        for row in sorted(state_rows, key=lambda row: row["state"])
    )
    imbalance = (f"State {largest_state['state']} is the most populated candidate ({largest_state['fraction']:.1%} of frames)."
                 if largest_state else "No candidate states were assigned.")
    report["largest_candidate_state"] = largest_state
    report["top_variable_features_by_sample_standard_deviation"] = variable["Feature"].astype(str).tolist()
    css = """
      :root{color-scheme:light;--ink:#17212b;--muted:#586777;--line:#d9e1e8;--blue:#245b78;--paper:#fff;--wash:#f3f7fa}
      *{box-sizing:border-box}body{margin:0;background:var(--wash);color:var(--ink);font:15px/1.55 system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif}
      main{max-width:1120px;margin:36px auto;padding:0 22px 48px}header{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:28px 32px;margin-bottom:18px}
      h1{font-size:30px;line-height:1.2;margin:0 0 8px}h2{font-size:20px;margin:0 0 12px}h3{font-size:15px;margin:24px 0 8px;color:var(--blue)}p{margin:6px 0;color:var(--muted)}
      .eyebrow{text-transform:uppercase;letter-spacing:.08em;font-size:12px;font-weight:700;color:var(--blue)}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
      .card,section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:20px}.metric{font-size:26px;font-weight:700;line-height:1.2}.label{font-size:12px;color:var(--muted);margin-top:5px}
      section{margin-top:14px}table{width:100%;border-collapse:collapse;margin-top:10px;font-size:14px}th,td{text-align:left;padding:9px 10px;border-bottom:1px solid var(--line)}th{font-size:12px;color:var(--muted);font-weight:650;background:#f8fafb}td:first-child{overflow-wrap:anywhere}
      img{display:block;max-width:100%;height:auto;margin:12px auto 2px}.note{border-left:3px solid #d19531;padding:10px 13px;background:#fff9eb;border-radius:3px;color:#564625}
      nav a{color:var(--blue);text-decoration:none;margin-right:16px}footer{font-size:12px;color:var(--muted);padding:20px 4px}
      @media(max-width:680px){main{margin:18px auto;padding:0 12px 30px}.cards{grid-template-columns:1fr}header,section{padding:18px}table{font-size:12px}th,td{padding:7px 5px}}
    """
    text = f"""<!doctype html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
      <title>MolScope | Streaming analysis</title><style>{css}</style></head><body><main>
      <header><div class='eyebrow'>MolScope · descriptive analysis</div><h1>Streaming trajectory report</h1>
      <p>State discovery was fitted on a reproducible sample and applied to all analyzed frames.</p>
      <nav><a href='#results'>Results</a><a href='#features'>Variable features</a><a href='#methods'>Scope and methods</a><a href='analysis_summary.json'>Full analysis record</a></nav></header>
      <div class='cards'><div class='card'><div class='metric'>{moments.count:,}</div><div class='label'>frames analyzed</div></div>
      <div class='card'><div class='metric'>{width:,}</div><div class='label'>measured feature columns</div></div>
      <div class='card'><div class='metric'>{len(sample):,}</div><div class='label'>frames used to fit PCA and clustering</div></div></div>
      <section id='results'><h2>Results at a glance</h2><p>{config.states} candidate states were assigned across all analyzed frames. {imbalance}</p>
      <table><thead><tr><th>Candidate state</th><th>Frames</th><th>Population</th></tr></thead><tbody>{population_rows}</tbody></table>
      <h3>State populations</h3><img src='state_populations.png' alt='Fraction of analyzed frames assigned to each candidate state'>
      <p>State labels are descriptive clusters in the fitted PCA representation; they are not experimentally validated states.</p></section>
      <section id='features'><h2>Most variable measured features</h2><p>Sorted by standard deviation in the training sample. Units remain those of the extracted feature.</p>
      <table><thead><tr><th>Feature</th><th>Mean</th><th>Std. dev.</th><th>Minimum</th><th>Maximum</th></tr></thead><tbody>{feature_rows}</tbody></table>
      <p><a href='feature_summary.csv'>Download complete feature summary</a></p></section>
      <section id='methods'><h2>Scope and interpretation</h2><p class='note'>This run performs sample-trained descriptive state discovery. It does not calculate replica stability, SAM-MFP prioritization, supervised prediction, or kinetic rates.</p>
      <h3>Method and resource record</h3><table><tbody><tr><th>Training sample</th><td>{len(sample):,} uniformly reservoir-sampled frames; seed {config.seed}</td></tr>
      <tr><th>Representation</th><td>Standard scaling followed by {components}-component PCA on {int(active.sum()):,} variable features</td></tr>
      <tr><th>Clustering</th><td>K-means with {config.states} requested states; 10 initializations</td></tr>
      <tr><th>Estimated array workspace</th><td>{estimate / (1024 * 1024):.1f} MiB (estimate, not a process memory limit)</td></tr></tbody></table>
      <h3>Limitations</h3><ul>{''.join(f'<li>{html.escape(line)}</li>' for line in report['limitations'])}</ul>
      <p><a href='state_populations.csv'>State population data</a> · <a href='feature_summary.csv'>Feature summary data</a> · <a href='analysis_summary.json'>Chunk hashes and complete provenance</a></p></section>
      <footer>MolScope streaming analysis · {moments.count:,} frames processed</footer></main></body></html>"""
    (destination / "report.html").write_text(text, encoding="utf-8")
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
