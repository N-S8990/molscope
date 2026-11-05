"""Reproducible, tool-neutral benchmark planning and result collation.

This module deliberately does not import or execute mdxplain.  A comparison is
only defensible when every method is run against the same declared trajectory
identity manifest; external-tool execution remains an explicit researcher
step.  The harness validates that condition, normalizes supplied metric tables,
and records unavailable methods rather than silently omitting them.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_IDENTITY_COLUMNS = ("system_id", "ensemble_id", "replica_id", "trajectory_id")


def _markdown_table(table: pd.DataFrame) -> str:
    """Render a small Markdown table without imposing an optional dependency."""
    if table.empty:
        return "No results available."
    columns = list(table.columns)
    rows = table.fillna("not available").astype(str).itertuples(index=False, name=None)
    return "\n".join([
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
        *["| " + " | ".join(row) + " |" for row in rows],
    ])


@dataclass(frozen=True)
class BenchmarkMetric:
    metric_id: str
    unit: str
    higher_is_better: bool | None = None
    description: str = ""


@dataclass(frozen=True)
class BenchmarkMethod:
    method_id: str
    label: str
    kind: str
    results_csv: str | None = None
    input_manifest: str | None = None
    checkout_path: str | None = None
    version: str | None = None
    protocol: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BenchmarkSpec:
    benchmark_id: str
    input_manifest: str
    methods: tuple[BenchmarkMethod, ...]
    metrics: tuple[BenchmarkMetric, ...]
    identity_columns: tuple[str, ...] = DEFAULT_IDENTITY_COLUMNS
    description: str = ""
    schema_version: str = "1"
    shared_protocol: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _require_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string.")
    return value


def load_benchmark_spec(path: str | Path) -> BenchmarkSpec:
    """Load and validate a benchmark declaration without resolving its files."""
    source = Path(path)
    raw = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Benchmark declaration must be a JSON object.")
    benchmark_id = _require_string(raw.get("benchmark_id"), "benchmark_id")
    input_manifest = _require_string(raw.get("input_manifest"), "input_manifest")
    identity_columns = tuple(raw.get("identity_columns", DEFAULT_IDENTITY_COLUMNS))
    if not identity_columns or not all(isinstance(column, str) and column for column in identity_columns):
        raise ValueError("identity_columns must be a non-empty list of column names.")
    raw_metrics = raw.get("metrics")
    if not isinstance(raw_metrics, list) or not raw_metrics:
        raise ValueError("metrics must be a non-empty list.")
    metrics = tuple(BenchmarkMetric(
        metric_id=_require_string(item.get("metric_id"), "metrics[].metric_id"),
        unit=_require_string(item.get("unit"), "metrics[].unit"),
        higher_is_better=item.get("higher_is_better"),
        description=str(item.get("description", "")),
    ) for item in raw_metrics if isinstance(item, dict))
    if len(metrics) != len(raw_metrics) or len({item.metric_id for item in metrics}) != len(metrics):
        raise ValueError("metrics must contain unique object metric_id values.")
    raw_methods = raw.get("methods")
    if not isinstance(raw_methods, list) or not raw_methods:
        raise ValueError("methods must be a non-empty list.")
    methods = []
    for item in raw_methods:
        if not isinstance(item, dict):
            continue
        protocol = item.get("protocol", {})
        if not isinstance(protocol, dict):
            raise ValueError("methods[].protocol must be an object when supplied.")
        methods.append(BenchmarkMethod(
            method_id=_require_string(item.get("method_id"), "methods[].method_id"),
            label=_require_string(item.get("label"), "methods[].label"),
            kind=_require_string(item.get("kind"), "methods[].kind"),
            results_csv=item.get("results_csv"), input_manifest=item.get("input_manifest"),
            checkout_path=item.get("checkout_path"), version=item.get("version"), protocol=protocol,
        ))
    methods = tuple(methods)
    if len(methods) != len(raw_methods) or len({item.method_id for item in methods}) != len(methods):
        raise ValueError("methods must contain unique object method_id values.")
    shared_protocol = raw.get("shared_protocol", {})
    if not isinstance(shared_protocol, dict):
        raise ValueError("shared_protocol must be an object when supplied.")
    return BenchmarkSpec(benchmark_id, input_manifest, methods, metrics, identity_columns, str(raw.get("description", "")), str(raw.get("schema_version", "1")), shared_protocol)


def _read_manifest(path: str | Path, identity_columns: tuple[str, ...]) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    missing = sorted(set(identity_columns) - set(manifest.columns))
    if missing:
        raise ValueError(f"Input manifest {path!s} lacks identity columns: {missing}")
    identity = manifest.loc[:, list(identity_columns)].copy()
    if identity.isna().any().any() or identity.duplicated().any():
        raise ValueError(f"Input manifest {path!s} has missing or duplicate trajectory identities.")
    return identity.sort_values(list(identity_columns)).reset_index(drop=True)


def inspect_mdxplain_checkout(path: str | Path | None) -> dict[str, str]:
    """Report checkout availability without importing an external dependency."""
    if path is None:
        return {"status": "not_configured", "reason": "No mdxplain checkout path was declared."}
    checkout = Path(path)
    if not checkout.is_dir():
        return {"status": "unavailable", "reason": f"Checkout directory does not exist: {checkout}"}
    if not (checkout / "pyproject.toml").is_file() or not (checkout / "mdxplain" / "__init__.py").is_file():
        return {"status": "unavailable", "reason": "Path is not a recognizable mdxplain checkout."}
    return {"status": "available_not_executed", "reason": "Checkout is available; run mdxplain separately and supply normalized results_csv."}


def _method_status(method: BenchmarkMethod, shared_identity: pd.DataFrame, identity_columns: tuple[str, ...]) -> tuple[str, str]:
    if method.input_manifest:
        method_identity = _read_manifest(method.input_manifest, identity_columns)
        if not shared_identity.equals(method_identity):
            return "input_mismatch", "Method input manifest does not exactly match the benchmark input identities."
    if method.kind == "mdxplain":
        checkout = inspect_mdxplain_checkout(method.checkout_path)
        if checkout["status"] == "unavailable":
            return checkout["status"], checkout["reason"]
    if not method.results_csv:
        return "pending_results", "No normalized results_csv was supplied."
    if not Path(method.results_csv).is_file():
        return "unavailable", f"Results CSV does not exist: {method.results_csv}"
    return "ready", "Matched inputs and a results CSV are available."


def _normalize_metrics(method: BenchmarkMethod, metric_ids: set[str]) -> pd.DataFrame:
    table = pd.read_csv(method.results_csv)
    required = {"metric_id", "value"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"Results CSV for {method.method_id!r} lacks columns: {missing}")
    normalized = table.loc[:, [column for column in table.columns if column in {"metric_id", "value", "subset", "notes"}]].copy()
    normalized["metric_id"] = normalized["metric_id"].astype(str)
    unknown = sorted(set(normalized["metric_id"]) - metric_ids)
    if unknown:
        raise ValueError(f"Results CSV for {method.method_id!r} reports undeclared metrics: {unknown}")
    normalized["value"] = pd.to_numeric(normalized["value"], errors="raise")
    normalized.insert(0, "method_id", method.method_id)
    normalized.insert(1, "method_label", method.label)
    return normalized


def build_benchmark(spec: BenchmarkSpec) -> dict[str, pd.DataFrame | dict | str]:
    """Validate matched inputs and collate available results into comparison tables."""
    shared_identity = _read_manifest(spec.input_manifest, spec.identity_columns)
    statuses, metrics = [], []
    metric_ids = {metric.metric_id for metric in spec.metrics}
    for method in spec.methods:
        status, reason = _method_status(method, shared_identity, spec.identity_columns)
        statuses.append({"method_id": method.method_id, "label": method.label, "kind": method.kind, "protocol_declared": bool(method.protocol), "status": status, "reason": reason})
        if status == "ready":
            metrics.append(_normalize_metrics(method, metric_ids))
    metric_table = pd.concat(metrics, ignore_index=True) if metrics else pd.DataFrame(columns=["method_id", "method_label", "metric_id", "value"])
    declared_metrics = pd.DataFrame([asdict(metric) for metric in spec.metrics])
    status_table = pd.DataFrame.from_records(statuses)
    summary = {
        "benchmark_id": spec.benchmark_id,
        "schema_version": spec.schema_version,
        "matched_trajectory_count": len(shared_identity),
        "ready_method_count": int((status_table["status"] == "ready").sum()),
        "method_statuses": status_table.set_index("method_id")["status"].to_dict(),
        "shared_protocol": spec.shared_protocol,
        "limitations": [
            "The harness collates supplied results; it does not execute external methods.",
            "Metrics are comparable only when each method used the declared matched input manifest and protocol.",
        ],
    }
    markdown = "\n".join([
        f"# Benchmark: {spec.benchmark_id}", "", spec.description, "",
        "## Matched inputs", "", f"Declared matched trajectories: {len(shared_identity)}.", "",
        "## Declared shared protocol", "", f"`{json.dumps(spec.shared_protocol, sort_keys=True)}`" if spec.shared_protocol else "No shared protocol details were declared.", "",
        "## Method readiness", "", _markdown_table(status_table), "",
        "## Result metrics", "", _markdown_table(metric_table) if not metric_table.empty else "No method results are available yet.", "",
        "## Interpretation boundary", "", *[f"- {item}" for item in summary["limitations"]], "",
    ])
    return {"method_status": status_table, "metrics": metric_table, "declared_metrics": declared_metrics, "summary": summary, "markdown": markdown}


def write_benchmark(outputs: dict[str, pd.DataFrame | dict | str], output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("method_status", "metrics", "declared_metrics"):
        value = outputs[name]
        assert isinstance(value, pd.DataFrame)
        value.to_csv(destination / f"{name}.csv", index=False)
    (destination / "benchmark_summary.json").write_text(json.dumps(outputs["summary"], indent=2) + "\n", encoding="utf-8")
    (destination / "benchmark_report.md").write_text(str(outputs["markdown"]) + "\n", encoding="utf-8")
    return destination
