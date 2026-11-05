"""Portable persistent tables and content-addressed extraction cache.

Parquet is used when an installed pandas engine supports it.  A CSV fallback
keeps sessions and caches usable in the base installation, while a small JSON
sidecar records the chosen representation explicitly.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pandas as pd


STORAGE_SCHEMA_VERSION = 1


def _json_default(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | frozenset | tuple):
        return sorted(value) if isinstance(value, (set, frozenset)) else list(value)
    raise TypeError(f"Cannot serialize {type(value).__name__} into cache metadata.")


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, default=_json_default, sort_keys=True, separators=(",", ":"))


def file_fingerprint(path: str | Path, *, chunk_size: int = 1 << 20) -> dict[str, object]:
    """Return a content fingerprint instead of trusting a mutable path name."""
    source = Path(path)
    digest = sha256()
    with source.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    stat = source.stat()
    return {"path": str(source.resolve()), "size_bytes": stat.st_size, "sha256": digest.hexdigest()}


def extraction_cache_key(
    topology: str | Path,
    trajectory: str | Path,
    *,
    generic_config: object,
    plugins: dict[str, Any],
) -> str:
    """Key extracted frames by source contents, settings, and plugin versions."""
    payload = {
        "schema_version": STORAGE_SCHEMA_VERSION,
        "topology": file_fingerprint(topology),
        "trajectory": file_fingerprint(trajectory),
        "generic_config": generic_config,
        "plugins": plugins,
    }
    return sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def write_table(table: pd.DataFrame, destination: str | Path) -> Path:
    """Write a table as Parquet when possible, otherwise as a documented CSV."""
    base = Path(destination)
    base.parent.mkdir(parents=True, exist_ok=True)
    parquet_path = base.with_suffix(".parquet")
    try:
        table.to_parquet(parquet_path, index=False)
    except (ImportError, ModuleNotFoundError, ValueError):
        csv_path = base.with_suffix(".csv")
        table.to_csv(csv_path, index=False)
        return csv_path
    return parquet_path


def read_table(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if source.suffix == ".parquet":
        return pd.read_parquet(source)
    if source.suffix == ".csv":
        return pd.read_csv(source)
    raise ValueError(f"Unsupported persisted table format: {source}")


class FeatureCache:
    """Small content-addressed cache for fully combined per-trajectory frames."""

    def __init__(self, cache_dir: str | Path):
        self.root = Path(cache_dir)

    def _metadata_path(self, key: str) -> Path:
        return self.root / key / "entry.json"

    def get(self, key: str) -> pd.DataFrame | None:
        metadata_path = self._metadata_path(key)
        if not metadata_path.is_file():
            return None
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("schema_version") != STORAGE_SCHEMA_VERSION:
                return None
            table_path = metadata_path.parent / metadata["table"]
            return read_table(table_path) if table_path.is_file() else None
        except (OSError, KeyError, ValueError, json.JSONDecodeError):
            return None

    def put(self, key: str, table: pd.DataFrame) -> Path:
        entry_dir = self.root / key
        entry_dir.mkdir(parents=True, exist_ok=True)
        table_path = write_table(table, entry_dir / "frame_features")
        metadata = {
            "schema_version": STORAGE_SCHEMA_VERSION,
            "table": table_path.name,
            "row_count": len(table),
            "columns": list(table.columns),
        }
        self._metadata_path(key).write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return table_path

    def get_evidence(self, key: str) -> dict | None:
        """Legacy frame-only entries are cache misses for audited extraction."""
        path = self.root / key / "evidence.json"
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("evidence_version") != 1:
                return None
            return payload
        except (OSError, ValueError, AttributeError):
            return None

    def put_evidence(self, key: str, evidence: dict) -> None:
        path = self.root / key / "evidence.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({**evidence, "evidence_version": 1}, indent=2, default=_json_default) + "\n", encoding="utf-8")
