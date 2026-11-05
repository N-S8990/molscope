"""Bounded-memory, checkpointed frame extraction (not the full plugin pipeline).

Coordinates are analyzed as stored, without imaging or minimum-image distances.
Supply an already whole/imaged trajectory when molecular geometry requires it.
CSV chunks are committed by atomic manifest replacement. Resume rechecks source
and chunk content hashes; an uncommitted chunk is recomputed, never appended.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
import fcntl
from itertools import combinations
import json
import os
from pathlib import Path

import MDAnalysis as mda
from MDAnalysis.lib.distances import calc_bonds
import numpy as np

from molscope.features.geometry.generic import _asphericity, _geometric_radius_of_gyration, _kabsch_align, _rmsd
from molscope.features.plugins import _safe_id
from molscope.io.storage import file_fingerprint

GEOMETRY = "streaming.global_geometry"
CA = "protein.ca_distances_contacts"
IDENTITY_COLUMNS = ["frame_index", "frame_order", "time_ps"]


@dataclass(frozen=True)
class StreamingConfig:
    chunk_size: int = 1000
    frame_stride: int = 1
    max_frames: int | None = None
    geometry_selection: str = "all"
    ca_selection: str = "protein and name CA"
    contact_cutoff_angstrom: float = 8.0
    minimum_sequence_separation: int = 3
    max_pairs: int = 5000
    memory_budget_mb: float = 512.0
    imaging_policy: str = "as_stored_nonperiodic"
    plugins: tuple[str, ...] = (GEOMETRY, CA)

    def __post_init__(self):
        for name in ("chunk_size", "frame_stride", "max_pairs", "minimum_sequence_separation"):
            value = getattr(self, name)
            minimum = 0 if name == "minimum_sequence_separation" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}.")
        if self.max_frames is not None and (isinstance(self.max_frames, bool) or not isinstance(self.max_frames, int) or self.max_frames < 1):
            raise ValueError("max_frames must be a positive integer.")
        for name in ("contact_cutoff_angstrom", "memory_budget_mb"):
            if isinstance(getattr(self, name), bool) or not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive.")
        if self.imaging_policy != "as_stored_nonperiodic":
            raise ValueError("Only explicit as_stored_nonperiodic imaging policy is implemented.")
        if not self.plugins or len(set(self.plugins)) != len(self.plugins):
            raise ValueError("plugins must be nonempty and unique.")


def _atomic_json(path, payload):
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


@contextmanager
def _exclusive_lock(root):
    with (root / ".extraction.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError("Another extraction is writing this output directory.") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def extract_streaming(topology, trajectory, out_dir, *, config=None, resume=False):
    """Return manifest path; memory is independent of trajectory frame count.

    The memory estimate covers numeric extraction working arrays, not Python,
    imported libraries, or trajectory-reader internal buffers. CSV disk estimate
    uses 25 bytes/value and is approximate. No RMSF/DSSP/torsion claims are made.
    """
    config = config or StreamingConfig()
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    with _exclusive_lock(root):
        return _extract(topology, trajectory, root, config, resume)


def _extract(topology, trajectory, root, config, resume):
    manifest_path = root / "manifest.json"
    if manifest_path.exists() and not resume:
        raise ValueError("Output already contains a manifest; use resume or a new directory.")
    if resume and not manifest_path.exists():
        raise ValueError("Resume requires an existing manifest.")
    sources = {"topology": file_fingerprint(topology), "trajectory": file_fingerprint(trajectory)}
    settings = json.loads(json.dumps(asdict(config)))
    universe = mda.Universe(str(topology), str(trajectory))
    try:
        return _extract_universe(universe, root, manifest_path, sources, settings, config, resume)
    finally:
        universe.trajectory.close()


def _extract_universe(universe, root, manifest_path, sources, settings, config, resume):
    geometry = universe.select_atoms(config.geometry_selection)
    ca = universe.select_atoms(f"protein and name CA and ({config.ca_selection})")
    columns, pairs, assessments = [], [], []
    for plugin in config.plugins:
        status, reason = "unsupported", "Plugin has no bounded-memory implementation; omitted without zero filling."
        if plugin == GEOMETRY:
            status = "applicable" if len(geometry) else "not_applicable"
            reason = f"As-stored nonperiodic geometry selection: {config.geometry_selection}; {len(geometry)} atoms."
            if len(geometry):
                columns.extend(["GEOM_global_radius_gyration", "GEOM_global_asphericity", "DYN_kabsch_rmsd_to_first_frame"])
        elif plugin == CA:
            for left_index, right_index in combinations(range(len(ca)), 2):
                left, right = ca[left_index], ca[right_index]
                left_chain = getattr(left, "chainID", "") or left.segid
                right_chain = getattr(right, "chainID", "") or right.segid
                if left_chain == right_chain and abs(left.resid - right.resid) < config.minimum_sequence_separation:
                    continue
                pair_id = "__".join(_safe_id(f"{chain or 'unlabeled'}__{atom.resname}_{atom.resid}") for chain, atom in ((left_chain, left), (right_chain, right)))
                pairs.append((left_index, right_index, pair_id))
                if len(pairs) > config.max_pairs:
                    raise ValueError(f"Cα pair count exceeds max_pairs={config.max_pairs}; narrow ca_selection.")
                columns.extend([f"CA_DISTANCE_angstrom__{pair_id}", f"CA_CONTACT__{pair_id}"])
            status = "applicable" if pairs else "not_applicable"
            reason = "As-stored nonperiodic Cα pairs with declared sequence-separation and cutoff policy." if pairs else "No eligible protein Cα pairs."
        assessments.append({"feature_id": plugin, "status": status, "reason": reason, "granularity": "frame"})
    if len(set(columns)) != len(columns):
        raise ValueError("Ambiguous residue labels produce duplicate feature columns.")
    # No chunk-sized feature matrix is allocated: each frame is serialized immediately.
    memory_bytes = len(universe.atoms) * 3 * 8 * 16 + len(pairs) * 256 + len(columns) * 256
    if memory_bytes > config.memory_budget_mb * 1024**2:
        raise ValueError(f"Estimated extraction working memory {memory_bytes} bytes exceeds memory_budget_mb.")
    target_rows = len(range(0, len(universe.trajectory), config.frame_stride))
    if config.max_frames is not None:
        target_rows = min(target_rows, config.max_frames)
    manifest = {"schema_version": 1, "kind": "molscope.streaming_extraction", "sources": sources,
                "config": settings, "identity_columns": IDENTITY_COLUMNS, "feature_columns": columns,
                "assessments": assessments, "chunks": [], "total_rows": 0, "target_rows": target_rows,
                "complete": False, "estimated_working_memory_bytes": memory_bytes,
                "estimated_csv_bytes": target_rows * (len(columns) + 3) * 25,
                "estimate_scope": "Numeric extraction arrays only; excludes interpreter/reader overhead. CSV size is approximate.",
                "imaging_policy": config.imaging_policy}
    if resume:
        old = json.loads(manifest_path.read_text())
        for field in ("schema_version", "kind", "sources", "config", "identity_columns", "feature_columns", "target_rows"):
            if old.get(field) != manifest[field]:
                raise ValueError(f"Resume identity/configuration mismatch: {field}.")
        rows = 0
        for index, chunk in enumerate(old["chunks"]):
            expected_file = f"chunks/chunk_{index:06d}.csv"
            count = min(config.chunk_size, target_rows - rows)
            if (chunk["file"] != expected_file or chunk["rows"] != count or count <= 0
                    or chunk["first_frame_index"] != rows * config.frame_stride
                    or chunk["last_frame_index"] != (rows + count - 1) * config.frame_stride):
                raise ValueError("Invalid committed chunk sequence.")
            if file_fingerprint(root / expected_file)["sha256"] != chunk["sha256"]:
                raise ValueError("Committed chunk checksum mismatch.")
            rows += count
        if rows != old["total_rows"] or (old["complete"] and rows != target_rows):
            raise ValueError("Invalid committed row count.")
        manifest = old
    else:
        _atomic_json(manifest_path, manifest)
    (root / "chunks").mkdir(exist_ok=True)
    universe.trajectory[0]
    reference = geometry.positions.astype(float).copy() if len(geometry) and GEOMETRY in config.plugins else None
    pair_indices = np.array([(left, right) for left, right, _ in pairs], dtype=int)
    for start in range(manifest["total_rows"], target_rows, config.chunk_size):
        end = min(start + config.chunk_size, target_rows)
        relative = f"chunks/chunk_{len(manifest['chunks']):06d}.csv"
        destination = root / relative
        temporary = destination.with_suffix(".csv.tmp")
        with temporary.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(IDENTITY_COLUMNS + columns)
            for order in range(start, end):
                frame = order * config.frame_stride
                ts = universe.trajectory[frame]
                row = [frame, order, float(ts.time)]
                for plugin in config.plugins:
                    if plugin == GEOMETRY and reference is not None:
                        positions = geometry.positions.astype(float)
                        row.extend([_geometric_radius_of_gyration(positions), _asphericity(positions), _rmsd(_kabsch_align(positions, reference), reference)])
                    elif plugin == CA and pairs:
                        positions = ca.positions
                        distances = calc_bonds(positions[pair_indices[:, 0]], positions[pair_indices[:, 1]])
                        for distance in distances:
                            row.extend([float(distance), float(distance <= config.contact_cutoff_angstrom)])
                if not np.isfinite(row).all():
                    raise ValueError(f"Nonfinite coordinates/features at frame {frame}; chunk not committed.")
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        manifest["chunks"].append({"file": relative, "rows": end - start,
                                   "sha256": file_fingerprint(destination)["sha256"],
                                   "first_frame_index": start * config.frame_stride,
                                   "last_frame_index": (end - 1) * config.frame_stride})
        manifest["total_rows"] = end
        _atomic_json(manifest_path, manifest)
    manifest["complete"] = True
    _atomic_json(manifest_path, manifest)
    return manifest_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True)
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--memory-budget-mb", type=float, default=512)
    parser.add_argument("--geometry-selection", default="all")
    parser.add_argument("--ca-selection", default="protein and name CA")
    parser.add_argument("--imaging-policy", choices=["as_stored_nonperiodic"], default="as_stored_nonperiodic")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = StreamingConfig(chunk_size=args.chunk_size, frame_stride=args.stride, max_frames=args.max_frames,
                             max_pairs=args.max_pairs, memory_budget_mb=args.memory_budget_mb,
                             geometry_selection=args.geometry_selection, ca_selection=args.ca_selection,
                             imaging_policy=args.imaging_policy)
    print(extract_streaming(args.topology, args.trajectory, args.out_dir, config=config, resume=args.resume))


if __name__ == "__main__":
    main()
