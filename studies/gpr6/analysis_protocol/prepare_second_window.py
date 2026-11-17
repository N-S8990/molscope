"""Freeze and validate published GPR6 frames 1000–1999 from growing downloads.

The source is copied into a fixed-size snapshot before MDAnalysis opens it.
The original 1000-frame prefix hash must match the previously audited first
window, then a byte-exact 2000-frame source prefix is scanned independently.
This is temporal stability analysis of the same replicas, not independent
simulation validation. Full author MD5 checks remain necessary.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from prepare_windows import md5, protein_identity


ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"
FIRST = HERE / "source_prefixes/provenance.json"
SOURCE_PREFIXES = HERE / "source_prefixes_first_200ns"
DERIVED = HERE / "protein_second_100ns_1000_frames"
EXPANDED_PROJECT = HERE / "protein_first_100ns_1000_frames/project_expanded.json"
BASELINE_PROJECT = HERE / "protein_first_100ns_1000_frames/project.json"
SNAPSHOT_BYTES = 650_000_000
TOPOLOGY_MD5 = {"no_ligand": "2911c38cd4355fb670d2ee6b3466d678",
                "iag": "a24e25de2ae8404c7accc248c0e9e6fe"}


def md5_prefix(path: Path, count: int) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        remaining = count
        while remaining:
            block = handle.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"Source ended before {count} bytes: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def copy_bytes(source: Path, target: Path, count: int) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, target.open("wb") as writer:
        remaining = count
        while remaining:
            block = reader.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError(f"Source ended before {count} bytes: {source}")
            writer.write(block)
            remaining -= len(block)


def prepare_one(condition: str, index: int, first_record: dict,
                expected_identity: list) -> dict:
    source_name = "replica_00.clean.xtc" if condition == "no_ligand" and index == 0 else f"replica_{index:02d}.xtc"
    source = DATASETS / condition / source_name
    topology = DATASETS / condition / "topology.pdb"
    if md5(topology) != TOPOLOGY_MD5[condition]:
        raise ValueError(f"Author topology MD5 mismatch: {topology}")
    if source.stat().st_size < SNAPSHOT_BYTES:
        raise ValueError(f"Need at least {SNAPSHOT_BYTES} source bytes: {source}")
    prefix = SOURCE_PREFIXES / condition / f"replica_{index:02d}.xtc"
    snapshot = prefix.with_suffix(".snapshot.xtc")
    if not (prefix.exists() and md5_prefix(prefix, first_record["prefix_bytes"]) == first_record["prefix_md5"]):
        copy_bytes(source, snapshot, SNAPSHOT_BYTES)
        try:
            if md5_prefix(snapshot, first_record["prefix_bytes"]) != first_record["prefix_md5"]:
                raise ValueError(f"First-window source identity mismatch: {source}")
            universe = mda.Universe(str(topology), str(snapshot))
            reader = universe.trajectory
            if len(reader) < 2001:
                raise ValueError(f"Snapshot has fewer than 2001 frames: {source}")
            times = [float(reader[i].time) for i in (0, 1000, 1999)]
            if not np.allclose(times, [0.0, 100000.0, 199900.0], atol=0.01, rtol=0):
                raise ValueError(f"Unexpected source frame times: {times}")
            end_offset = int(reader._xdr.offsets[2000])
            reader.close()
            copy_bytes(snapshot, prefix, end_offset)
        finally:
            snapshot.unlink(missing_ok=True)
            for suffix in ("npz", "lock"):
                snapshot.with_name(f".{snapshot.name}_offsets.{suffix}").unlink(missing_ok=True)
    universe = mda.Universe(str(topology), str(prefix))
    protein = universe.select_atoms("protein")
    alpha = universe.select_atoms("protein and name CA")
    if protein_identity(protein) != expected_identity or len(universe.trajectory) != 2000:
        raise ValueError(f"Protein identity or 2000-frame prefix mismatch: {prefix}")
    maximum_ca_step = 0.0
    for frame in universe.trajectory:
        if not np.isfinite(frame.positions).all():
            raise ValueError(f"Nonfinite source coordinates: {prefix} frame {frame.frame}")
        maximum_ca_step = max(maximum_ca_step, float(np.linalg.norm(np.diff(alpha.positions, axis=0), axis=1).max()))
    if maximum_ca_step > 5.0:
        raise ValueError(f"Receptor Cα chain break: {prefix} max {maximum_ca_step:.3f} Å")
    destination = DERIVED / condition / f"replica_{index:02d}.xtc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with mda.Writer(str(destination), n_atoms=len(protein), format="XTC") as writer:
        for frame in universe.trajectory[1000:2000]:
            writer.write(protein)
    universe.trajectory.close()
    derived = mda.Universe(str(DERIVED / "protein.pdb"), str(destination))
    if len(derived.trajectory) != 1000 or not np.allclose(
        [derived.trajectory[i].time for i in (0, 999)], [100000.0, 199900.0], atol=0.01, rtol=0
    ):
        raise ValueError(f"Derived second window frame/time mismatch: {destination}")
    derived.trajectory.close()
    return {"condition": condition, "replica": index, "source": str(source),
            "source_size_observed_after_projection": source.stat().st_size,
            "snapshot_bytes_copied": SNAPSHOT_BYTES, "source_prefix": str(prefix),
            "source_prefix_bytes": prefix.stat().st_size, "source_prefix_md5": md5(prefix),
            "first_window_prefix_md5_reverified": first_record["prefix_md5"],
            "source_frames_scanned": 2000, "maximum_adjacent_ca_angstrom": maximum_ca_step,
            "derived": str(destination), "derived_frames": 1000,
            "derived_time_ps": [100000.0, 199900.0], "derived_md5": md5(destination),
            "full_author_md5_verified": False}


def main() -> None:
    first = json.loads(FIRST.read_text())
    if len(first) != 12:
        raise ValueError("First-window provenance must contain 12 replicas")
    base = json.loads(EXPANDED_PROJECT.read_text())
    common_topology = Path(base["systems"][0]["topology"])
    DERIVED.mkdir(parents=True, exist_ok=True)
    copy_bytes(common_topology, DERIVED / "protein.pdb", common_topology.stat().st_size)
    expected = protein_identity(mda.Universe(str(common_topology)).select_atoms("protein"))
    records = []
    for condition in ("no_ligand", "iag"):
        for index in range(6):
            key = f"{condition}/replica_{index:02d}"
            record = prepare_one(condition, index, first[key], expected)
            records.append(record)
            print(f"Verified second window: {key}", flush=True)
    for original, output, project_id in (
        (EXPANDED_PROJECT, "project_expanded.json", "gpr6_public_expanded_motifs_second_100ns_temporal_check"),
        (BASELINE_PROJECT, "project_baseline.json", "gpr6_public_39_features_second_100ns_temporal_check"),
    ):
        project = json.loads(original.read_text())
        for ensemble in project["systems"][0]["ensembles"]:
            condition = ensemble["ensemble_id"]
            for index, replica in enumerate(ensemble["replicas"]):
                replica["trajectory"] = str(DERIVED / condition / f"replica_{index:02d}.xtc")
        project["systems"][0]["topology"] = str(DERIVED / "protein.pdb")
        project["project_id"] = project_id
        (DERIVED / output).write_text(json.dumps(project, indent=2) + "\n")
    (DERIVED / "provenance.json").write_text(json.dumps({
        "interpretation": "same-replica temporal stability check, not independent validation",
        "window_frames": [1000, 1999], "window_time_ps": [100000.0, 199900.0],
        "source_full_author_md5_verified": False, "sources": records,
    }, indent=2) + "\n")
    print(f"Prepared 12 verified second-window trajectories: {DERIVED}", flush=True)


if __name__ == "__main__":
    main()
