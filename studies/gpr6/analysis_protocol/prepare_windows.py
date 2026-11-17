"""Prepare a fixed-window, receptor-only 0–100 ns GPR6 comparison.

Full mode requires the author MD5 for every source XTC/PDB before it creates
derived inputs. ``--baseline-test`` only verifies the extraction path on the
local recovered no-ligand prefix; it cannot produce a condition comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import MDAnalysis as mda
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
DERIVED = Path(__file__).resolve().parent / "protein_first_100ns_1000_frames"
PREFIXES = Path(__file__).resolve().parent / "source_prefixes"
SOURCE_MD5 = {
    "no_ligand": [
        "1b7333a821f224e0831713ae046be26c", "fc34ba5e1114d16215822be6e1516933",
        "04cb508685fa22942e405f67e81af1ed", "818f71ef45f3c0df602b6ad285055b3e",
        "e543d0a932bf73b778d1b8780e593f54", "33c3c0a288a95b3a09ea2cd6117647f2",
    ],
    "iag": [
        "40adfd0b1fed9f3a36d8a7051c70f38c", "aa1e2c834c7489a84e39176cd092e4ba",
        "2b687ec4b6952101f7326d3d419438af", "073c55a12276d5c16358c8873f8b6e27",
        "dc0293bfb46523bbc0cd7133c474e174", "8d259a6e964bdd5405f0fc843261e9e9",
    ],
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def protein_identity(atoms) -> list[tuple[str, str, int, str]]:
    return [(atom.name, atom.resname, int(atom.resid), atom.segid) for atom in atoms]


def extract_window(topology: Path, trajectory: Path, destination: Path, expected_identity=None) -> dict:
    universe = mda.Universe(str(topology), str(trajectory))
    protein = universe.select_atoms("protein")
    identity = protein_identity(protein)
    if expected_identity is not None and identity != expected_identity:
        raise ValueError(f"Protein atom order differs in {topology}")
    if len(universe.trajectory) < 1000:
        raise ValueError(f"Fewer than 1000 indexed frames in {trajectory}")
    try:
        times = [float(universe.trajectory[index].time) for index in (0, 1, 999)]
        if not np.allclose(times, [0.0, 100.0, 99900.0], atol=0.01, rtol=0):
            raise ValueError(f"Expected 0, 100, 99900 ps at frames 0, 1, 999: {times}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        with mda.Writer(str(destination), n_atoms=len(protein), format="XTC") as writer:
            for ts in universe.trajectory[:1000]:
                writer.write(protein)
    finally:
        universe.trajectory.close()
    return {"source_topology": str(topology), "source_trajectory": str(trajectory),
            "derived_trajectory": str(destination), "frames": 1000,
            "source_frame_indices": [0, 999], "time_ps": [0.0, 99900.0],
            "protein_atoms": len(identity), "derived_md5": md5(destination)}


def project_payload(topology: Path, report: list[dict]) -> dict:
    assignments = {}
    ensembles = []
    for condition, target in (("no_ligand", "no_added_inverse_agonist"), ("iag", "3h_bound")):
        replicas = []
        for index in range(6):
            trajectory_id = f"GPR6:{condition}:replica_{index:02d}"
            assignments[trajectory_id] = target
            replicas.append({"replica_id": f"replica_{index:02d}",
                             "trajectory": str(DERIVED / condition / f"replica_{index:02d}.xtc")})
        ensembles.append({"ensemble_id": condition, "replicas": replicas})
    return {
        "schema_version": "1", "project_id": "gpr6_public_3h_vs_no_added_inverse_agonist_first_100ns",
        "systems": [{"system_id": "GPR6", "topology": str(topology), "ensembles": ensembles}],
        "labels": [{"label_id": "author_deposited_condition", "label_type": "categorical",
                    "level": "trajectory", "source": "external_annotation",
                    "description": "Author-deposited GPR6 no-added-inverse-agonist versus 3h Set 3; six independent replicas per class.",
                    "assignments": assignments}],
        "prediction_tasks": [{"task_id": "condition_classification", "task_type": "classification",
                              "target_label_id": "author_deposited_condition"}],
        "analysis": {"generic_feature_parameters": {"frame_stride": 1, "use_pbc": False},
                     "plugins": ["protein.ca_distances_contacts"],
                     "plugin_parameters": {"protein.ca_distances_contacts": {
                         "selection": "protein and name CA and resid 150 151 152 153 290 291 292 293",
                         "contact_cutoff_angstrom": 8.0,
                         "minimum_sequence_separation": 3,
                         "max_pairs": 18}}},
    }


def snapshot_prefix(topology: Path, growing_source: Path, destination: Path) -> dict:
    """Read a bounded snapshot, then save a byte-exact, fully verified XTC prefix.

    The growing source is never opened by MDAnalysis. Its snapshot may contain
    an incomplete final frame; only the complete first 1000 frames are kept.
    """
    snapshot = destination.with_suffix(".snapshot.xtc")
    size = growing_source.stat().st_size
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    with growing_source.open("rb") as source, snapshot.open("wb") as target:
        remaining = size
        while remaining:
            block = source.read(min(8 * 1024 * 1024, remaining))
            if not block:
                raise ValueError("Source shrank while taking snapshot")
            target.write(block)
            remaining -= len(block)
    try:
        universe = mda.Universe(str(topology), str(snapshot))
        reader = universe.trajectory
        if len(reader) < 1001:
            raise ValueError("Snapshot has fewer than 1001 indexed frames; retry after growth")
        end_offset = int(reader._xdr.offsets[1000])
        if not np.allclose([reader[i].time for i in (0, 1, 999)], [0, 100, 99900], atol=0.01, rtol=0):
            raise ValueError("Unexpected source frame times")
        snapshot_protein = universe.select_atoms("protein")
        for frame in reader[:1000]:
            if not np.isfinite(snapshot_protein.positions).all():
                raise ValueError(f"Nonfinite receptor coordinates at frame {frame.frame}")
        reader.close()
        with snapshot.open("rb") as source, destination.open("wb") as target:
            remaining = end_offset
            while remaining:
                block = source.read(min(8 * 1024 * 1024, remaining))
                if not block:
                    raise ValueError("Snapshot ended before frame-1000 offset")
                target.write(block)
                remaining -= len(block)
        independent = mda.Universe(str(topology), str(destination))
        if len(independent.trajectory) != 1000:
            raise ValueError("Saved prefix does not contain exactly 1000 frames")
        ca = independent.select_atoms("protein and name CA")
        worst_adjacent_ca = 0.0
        for frame in independent.trajectory:
            if not np.isfinite(frame.positions).all():
                raise ValueError(f"Nonfinite saved frame {frame.frame}")
            adjacent = np.linalg.norm(np.diff(ca.positions, axis=0), axis=1)
            worst_adjacent_ca = max(worst_adjacent_ca, float(adjacent.max()))
            if (adjacent > 5.0).any():
                raise ValueError(f"Receptor Cα backbone is split or distorted at frame {frame.frame}")
        if not np.isclose(independent.trajectory[999].time, 99900, atol=0.01):
            raise ValueError("Saved prefix terminal time is not 99900 ps")
        independent.trajectory.close()
        return {"source_path": str(growing_source), "source_size_at_snapshot_bytes": size,
                "prefix_path": str(destination), "prefix_bytes": end_offset,
                "prefix_md5": md5(destination), "frames_verified": 1000,
                "maximum_adjacent_ca_distance_angstrom": worst_adjacent_ca,
                "full_source_md5_verified": False}
    finally:
        snapshot.unlink(missing_ok=True)
        for suffix in ("npz", "lock"):
            snapshot.with_name(f".{snapshot.name}_offsets.{suffix}").unlink(missing_ok=True)


def prepare_growing_sources(topologies: dict[str, Path], identities: dict[str, list]) -> None:
    """Wait for independently generated downloads to expose 1000 valid frames."""
    provenance = PREFIXES / "provenance.json"
    PREFIXES.mkdir(parents=True, exist_ok=True)
    records = json.loads(provenance.read_text()) if provenance.exists() else {}
    for condition in SOURCE_MD5:
        for index in range(6):
            key = f"{condition}/replica_{index:02d}"
            source = DATASETS / condition / f"replica_{index:02d}.xtc"
            prefix = PREFIXES / condition / f"replica_{index:02d}.xtc"
            if key in records and prefix.exists() and md5(prefix) == records[key]["prefix_md5"]:
                continue
            if key == "no_ligand/replica_00":
                raise ValueError(
                    "no_ligand/replica_00 live resumed bytes diverged from a verified Zenodo range; "
                    "supply an independently validated exact 1000-frame prefix and provenance"
                )
            last_attempt_size = 0
            while True:
                size = source.stat().st_size if source.exists() else 0
                if size >= 280_000_000 and size - last_attempt_size >= 10_000_000:
                    last_attempt_size = size
                    try:
                        record = snapshot_prefix(topologies[condition], source, prefix)
                        records[key] = record
                        provenance.write_text(json.dumps(records, indent=2) + "\n")
                        print(f"Verified 1000-frame source prefix: {key}", flush=True)
                        break
                    except (ValueError, OSError, EOFError, RuntimeError, IndexError) as error:
                        prefix.unlink(missing_ok=True)
                        print(f"Waiting for more bytes: {key}: {error}", flush=True)
                time.sleep(30)
    common_topology = DERIVED / "protein.pdb"
    DERIVED.mkdir(parents=True, exist_ok=True)
    no_ligand = mda.Universe(str(topologies["no_ligand"]))
    no_ligand.select_atoms("protein").write(str(common_topology))
    report = []
    for condition in SOURCE_MD5:
        for index in range(6):
            prefix = PREFIXES / condition / f"replica_{index:02d}.xtc"
            destination = DERIVED / condition / f"replica_{index:02d}.xtc"
            report.append(extract_window(topologies[condition], prefix, destination, identities["no_ligand"]))
    (DERIVED / "provenance.json").write_text(json.dumps({"protocol": "first 1000 strided frames (0–99.9 ns inclusive); receptor-only atom projection", "full_source_md5_verified": False, "sources": report}, indent=2) + "\n")
    (DERIVED / "project.json").write_text(json.dumps(project_payload(common_topology, report), indent=2) + "\n")
    print(f"Prepared 12 receptor-only first-100-ns trajectories: {DERIVED}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-test", action="store_true")
    parser.add_argument("--from-growing-downloads", action="store_true")
    args = parser.parse_args()
    if args.baseline_test:
        topology = ROOT / "inputs/step5_input_Set_0_0_1.pdb"
        trajectory = ROOT / "preparation/gpr6_first_1831_frames.xtc"
        destination = Path(__file__).resolve().parent / "baseline_100ns_test/protein.xtc"
        protein_topology = destination.parent / "protein.pdb"
        destination.parent.mkdir(parents=True, exist_ok=True)
        source = mda.Universe(str(topology))
        source.select_atoms("protein").write(str(protein_topology))
        report = extract_window(topology, trajectory, destination)
        universe = mda.Universe(str(protein_topology), str(destination))
        assert len(universe.trajectory) == 1000
        assert len(universe.select_atoms("protein")) == 4273
        universe.trajectory.close()
        print(json.dumps(report, indent=2))
        return

    topologies = {condition: DATASETS / condition / "topology.pdb" for condition in SOURCE_MD5}
    expected_pdb_md5 = {"no_ligand": "2911c38cd4355fb670d2ee6b3466d678",
                        "iag": "a24e25de2ae8404c7accc248c0e9e6fe"}
    for condition, topology in topologies.items():
        if not topology.is_file() or md5(topology) != expected_pdb_md5[condition]:
            raise ValueError(f"Missing or author-MD5-mismatched {topology}")
    source_paths = [(condition, index, DATASETS / condition / f"replica_{index:02d}.xtc")
                    for condition in SOURCE_MD5 for index in range(6)]
    identities = {}
    for condition, topology in topologies.items():
        universe = mda.Universe(str(topology))
        identities[condition] = protein_identity(universe.select_atoms("protein"))
    if identities["no_ligand"] != identities["iag"]:
        raise ValueError("Condition protein atom identities differ")
    if args.from_growing_downloads:
        prepare_growing_sources(topologies, identities)
        return
    for condition, index, trajectory in source_paths:
        if not trajectory.is_file() or md5(trajectory) != SOURCE_MD5[condition][index]:
            raise ValueError(f"Missing/incomplete or author-MD5-mismatched {trajectory}")
    DERIVED.mkdir(parents=True, exist_ok=True)
    common_topology = DERIVED / "protein.pdb"
    no_ligand = mda.Universe(str(topologies["no_ligand"]))
    no_ligand.select_atoms("protein").write(str(common_topology))
    report = []
    for condition, index, trajectory in source_paths:
        destination = DERIVED / condition / f"replica_{index:02d}.xtc"
        report.append(extract_window(topologies[condition], trajectory, destination, identities["no_ligand"]))
    (DERIVED / "provenance.json").write_text(json.dumps({"protocol": "first 1000 strided frames (0–99.9 ns inclusive); receptor-only atom projection", "full_source_md5_verified": True, "sources": report}, indent=2) + "\n")
    (DERIVED / "project.json").write_text(json.dumps(project_payload(common_topology, report), indent=2) + "\n")
    print(f"Prepared {len(report)} receptor-only 0–100 ns trajectories: {DERIVED}")


if __name__ == "__main__":
    main()
