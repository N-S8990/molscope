"""Resume the 12 author-deposited GPR6 production trajectories from Zenodo.

Run from the repository root. Sources and checksums are listed in
docs/validation/gpr6-source-provenance.md. This script preserves partial files
on interruption so the next invocation can resume them.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1] / "datasets"
RECORDS = {"no_ligand": 14060756, "iag": 14060604}
EXPECTED_MD5 = {
    "no_ligand": (
        "1b7333a821f224e0831713ae046be26c", "fc34ba5e1114d16215822be6e1516933",
        "04cb508685fa22942e405f67e81af1ed", "818f71ef45f3c0df602b6ad285055b3e",
        "e543d0a932bf73b778d1b8780e593f54", "33c3c0a288a95b3a09ea2cd6117647f2",
    ),
    "iag": (
        "40adfd0b1fed9f3a36d8a7051c70f38c", "aa1e2c834c7489a84e39176cd092e4ba",
        "2b687ec4b6952101f7326d3d419438af", "073c55a12276d5c16358c8873f8b6e27",
        "dc0293bfb46523bbc0cd7133c474e174", "8d259a6e964bdd5405f0fc843261e9e9",
    ),
}


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(condition: str, replica: int) -> tuple[str, int, int]:
    destination = ROOT / condition / f"replica_{replica:02d}.xtc"
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = (f"https://zenodo.org/records/{RECORDS[condition]}/files/"
           f"step7_filtered_skip10_Set_0_0_{replica}.xtc?download=1")
    command = ["curl", "--location", "--fail", "--retry", "5", "--retry-delay", "10",
               "--continue-at", "-", "--silent", "--show-error", "--output", str(destination), url]
    if destination.exists() and md5(destination) == EXPECTED_MD5[condition][replica]:
        return condition, replica, destination.stat().st_size
    for attempt in range(1, 101):
        result = subprocess.run(command, check=False)
        if result.returncode == 0:
            if md5(destination) != EXPECTED_MD5[condition][replica]:
                raise ValueError(f"Downloaded file has wrong author MD5: {destination}")
            break
        print(f"retrying {condition}/{replica:02d} after curl exit {result.returncode}; attempt {attempt}", flush=True)
        time.sleep(min(60, 5 * attempt))
    else:
        raise RuntimeError(f"Failed to download {condition}/{replica:02d} after 100 attempts")
    return condition, replica, destination.stat().st_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude", action="append", default=[],
                        help="Skip an active destination, e.g. iag:5")
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    # One resumable task per published replica. Do not run this script at the
    # same time as another downloader writing the same destinations.
    excluded = set(args.exclude)
    tasks = [(condition, replica) for condition in RECORDS for replica in range(6)
             if f"{condition}:{replica}" not in excluded]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, *task) for task in tasks]
        for future in as_completed(futures):
            print("completed", *future.result(), flush=True)
