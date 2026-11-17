"""Independently spot-check 100–200 ns source bytes against Zenodo ranges.

The three fixed positions per trajectory span the byte region containing
frames 1000–1999. This is an interim check, not a full author MD5 check.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).with_name("second_window_range_audit.json")
RECORDS = {"no_ligand": 14060756, "iag": 14060604}
STARTS = (300_000_000, 430_000_000, 550_000_000)
LENGTH = 1_000_000


def check(condition: str, replica: int, start: int) -> dict:
    source = ROOT / "datasets" / condition / (
        "replica_00.clean.xtc" if condition == "no_ligand" and replica == 0
        else f"replica_{replica:02d}.xtc"
    )
    end = start + LENGTH - 1
    if source.stat().st_size <= end:
        raise ValueError(f"Local source lacks requested bytes: {source}")
    url = (f"https://zenodo.org/records/{RECORDS[condition]}/files/"
           f"step7_filtered_skip10_Set_0_0_{replica}.xtc?download=1")
    with tempfile.TemporaryDirectory(prefix="gpr6_range_") as directory:
        data_path = Path(directory) / "data.bin"
        header_path = Path(directory) / "headers.txt"
        command = ["curl", "--location", "--fail", "--silent", "--show-error",
                   "--retry", "3", "--retry-delay", "3", "--range", f"{start}-{end}",
                   "--dump-header", str(header_path), "--output", str(data_path), url]
        subprocess.run(command, check=True)
        headers = header_path.read_text(errors="replace")
        statuses = re.findall(r"^HTTP/\S+\s+(\d+)", headers, flags=re.MULTILINE | re.IGNORECASE)
        content_ranges = re.findall(r"^content-range:\s*bytes\s+(\d+)-(\d+)/(\d+)",
                                    headers, flags=re.MULTILINE | re.IGNORECASE)
        if not statuses or statuses[-1] != "206" or not content_ranges:
            raise ValueError(f"Expected HTTP 206 with Content-Range: {condition}/{replica}, {start}")
        remote_start, remote_end, total = map(int, content_ranges[-1])
        if (remote_start, remote_end) != (start, end):
            raise ValueError(f"Unexpected Content-Range: {condition}/{replica}, {content_ranges[-1]}")
        remote = data_path.read_bytes()
        if len(remote) != LENGTH:
            raise ValueError(f"Unexpected response length: {condition}/{replica}, {len(remote)}")
        with source.open("rb") as handle:
            handle.seek(start)
            local = handle.read(LENGTH)
        if remote != local:
            raise ValueError(f"Zenodo byte mismatch: {condition}/{replica}, {start}-{end}")
    return {"condition": condition, "replica": replica, "start": start, "end": end,
            "http_status": 206, "source_total_bytes": total, "matched_bytes": LENGTH,
            "sample_md5": hashlib.md5(remote).hexdigest()}


def main() -> None:
    tasks = [(condition, replica, start) for condition in RECORDS
             for replica in range(6) for start in STARTS]
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(check, *task) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"matched {record['condition']}/{record['replica']:02d} @{record['start']}", flush=True)
    records.sort(key=lambda item: (item["condition"], item["replica"], item["start"]))
    OUTPUT.write_text(json.dumps({"method": __doc__.strip(), "samples_checked": len(records),
                                  "bytes_compared": len(records) * LENGTH,
                                  "full_author_md5_verified": False, "samples": records}, indent=2) + "\n")
    print(f"Audited {len(records)} source regions: {OUTPUT}")


if __name__ == "__main__":
    main()
