"""Readable descriptive outputs for an extracted trajectory session."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import pandas as pd

from molscope.io.storage import read_table


def write_extraction_report(session: str | Path) -> Path:
    root = Path(session)
    storage = json.loads((root / "session_storage.json").read_text())
    features = read_table(root / storage["features"])
    manifest = read_table(root / storage["manifest"])
    if len(features) != len(manifest):
        raise ValueError("Feature and manifest row counts differ.")
    features.to_csv(root / "frame_features.csv", index=False)
    manifest.to_csv(root / "frame_manifest.csv", index=False)
    summary = features.describe().T
    summary.to_csv(root / "feature_summary.csv")
    figures = []
    # Individual trajectories remain separate, including time axes and maps.
    for number, (trajectory, group) in enumerate(manifest.groupby("trajectory_id", sort=False), 1):
        frame = features.iloc[group.index]
        time = group["time_ps"].to_numpy()
        geometry = [c for c in frame if c.startswith(("GEOM_global", "DYN_kabsch"))]
        if geometry:
            fig, axes = plt.subplots(len(geometry), 1, figsize=(10, 3 * len(geometry)), squeeze=False)
            for axis, column in zip(axes[:, 0], geometry):
                axis.plot(time, frame[column])
                axis.set(xlabel="Time (ps)", ylabel=column.replace("_", " "))
            fig.suptitle(str(trajectory))
            fig.tight_layout()
            name = f"trajectory_{number}_geometry.png"
            fig.savefig(root / name, dpi=160); plt.close(fig)
            figures.append((name, "Geometry over sampled frames"))
        columns = [c for c in frame if c.startswith("CA_CONTACT__")]
        pairs = []
        for column in columns:
            parts = column.removeprefix("CA_CONTACT__").split("__")
            if len(parts) == 4:
                pairs.append((column, ":".join(parts[:2]), ":".join(parts[2:])))
        if pairs:
            residues = list(dict.fromkeys(r for _, left, right in pairs for r in (left, right)))
            residues.sort(key=lambda r: (r.split(":")[0], int(r.rsplit("_", 1)[1])))
            lookup = {r: i for i, r in enumerate(residues)}
            matrix = np.full((len(residues), len(residues)), np.nan)
            for column, left, right in pairs:
                matrix[lookup[left], lookup[right]] = matrix[lookup[right], lookup[left]] = frame[column].mean()
            pd.DataFrame(matrix, index=residues, columns=residues).to_csv(root / f"trajectory_{number}_contact_occupancy.csv")
            fig, axis = plt.subplots(figsize=(10, 9))
            cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("lightgray")
            im = axis.imshow(matrix, vmin=0, vmax=1, cmap=cmap)
            ticks = np.arange(0, len(residues), max(1, len(residues) // 16))
            axis.set(xticks=ticks, yticks=ticks, xticklabels=[residues[i] for i in ticks],
                     yticklabels=[residues[i] for i in ticks], title="CA contact occupancy; gray = unmeasured/excluded pairs")
            plt.setp(axis.get_xticklabels(), rotation=90)
            fig.colorbar(im, ax=axis, label="Fraction of sampled frames")
            fig.tight_layout()
            name = f"trajectory_{number}_contacts.png"
            fig.savefig(root / name, dpi=160); plt.close(fig)
            figures.append((name, "Residue contact occupancy"))
        helix = [c for c in frame if c.startswith("DSSP_helix__")]
        if helix:
            identifiers = [c.removeprefix("DSSP_helix__") for c in helix]
            matrix = np.stack([frame[f"DSSP_helix__{r}"].to_numpy() +
                               2 * frame[f"DSSP_sheet__{r}"].to_numpy() for r in identifiers])
            fig, axis = plt.subplots(figsize=(12, 7))
            im = axis.imshow(matrix, aspect="auto", interpolation="nearest", vmin=-.5, vmax=2.5,
                             cmap=ListedColormap(["#bdbdbd", "#d95f02", "#1b9e77"]))
            ticks = np.linspace(0, len(time)-1, min(6, len(time)), dtype=int)
            axis.set(xticks=ticks, xticklabels=[f"{time[i]:g}" for i in ticks],
                     xlabel="Sampled time (ps)", ylabel="Residue row (mapping in extraction_evidence.json)", title="DSSP secondary structure")
            fig.colorbar(im, ax=axis, ticks=[0, 1, 2]).ax.set_yticklabels(["Coil", "Helix", "Sheet"])
            fig.tight_layout()
            name = f"trajectory_{number}_dssp.png"
            fig.savefig(root / name, dpi=160); plt.close(fig)
            figures.append((name, "Secondary structure across residues and time"))
    content = ["<!doctype html><meta charset='utf-8'><title>MolScope extraction report</title>",
               "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}img{max-width:100%}table{border-collapse:collapse}td,th{padding:8px}</style>",
               "<h1>MolScope trajectory extraction</h1>",
               f"<p>{len(features)} sampled frames; {features.shape[1]} measured feature columns; {manifest.trajectory_id.nunique()} trajectory/trajectories.</p>",
               "<p>Descriptive extraction report. Independent-replica stability, supervised prediction and SAM-MFP were not run.</p>",
               "<p><a href='frame_features.csv'>Feature CSV</a> · <a href='frame_manifest.csv'>Frame identity CSV</a> · <a href='feature_summary.csv'>Summary CSV</a> · <a href='extraction_evidence.json'>Applicability and provenance</a></p>"]
    for name, title in figures:
        content.append(f"<h2>{html.escape(title)}</h2><img src='{name}' alt='{html.escape(title)}'>")
    target = root / "extraction_report.html"
    target.write_text("\n".join(content), encoding="utf-8")
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    print(write_extraction_report(args.session))


if __name__ == "__main__":
    main()
