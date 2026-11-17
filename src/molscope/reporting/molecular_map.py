"""Map feature evidence onto real receptor coordinates without claiming causality."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from tempfile import TemporaryDirectory

import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import plotly.graph_objects as go

_RESIDUE = re.compile(r"__(?:X|chain_X)__([A-Za-z0-9]+)_(\d+)")


def _ca_positions(topology: Path) -> dict[int, tuple[str, np.ndarray]]:
    atoms = mda.Universe(str(topology)).select_atoms("protein and name CA")
    positions: dict[int, tuple[str, np.ndarray]] = {}
    for atom in atoms:
        number = int(atom.resid)
        if number in positions:
            raise ValueError(f"Duplicate protein C-alpha residue number {number}; select one receptor chain.")
        positions[number] = (str(atom.resname), atom.position.astype(float).copy())
    if len(positions) < 3:
        raise ValueError("Topology must contain at least three uniquely numbered protein C-alpha atoms.")
    return positions


def _chemical_map(path: Path, positions: dict[int, tuple[str, np.ndarray]]) -> dict[str, tuple[int, ...]]:
    """Use the declared atom-group definitions for chemistry feature locations."""
    records = json.loads(path.read_text(encoding="utf-8"))["features"]
    mapping = {}
    for record in records:
        name = record["feature"]
        ids = tuple(int(number) for number in record["residue_ids"])
        if not name.startswith("CHEM_") or len(ids) != 2 or name in mapping:
            raise ValueError("Invalid or duplicate chemical feature definition.")
        for group_key, number in zip(("left_group", "right_group"), ids):
            match = re.match(r"^([A-Z]{3})(\d+)", record[group_key])
            if match is None or int(match.group(2)) != number or number not in positions or positions[number][0].upper() != match.group(1):
                raise ValueError(f"Chemical feature {name} does not match receptor residue {number}.")
        mapping[name] = ids
    return mapping


def _mapping(name: str, positions: dict[int, tuple[str, np.ndarray]],
             chemical: dict[str, tuple[int, ...]] | None = None) -> tuple[int, ...]:
    if name.startswith("CHEM_"):
        return chemical.get(name, ()) if chemical else ()
    grouped_sidechain = re.fullmatch(r"SIDECHAIN_residue_([A-Z]{3})_(\d+)", name)
    if grouped_sidechain:
        residue_name, raw_number = grouped_sidechain.groups()
        number = int(raw_number)
        return (number,) if number in positions and positions[number][0].upper() == residue_name else ()
    matches = _RESIDUE.findall(name)
    if not matches or len(matches) > 2:
        return ()
    numbers = tuple(int(number) for _, number in matches)
    if len(set(numbers)) != len(numbers):
        return ()
    for residue_name, raw_number in matches:
        number = int(raw_number)
        if number not in positions or positions[number][0].upper() != residue_name.upper():
            return ()
    return numbers


def _color(value: float, scale: float) -> str:
    rgba = plt.get_cmap("coolwarm")(0.5 + 0.5 * float(np.clip(value / scale, -1, 1)))
    return colors.to_hex(rgba)


def _render(topology: Path, positions, evidence: pd.DataFrame, output: Path, *, title: str,
            colorbar: str, interpretation: str) -> None:
    per_residue: dict[int, list[tuple[str, float]]] = {number: [] for number in positions}
    for row in evidence.itertuples(index=False):
        for number in row.residues:
            per_residue[number].append((row.feature, row.score))
    # The strongest absolute mapped feature determines each residue's color.
    selected = {number: max(items, key=lambda item: abs(item[1])) if items else ("", 0.0)
                for number, items in per_residue.items()}
    scale = max((abs(score) for _, score in selected.values()), default=0.0) or 1.0
    numbers = list(positions)
    xyz = np.array([positions[number][1] for number in numbers])
    scores = np.array([selected[number][1] for number in numbers])
    hovers = [f"{positions[number][0]} {number}<br>Mapped features: {len(per_residue[number])}<br>Strongest: {selected[number][0] or 'none'}<br>Score: {scores[i]:+.4g}"
              for i, number in enumerate(numbers)]
    figure = go.Figure()
    figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2],
                                  mode="lines", line=dict(color="rgba(130,130,130,0.55)", width=5),
                                  name="Receptor C-alpha trace", hoverinfo="skip"))
    figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode="markers",
                                  marker=dict(size=4, color=scores, cmin=-scale, cmax=scale,
                                              colorscale="RdBu_r", colorbar=dict(title=colorbar)),
                                  text=hovers, hovertemplate="%{text}<extra></extra>",
                                  name="Residue evidence"))
    for row in evidence.itertuples(index=False):
        if len(row.residues) != 2:
            continue
        left, right = [positions[number][1] for number in row.residues]
        figure.add_trace(go.Scatter3d(x=[left[0], right[0]], y=[left[1], right[1]], z=[left[2], right[2]],
                                      mode="lines", line=dict(color=_color(row.score, scale), width=3),
                                      name=row.feature, showlegend=False,
                                      hovertemplate=f"{row.feature}<br>Score: {row.score:+.4g}<extra></extra>"))
    figure.update_layout(title=f"{title}<br><sup>{interpretation}</sup>",
                         scene=dict(xaxis_title="x (Å)", yaxis_title="y (Å)", zaxis_title="z (Å)",
                                    aspectmode="data"), margin=dict(l=0, r=0, t=85, b=0),
                         template="plotly_white")
    figure.write_html(output.with_suffix(".html"), include_plotlyjs=True, full_html=True)

    static = plt.figure(figsize=(11, 8))
    axis = static.add_subplot(111, projection="3d")
    axis.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color="0.7", lw=0.8)
    norm = colors.TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    points = axis.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=scores, cmap="coolwarm", norm=norm, s=14)
    for number in numbers:
        if per_residue[number]:
            px, py, pz = positions[number][1]
            axis.text(px, py, pz, f"{positions[number][0]}{number}", fontsize=7)
    for row in evidence.itertuples(index=False):
        if len(row.residues) == 2:
            segment = np.array([positions[number][1] for number in row.residues])
            axis.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=_color(row.score, scale), lw=1.4, alpha=0.8)
    static.colorbar(points, ax=axis, shrink=0.65, pad=0.07, label=colorbar)
    axis.set(xlabel="x (Å)", ylabel="y (Å)", zlabel="z (Å)")
    axis.set_title(title, pad=20)
    static.text(0.02, 0.02, interpretation, fontsize=9, wrap=True)
    static.tight_layout()
    static.savefig(output.with_suffix(".png"), dpi=170, bbox_inches="tight")
    plt.close(static)

    try:
        import pymol2
    except ImportError:
        return
    with TemporaryDirectory(prefix="molscope-pymol-") as temporary:
        raw = Path(temporary) / "receptor.png"
        pocket_raw = Path(temporary) / "pocket.png"
        viewer = pymol2.PyMOL()
        viewer.start()
        try:
            cmd = viewer.cmd
            cmd.load(str(topology), "receptor")
            cmd.hide("everything")
            cmd.show("cartoon", "receptor")
            cmd.bg_color("white")
            cmd.alter("receptor", "b=0.0")
            for number, (_, score) in selected.items():
                if per_residue[number]:
                    cmd.alter(f"receptor and resi {number}", f"b={float(score)}")
            cmd.spectrum("b", "blue_white_red", "receptor", minimum=-scale, maximum=scale)
            selection = " or ".join(f"resi {number}" for number in numbers if per_residue[number])
            if selection:
                cmd.select("mapped", f"receptor and ({selection})")
                cmd.show("sticks", "mapped and not name CA+C+N+O")
            cmd.orient("receptor")
            cmd.zoom("receptor", buffer=7)
            cmd.png(str(raw), width=1300, height=1000, ray=1)
            if selection:
                cmd.set("cartoon_transparency", 0.6, "receptor and not mapped")
                cmd.show("spheres", "mapped and name CA")
                cmd.set("sphere_scale", 0.35)
                cmd.orient("mapped")
                cmd.zoom("mapped", buffer=11)
                cmd.png(str(pocket_raw), width=1300, height=1000, ray=1)
        finally:
            viewer.stop()
        cartoon = plt.figure(figsize=(13, 10))
        axis = cartoon.add_subplot(111)
        axis.imshow(plt.imread(raw))
        axis.axis("off")
        axis.set_title(title + " — receptor cartoon", fontsize=15)
        sm = plt.cm.ScalarMappable(norm=norm, cmap="coolwarm")
        cartoon.colorbar(sm, ax=axis, shrink=0.6, pad=0.01, label=colorbar)
        cartoon.text(0.02, 0.02, interpretation, fontsize=9, wrap=True)
        cartoon.savefig(output.with_name(output.stem + "_cartoon.png"), dpi=160, bbox_inches="tight")
        plt.close(cartoon)
        if pocket_raw.is_file():
            pocket = plt.figure(figsize=(13, 10))
            pocket_axis = pocket.add_subplot(111)
            pocket_axis.imshow(plt.imread(pocket_raw))
            pocket_axis.axis("off")
            pocket_axis.set_title(title + " — mapped region", fontsize=15)
            pocket.colorbar(sm, ax=pocket_axis, shrink=0.6, pad=0.01, label=colorbar)
            motif = [number for number in (152, 292) if number in per_residue and per_residue[number]]
            if motif:
                handles = [Line2D([0], [0], marker="o", color="none",
                                  markerfacecolor=_color(selected[number][1], scale), markersize=10,
                                  label=f"{positions[number][0]}{number}") for number in motif]
                pocket_axis.legend(handles=handles, loc="lower center", ncol=len(handles),
                                   frameon=True, title="Mapped motif residues")
            pocket.text(0.02, 0.02, interpretation, fontsize=9, wrap=True)
            pocket.savefig(output.with_name(output.stem + "_pocket.png"), dpi=160, bbox_inches="tight")
            plt.close(pocket)


def write_molecular_maps(topology: str | Path, run_dir: str | Path,
                         *, chemical_definitions: str | Path | None = None) -> dict[str, object]:
    """Write actual-coordinate views for mapped supervised and exploratory evidence."""
    topology, root = Path(topology), Path(run_dir)
    positions = _ca_positions(topology)
    definitions_path = Path(chemical_definitions) if chemical_definitions else root / "gpr6_chemical_definitions.json"
    chemical = _chemical_map(definitions_path, positions) if definitions_path.is_file() else {}
    outputs = {}
    specs = [
        (next((name for name in ("augmented_lofo_importance.csv", "supervised_lofo_importance.csv",
                                "chemical_lofo_importance.csv") if (root / name).is_file()), ""),
         "Feature_Group", "Trajectory_Mean_Accuracy_Drop_Mean",
         "gpr6_receptor_predictive_map", "GPR6 predictive feature impact",
         "LOFO accuracy drop", "Positive = prediction worsened on removal; negative = removal helped. Pair links are measured features, not physical bonds."),
        (next((name for name in ("gpr6_augmented_condition_comparison.csv", "gpr6_feature_condition_comparison.csv",
                                "gpr6_chemical_condition_comparison.csv") if (root / name).is_file()), ""),
         "Feature", "BH_Q_Across_39_Features",
         "gpr6_receptor_condition_map", "GPR6 exploratory condition associations",
         "Signed -log10(q)", "Color sign follows condition mean difference; intensity follows exploratory BH q. This does not establish causality."),
    ]
    for filename, feature_col, value_col, stem, title, bar, note in specs:
        source = root / filename
        if not source.is_file():
            continue
        table = pd.read_csv(source)
        if stem.endswith("condition_map") and value_col not in table:
            q_columns = [column for column in table if column.startswith("BH_Q_Across_")]
            if len(q_columns) != 1:
                raise ValueError("Condition map requires one BH-adjusted q-value column.")
            value_col = q_columns[0]
        rows = []
        for record in table.to_dict("records"):
            name = str(record[feature_col])
            numbers = _mapping(name, positions, chemical)
            if not numbers:
                continue
            if stem.endswith("condition_map"):
                q = pd.to_numeric(record.get(value_col), errors="coerce")
                difference_column = next((column for column in table if column.startswith("Three_H_Minus_")), None)
                difference = pd.to_numeric(record.get(difference_column), errors="coerce")
                if not np.isfinite(q) or not np.isfinite(difference):
                    continue
                score = float(np.sign(difference) * min(-np.log10(max(float(q), 1e-12)), 3.0))
            else:
                score = float(pd.to_numeric(record.get(value_col), errors="coerce"))
                if not np.isfinite(score):
                    continue
            rows.append({"feature": name, "residues": numbers, "score": score})
        if not rows:
            continue
        evidence = pd.DataFrame(rows)
        _render(topology, positions, evidence, root / stem, title=title, colorbar=bar, interpretation=note)
        mapping = evidence.explode("residues").rename(columns={"residues": "residue_number", "score": "mapped_score"})
        mapping.to_csv(root / f"{stem}_mapping.csv", index=False)
        outputs[stem] = {"mapped_features": len(evidence), "mapped_residues": mapping.residue_number.nunique(),
                         "html": f"{stem}.html", "png": f"{stem}.png",
                         "cartoon_png": f"{stem}_cartoon.png" if (root / f"{stem}_cartoon.png").is_file() else None,
                         "pocket_png": f"{stem}_pocket.png" if (root / f"{stem}_pocket.png").is_file() else None,
                         "mapping": f"{stem}_mapping.csv"}
    (root / "gpr6_molecular_map_manifest.json").write_text(json.dumps({
        "topology": str(topology.resolve()), "coordinate_kind": "deposited_receptor_CA_positions",
        "chemical_definitions": str(definitions_path.resolve()) if chemical else None,
        "aggregation": "largest absolute mapped feature score per residue; signed value retained",
        "unmapped_features": "Global geometry, DSSP fractions, and unknown/mismatched residue IDs are intentionally omitted",
        "limits": "LOFO is grouped predictive dependence, not statistical or causal significance. Condition q values are exploratory, based on a feature family and independent replica means. Pair links are measurements, not bonds.",
        "outputs": outputs,
    }, indent=2) + "\n")
    if outputs:
        entries = []
        for stem, details in outputs.items():
            links = [f"[{stem}.html]({stem}.html) — interactive 3D structure"]
            if details["cartoon_png"]:
                links.append(f"[{stem}_cartoon.png]({stem}_cartoon.png) — receptor cartoon")
            if details["pocket_png"]:
                links.append(f"[{stem}_pocket.png]({stem}_pocket.png) — mapped-region close-up")
            links.extend([f"[{stem}.png]({stem}.png) — coordinate trace",
                          f"[{stem}_mapping.csv]({stem}_mapping.csv) — feature-to-residue assignments"])
            entries.append("- " + "; ".join(links) + ".")
        (root / "MOLECULAR_MAPS.md").write_text(
            "# GPR6 receptor structure views\n\n"
            "The views use the deposited receptor C-alpha coordinates (Å). Color marks the "
            "largest absolute feature score mapped to each residue; pair links show measured "
            "residue pairs and are not bonds. Hover in HTML to see residue identities.\n\n"
            + "\n".join(entries)
            + "\n\nPositive predictive score means removing the feature reduced grouped "
            "accuracy; negative means removal improved it. The condition view uses the sign of "
            "the replica-mean difference and the exploratory adjusted q value. Neither view "
            "establishes a causal mechanism, and global features cannot be localized to a residue.\n",
            encoding="utf-8",
        )
        index = root / "START_HERE.md"
        if index.is_file():
            marker = "[GPR6 receptor structure views](MOLECULAR_MAPS.md)"
            existing = index.read_text(encoding="utf-8")
            if marker not in existing:
                index.write_text(existing.rstrip() + "\n\n## Receptor structure views\n\n"
                                 + marker + " links the actual-coordinate 3D views and their interpretation limits.\n",
                                 encoding="utf-8")
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topology", required=True, help="Receptor-only PDB with unique residue numbers")
    parser.add_argument("--run-dir", required=True, help="Directory containing supervised and condition evidence tables")
    parser.add_argument("--chemical-definitions", help="JSON mapping named chemistry features to verified receptor residues; defaults to run directory")
    args = parser.parse_args()
    print(json.dumps(write_molecular_maps(args.topology, args.run_dir,
                                         chemical_definitions=args.chemical_definitions), indent=2))


if __name__ == "__main__":
    main()
