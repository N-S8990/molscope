"""Map feature importance metrics onto PDB structure Beta-factor columns for 3D visualization.

Supports PyMOL, ChimeraX, and Mol* (Antigravity Biomolecular Structure Viewer sidecar).
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


_RES_PATTERN = re.compile(r"(?:[A-Za-z]{3}|res|residue)[_:\s]+([0-9]+)", flags=re.IGNORECASE)
_DIGITS_PATTERN = re.compile(r"(?:^|[_:\s])(\d+)(?:[_:\s]|$)")
_PAIR_PATTERN = re.compile(r"([A-Za-z0-9]+)__([A-Za-z]{3})_(\d+)__([A-Za-z0-9]+)__([A-Za-z]{3})_(\d+)")
_CHAIN_RES_PATTERN = re.compile(r"__(?:X|chain_([A-Za-z0-9]+))__([A-Za-z]{3})_(\d+)")
_SIDECHAIN_PATTERN = re.compile(r"SIDECHAIN_.*?([A-Za-z]{3})_(\d+)")


def extract_residues_from_feature_name(feature_name: str) -> list[int]:
    """Parse all residue sequence numbers from standard MolScope feature identifiers.

    Matches:
    - Standard pair formats: chainA__ARG_37__chainA__LYS_42
    - Sidechain dihedrals: SIDECHAIN_CHI1__ARG_37, TORSION_CHI1_37
    - Triplet residue patterns: ALA_25, ARG_37, residue_14, resSeq 42
    - Chain patterns: __chain_A__ARG_37, __X__ALA_25
    - Fallback digit patterns
    """
    s = str(feature_name).strip()
    res_nums: list[int] = []

    # Check pair pattern first
    pair_match = _PAIR_PATTERN.search(s)
    if pair_match:
        res_nums.extend([int(pair_match.group(3)), int(pair_match.group(6))])
        return sorted(set(res_nums))

    # Check chain-residue pattern
    chain_matches = _CHAIN_RES_PATTERN.findall(s)
    if chain_matches:
        for _, _, resid_str in chain_matches:
            res_nums.append(int(resid_str))
        return sorted(set(res_nums))

    # Check sidechain pattern
    sc_match = _SIDECHAIN_PATTERN.search(s)
    if sc_match:
        res_nums.append(int(sc_match.group(2)))
        return res_nums

    # Standard amino acid or residue prefix pattern
    matches = _RES_PATTERN.findall(s)
    if matches:
        return sorted(set(int(m) for m in matches))

    # Standalone double underscore pattern (e.g. __42__)
    digits = re.findall(r"__([0-9]+)__", s)
    if digits:
        return sorted(set(int(d) for d in digits))

    return res_nums


def extract_residue_scores(
    importance_df: pd.DataFrame,
    feature_col: str | None = None,
    importance_col: str | None = None,
    aggregation: str = "max",
) -> dict[int, float]:
    """Aggregate feature-level importance metrics to individual residue sequence numbers.

    Parameters
    ----------
    importance_df : pd.DataFrame
        Table with feature identifiers and numerical scores.
    feature_col : str | None, default=None
        Column name holding feature identifiers. Auto-detected if None.
    importance_col : str | None, default=None
        Column name holding importance metrics. Auto-detected if None.
    aggregation : str, default="max"
        Method to aggregate multiple features mapping to the same residue:
        "max" (highest absolute score), "sum" (cumulative importance), or "mean".

    Returns
    -------
    dict[int, float]
        Mapping from residue sequence number to aggregated scalar score.
    """
    df = importance_df.copy()
    if df.empty:
        return {}

    # Auto-detect feature column
    if feature_col is None:
        for cand in ("Feature", "Feature_Group", "feature", "name"):
            if cand in df.columns:
                feature_col = cand
                break
        if feature_col is None:
            feature_col = df.columns[0]

    # Auto-detect importance column
    if importance_col is None:
        for cand in (
            "Gini_Importance",
            "Normalized_Score",
            "Importance_MDI",
            "Mean_LOFO_Importance",
            "Trajectory_Mean_Accuracy_Drop_Mean",
            "PCA_Salience",
            "Kinetic_Salience",
            "TICA_Salience",
            "Contact_Salience",
            "Importance",
            "Score",
        ):
            if cand in df.columns:
                importance_col = cand
                break
        if importance_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                importance_col = numeric_cols[0]
            else:
                return {}

    sub = df.dropna(subset=[feature_col, importance_col])
    accum: dict[int, list[float]] = {}

    for _, row in sub.iterrows():
        feat = str(row[feature_col])
        try:
            val = float(row[importance_col])
        except (ValueError, TypeError):
            continue
        if not np.isfinite(val):
            continue

        res_nums = extract_residues_from_feature_name(feat)
        for r in res_nums:
            accum.setdefault(r, []).append(abs(val))

    scores: dict[int, float] = {}
    for r, vals in accum.items():
        if aggregation == "sum":
            scores[r] = float(np.sum(vals))
        elif aggregation == "mean":
            scores[r] = float(np.mean(vals))
        else:  # default "max"
            scores[r] = float(np.max(vals))

    return scores


def map_feature_importance_to_pdb(
    pdb_input_path: str | Path,
    importance_df: pd.DataFrame | Mapping[int, float],
    pdb_output_path: str | Path,
    *,
    feature_col: str | None = None,
    importance_col: str | None = None,
    normalize: bool = True,
    max_scale: float = 100.0,
) -> Path:
    """Map feature importances to residue Beta-factors in a reference PDB structure.

    Parameters
    ----------
    pdb_input_path : str or Path
        Path to reference input PDB file.
    importance_df : pd.DataFrame or Mapping[int, float]
        DataFrame with feature identifiers and scores, or precomputed dict of {residue: score}.
    pdb_output_path : str or Path
        Destination path for the annotated PDB file.
    feature_col : str | None, default=None
        Column holding feature identifiers.
    importance_col : str | None, default=None
        Column holding importance values.
    normalize : bool, default=True
        Whether to scale residue scores to [0.0, max_scale] (e.g. 100.0 for standard PDB B-factors).
    max_scale : float, default=100.0
        Maximum target score for the normalization ceiling.

    Returns
    -------
    Path
        Resolved path to the newly written PDB file.
    """
    input_path = Path(pdb_input_path).resolve()
    output_path = Path(pdb_output_path).resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input PDB file not found: {input_path}")

    if isinstance(importance_df, dict):
        residue_scores = {int(k): float(v) for k, v in importance_df.items() if np.isfinite(v)}
    else:
        residue_scores = extract_residue_scores(
            importance_df=importance_df,
            feature_col=feature_col,
            importance_col=importance_col,
        )

    # Normalize residue scores
    normalized: dict[int, float] = {}
    if residue_scores:
        max_score = max(residue_scores.values())
        if max_score > 0 and normalize:
            for r, val in residue_scores.items():
                normalized[r] = float(np.clip((val / max_score) * max_scale, 0.0, max_scale))
        else:
            normalized = dict(residue_scores)

    lines = input_path.read_text(encoding="utf-8").splitlines()
    out_lines: list[str] = []

    for line in lines:
        if line.startswith(("ATOM  ", "HETATM")):
            try:
                res_seq = int(line[22:26].strip())
                bfactor = normalized.get(res_seq, 0.0)
                # Standard PDB format: B-factor in cols 61-66 (6 characters, %6.2f)
                new_line = line[:60] + f"{bfactor:6.2f}" + line[66:]
                out_lines.append(new_line)
            except Exception:
                out_lines.append(line)
        else:
            out_lines.append(line)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(out_lines) + "\n", encoding="utf-8")
    return output_path


def generate_pymol_script(
    pdb_path: str | Path,
    output_pml_path: str | Path,
    residue_scores: Mapping[int, float] | None = None,
    top_label_count: int = 8,
    object_name: str = "molscope_protein",
    min_b: float = 0.0,
    max_b: float = 100.0,
) -> Path:
    """Generate a publication-ready PyMOL visualization script (.pml).

    Loads the B-factor mapped structure, applies cartoon putty, colors by spectrum
    (blue for low importance, red for high importance), and highlights the top
    discriminative residues with sticks and CA labels.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the annotated PDB file (with B-factors).
    output_pml_path : str or Path
        Destination for the generated .pml script.
    residue_scores : Mapping[int, float] | None, default=None
        Residue sequence number to score mapping, used to identify top residues.
    top_label_count : int, default=8
        Number of top-scoring residues to display with sidechain sticks and labels.
    object_name : str, default="molscope_protein"
        Name of the loaded PyMOL molecular object.
    min_b : float, default=0.0
        Minimum B-factor for spectrum coloring.
    max_b : float, default=100.0
        Maximum B-factor for spectrum coloring.

    Returns
    -------
    Path
        Path to the written .pml script.
    """
    p_pdb = Path(pdb_path).resolve()
    p_out = Path(output_pml_path).resolve()
    p_out.parent.mkdir(parents=True, exist_ok=True)

    top_resids: list[str] = []
    if residue_scores:
        sorted_res = sorted(residue_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_label_count]
        top_resids = [str(r) for r, score in sorted_res if score > 0]

    res_selection = "+".join(top_resids) if top_resids else ""

    lines = [
        "# MolScope PyMOL Visualization Script",
        "# Auto-generated for publication-quality 3D feature importance inspection",
        "reinitialize",
        f"load {p_pdb.name}, {object_name}",
        f"hide everything, {object_name}",
        f"show cartoon, {object_name}",
        f"cartoon putty, {object_name}",
        f"spectrum b, blue_white_red, {object_name}, minimum={min_b:.1f}, maximum={max_b:.1f}",
        "set cartoon_putty_scale_min, 0.35",
        "set cartoon_putty_scale_max, 1.85",
        "set cartoon_putty_transform, 0",
        "bg_color white",
        "set ray_opaque_background, 1",
        "set antialias, 2",
        "set ray_shadows, 1",
    ]

    if res_selection:
        lines.extend([
            "",
            "# Highlight top discriminative residues",
            f"select top_discriminators, {object_name} and resi {res_selection}",
            "show sticks, top_discriminators and not (name c+n+o)",
            "set stick_radius, 0.22",
            "select top_ca, top_discriminators and name CA",
            "label top_ca, '%s%s' % (resn, resi)",
            "set label_color, black",
            "set label_size, 14",
            "set label_font_id, 7",
            "set label_position, (0, 0, 2.5)",
            "deselect",
        ])

    lines.extend([
        "",
        "zoom",
        f"# Run via: pymol {p_out.name}",
    ])

    p_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p_out


def generate_chimerax_script(
    pdb_path: str | Path,
    output_cxc_path: str | Path,
    residue_scores: Mapping[int, float] | None = None,
    top_label_count: int = 8,
    min_b: float = 0.0,
    max_b: float = 100.0,
) -> Path:
    """Generate a UCSF ChimeraX visualization script (.cxc).

    Parameters
    ----------
    pdb_path : str or Path
        Path to the annotated PDB file.
    output_cxc_path : str or Path
        Destination for the .cxc script.
    residue_scores : Mapping[int, float] | None, default=None
        Residue scores for highlighting top discriminators.
    top_label_count : int, default=8
        Number of top residues to label.
    min_b : float, default=0.0
        Minimum B-factor for palette scaling.
    max_b : float, default=100.0
        Maximum B-factor for palette scaling.

    Returns
    -------
    Path
        Path to the written .cxc script.
    """
    p_pdb = Path(pdb_path).resolve()
    p_out = Path(output_cxc_path).resolve()
    p_out.parent.mkdir(parents=True, exist_ok=True)

    top_resids: list[str] = []
    if residue_scores:
        sorted_res = sorted(residue_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_label_count]
        top_resids = [str(r) for r, score in sorted_res if score > 0]

    lines = [
        "# MolScope UCSF ChimeraX Visualization Script",
        "close",
        f"open {p_pdb.name}",
        "hide atoms",
        "show cartoon",
        f"color bfactor palette blue:white:red range {min_b:.1f},{max_b:.1f}",
        "set bgColor white",
    ]

    if top_resids:
        res_spec = ",".join(top_resids)
        lines.extend([
            f"show :{res_spec} & !@C,N,O sticks",
            f"label :{res_spec}@CA text '{{0.name}}{{0.number}}' color black height 1.2",
        ])

    lines.extend([
        "view",
        f"# Run via: chimerax {p_out.name}",
    ])

    p_out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p_out


def generate_3d_visualization(
    topology_path: str | Path,
    importance_df: pd.DataFrame | Mapping[int, float],
    output_dir: str | Path,
    basename: str = "molscope_3d",
    feature_col: str | None = None,
    importance_col: str | None = None,
    top_label_count: int = 8,
    normalize: bool = True,
    max_scale: float = 100.0,
) -> dict[str, Any]:
    """Complete 3D visualization suite generator.

    Produces:
    1. Annotated PDB structure with per-residue importance in B-factor column
    2. PyMOL .pml script with cartoon putty and top residue highlights
    3. ChimeraX .cxc script
    4. Per-residue scores table (CSV) for quantitative verification and Mol* sidecar loading

    Parameters
    ----------
    topology_path : str or Path
        Input PDB reference structure.
    importance_df : pd.DataFrame or Mapping[int, float]
        Feature importance table or dictionary.
    output_dir : str or Path
        Directory where visualization artifacts will be saved.
    basename : str, default="molscope_3d"
        Base filename for generated files.
    feature_col : str | None, default=None
        Column holding feature identifiers.
    importance_col : str | None, default=None
        Column holding importance metrics.
    top_label_count : int, default=8
        Number of top residues to label in 3D viewer scripts.
    normalize : bool, default=True
        Whether to normalize B-factors to [0.0, max_scale].
    max_scale : float, default=100.0
        Maximum B-factor scale ceiling.

    Returns
    -------
    dict[str, Any]
        Dictionary with paths to 'pdb', 'pymol_script', 'chimerax_script',
        'residue_scores_csv', and 'top_residues'.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(importance_df, dict):
        res_scores = {int(k): float(v) for k, v in importance_df.items() if np.isfinite(v)}
    else:
        res_scores = extract_residue_scores(
            importance_df=importance_df,
            feature_col=feature_col,
            importance_col=importance_col,
        )

    # 1. Map scores to PDB B-factors
    pdb_out = out_dir / f"{basename}.pdb"
    map_feature_importance_to_pdb(
        pdb_input_path=topology_path,
        importance_df=res_scores,
        pdb_output_path=pdb_out,
        normalize=normalize,
        max_scale=max_scale,
    )

    # 2. PyMOL script
    pml_out = out_dir / f"{basename}.pml"
    generate_pymol_script(
        pdb_path=pdb_out,
        output_pml_path=pml_out,
        residue_scores=res_scores,
        top_label_count=top_label_count,
        min_b=0.0,
        max_b=max_scale if normalize else (max(res_scores.values(), default=100.0)),
    )

    # 3. ChimeraX script
    cxc_out = out_dir / f"{basename}.cxc"
    generate_chimerax_script(
        pdb_path=pdb_out,
        output_cxc_path=cxc_out,
        residue_scores=res_scores,
        top_label_count=top_label_count,
        min_b=0.0,
        max_b=max_scale if normalize else (max(res_scores.values(), default=100.0)),
    )

    # 4. Save per-residue scores CSV
    sorted_res = sorted(res_scores.items(), key=lambda kv: kv[1], reverse=True)
    df_scores = pd.DataFrame([
        {"Residue": r, "Importance_Score": sc, "Normalized_BFactor": (sc / max(res_scores.values())) * max_scale if max(res_scores.values(), default=0) > 0 else 0.0}
        for r, sc in sorted_res
    ])
    scores_csv = out_dir / f"{basename}_residue_scores.csv"
    df_scores.to_csv(scores_csv, index=False)

    return {
        "pdb": pdb_out,
        "pymol_script": pml_out,
        "chimerax_script": cxc_out,
        "residue_scores_csv": scores_csv,
        "top_residues": [r for r, _ in sorted_res[:top_label_count]],
    }
