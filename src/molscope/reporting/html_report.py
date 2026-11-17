"""Generate a consolidated, single-page interactive HTML report dashboard.

Replaces the overwhelming 50+ loose files with an elegant, tabbed interactive
dashboard for inspecting MD analysis results, free energy landscapes, cluster
timelines, state transitions, decision tree explainability, and 3D molecular structures.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from molscope.visualization.rules import render_decision_tree_html


def _clean_val(val: Any) -> str:
    if pd.isna(val) or val is None:
        return "—"
    if isinstance(val, float):
        return f"{val:.4g}"
    return html.escape(str(val))


def _df_to_html_table(df: pd.DataFrame, max_rows: int = 15) -> str:
    if df.empty:
        return "<p class='empty-note'>No records available.</p>"
    head = df.head(max_rows)
    headers = "".join(f"<th>{html.escape(str(col))}</th>" for col in head.columns)
    rows = []
    for _, row in head.iterrows():
        cells = "".join(f"<td>{_clean_val(v)}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    tbody = "".join(rows)
    footer = f"<p class='table-sub'>Showing top {len(head)} of {len(df)} rows. Full CSV in <code>tables/</code>.</p>" if len(df) > max_rows else ""
    return f"<div class='table-container'><table><thead><tr>{headers}</tr></thead><tbody>{tbody}</tbody></table></div>{footer}"


def generate_html_report(
    destination: Path,
    outputs: Mapping[str, Any],
    manifest: pd.DataFrame | None = None,
    features: pd.DataFrame | None = None,
    figures: list[str] | None = None,
) -> Path:
    """Assemble a single offline HTML dashboard in the run directory."""
    figures = figures or []
    dest_path = Path(destination)
    report_file = dest_path / "report.html"

    # Gather KPIs
    num_frames = len(manifest) if manifest is not None else "—"
    num_replicas = len(manifest["replica_id"].unique()) if manifest is not None and "replica_id" in manifest.columns else "—"
    num_features = features.shape[1] if features is not None else "—"

    sam_mfp = outputs.get("sam_mfp_candidates")
    num_pareto = "—"
    if isinstance(sam_mfp, pd.DataFrame) and "SAM_MFP_Pareto_Front" in sam_mfp.columns:
        num_pareto = int(sam_mfp["SAM_MFP_Pareto_Front"].sum())

    model_summary = outputs.get("supervised_model_summary")
    cv_acc = "Not Run"
    if isinstance(model_summary, pd.DataFrame) and "Trajectory_Mean_Accuracy_Mean" in model_summary.columns:
        try:
            cv_acc = f"{model_summary['Trajectory_Mean_Accuracy_Mean'].iloc[0]:.1%}"
        except Exception:
            pass

    # Extract tables
    sam_mfp_html = _df_to_html_table(sam_mfp) if isinstance(sam_mfp, pd.DataFrame) else "<p>No SAM-MFP data.</p>"
    
    quality_df = outputs.get("feature_quality_metrics")
    quality_html = _df_to_html_table(quality_df) if isinstance(quality_df, pd.DataFrame) else "<p>No quality data.</p>"

    state_pop = outputs.get("state_populations")
    state_pop_html = _df_to_html_table(state_pop) if isinstance(state_pop, pd.DataFrame) else "<p>No state population data.</p>"

    pred_ranking = outputs.get("supervised_predictive_feature_ranking")
    pred_ranking_html = _df_to_html_table(pred_ranking) if isinstance(pred_ranking, pd.DataFrame) else "<p>No predictive ranking (unsupervised run).</p>"

    # Check for decision tree rules
    dt_rules = outputs.get("decision_tree_rules")
    if dt_rules is None:
        for cand_path in (dest_path / "decision_tree_rules.csv", dest_path / "tables/decision_tree_rules.csv"):
            if cand_path.is_file():
                try:
                    dt_rules = pd.read_csv(cand_path)
                    break
                except Exception:
                    pass
    dt_rules_html = render_decision_tree_html(dt_rules) if dt_rules is not None else "<p style='color: var(--text-secondary);'>No decision tree rules extracted.</p>"

    # Check for specific diagnostic plots
    fig_dir = dest_path / "figures"
    timeline_img = None
    transitions_img = None
    fes_img = None

    for fpath in (fig_dir / "cluster_timeline.png", dest_path / "cluster_timeline.png"):
        if fpath.is_file():
            timeline_img = f"figures/{fpath.name}" if fpath.parent.name == "figures" else fpath.name
            break

    for fpath in (fig_dir / "state_transitions.png", dest_path / "state_transitions.png"):
        if fpath.is_file():
            transitions_img = f"figures/{fpath.name}" if fpath.parent.name == "figures" else fpath.name
            break

    for fpath in (fig_dir / "pca_free_energy_landscape.png", dest_path / "pca_free_energy_landscape.png"):
        if fpath.is_file():
            fes_img = f"figures/{fpath.name}" if fpath.parent.name == "figures" else fpath.name
            break

    # Top residue scores for 3D view
    residue_scores_df = None
    for cand_p in (dest_path / "3d_visualization/molscope_3d_residue_scores.csv", dest_path / "molscope_3d_residue_scores.csv"):
        if cand_p.is_file():
            try:
                residue_scores_df = pd.read_csv(cand_p)
                break
            except Exception:
                pass
    res_scores_html = _df_to_html_table(residue_scores_df.head(10)) if isinstance(residue_scores_df, pd.DataFrame) else ""

    # Check for all images in figures/
    fig_cards = []
    if fig_dir.is_dir():
        for fig_path in sorted(fig_dir.glob("*.png")):
            name_clean = fig_path.stem.replace("_", " ").title()
            fig_cards.append(f"""
            <div class="card">
                <h3>{name_clean}</h3>
                <a href="figures/{fig_path.name}" target="_blank">
                    <img src="figures/{fig_path.name}" alt="{name_clean}" loading="lazy"/>
                </a>
            </div>
            """)
    figures_grid = "".join(fig_cards) if fig_cards else "<p>No figures rendered.</p>"

    # Build States tab extra visuals
    timeline_card = ""
    if timeline_img:
        timeline_card = f"""
        <div class="card">
            <h3>Cluster Membership Timeline</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 8px;">
                Sequential cluster and macrostate transitions across continuous simulation trajectories.
            </p>
            <a href="{timeline_img}" target="_blank">
                <img src="{timeline_img}" alt="Cluster Timeline" loading="lazy"/>
            </a>
        </div>
        """

    transitions_card = ""
    if transitions_img:
        transitions_card = f"""
        <div class="card">
            <h3>Empirical State Transition Matrix</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 8px;">
                Markovian jump probabilities P(S_(t+1) = j | S_t = i) between candidate conformational states.
            </p>
            <a href="{transitions_img}" target="_blank">
                <img src="{transitions_img}" alt="State Transitions" loading="lazy" style="max-width: 550px;"/>
            </a>
        </div>
        """

    fes_card = ""
    if fes_img:
        fes_card = f"""
        <div class="card">
            <h3>Leading PCA Free Energy Landscape (FES)</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 8px;">
                2D empirical Potential of Mean Force (PMF) ΔG(PC1, PC2) with identified conformational basins (B1, B2, ...).
            </p>
            <a href="{fes_img}" target="_blank">
                <img src="{fes_img}" alt="PCA Free Energy Landscape" loading="lazy" style="max-width: 650px;"/>
            </a>
        </div>
        """

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MolScope Analysis Dashboard</title>
    <style>
        :root {{
            --bg-primary: #0f172a;
            --bg-secondary: #1e293b;
            --bg-card: #1e293b;
            --border: #334155;
            --text-primary: #f8fafc;
            --text-secondary: #94a3b8;
            --accent-primary: #38bdf8;
            --accent-orange: #fb923c;
            --accent-green: #4ade80;
            --font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background: var(--bg-primary);
            color: var(--text-primary);
            font-family: var(--font-family);
            line-height: 1.5;
            padding: 24px;
        }}
        header {{
            margin-bottom: 24px;
            padding-bottom: 16px;
            border-bottom: 1px solid var(--border);
        }}
        .header-title {{
            display: flex;
            align-items: center;
            justify-content: space-between;
        }}
        h1 {{ font-size: 24px; font-weight: 700; color: var(--accent-primary); }}
        .badge {{
            background: #0284c7;
            color: white;
            padding: 4px 10px;
            border-radius: 9999px;
            font-size: 12px;
            font-weight: 600;
        }}
        .kpi-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 16px;
            margin: 20px 0;
        }}
        .kpi-card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
        }}
        .kpi-title {{ font-size: 12px; text-transform: uppercase; color: var(--text-secondary); font-weight: 600; }}
        .kpi-value {{ font-size: 26px; font-weight: 700; color: var(--text-primary); margin-top: 4px; }}
        
        /* Tabs */
        .tabs {{
            display: flex;
            gap: 8px;
            margin-bottom: 20px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-secondary);
            font-size: 14px;
            font-weight: 600;
            padding: 8px 16px;
            cursor: pointer;
            border-radius: 6px;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ color: var(--text-primary); background: var(--bg-secondary); }}
        .tab-btn.active {{
            color: var(--accent-primary);
            background: #0369a1;
        }}
        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        /* Cards & Grids */
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
            margin-bottom: 20px;
        }}
        .card h3 {{ font-size: 16px; color: var(--accent-primary); margin-bottom: 12px; }}
        .grid-2 {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 6px;
            border: 1px solid var(--border);
            margin-top: 8px;
        }}
        
        /* Tables */
        .table-container {{
            overflow-x: auto;
            border: 1px solid var(--border);
            border-radius: 6px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
        }}
        th, td {{
            padding: 10px 14px;
            text-align: left;
            border-bottom: 1px solid var(--border);
        }}
        th {{ background: #111e33; color: var(--text-secondary); font-weight: 600; }}
        tr:hover {{ background: rgba(56, 189, 248, 0.05); }}
        .table-sub {{ font-size: 12px; color: var(--text-secondary); margin-top: 8px; }}
        
        code {{
            background: #0f172a;
            padding: 2px 6px;
            border-radius: 4px;
            color: var(--accent-orange);
            font-family: monospace;
            font-size: 12px;
        }}
        pre {{
            background: #090d16;
            padding: 14px;
            border-radius: 6px;
            overflow-x: auto;
            color: var(--accent-green);
            font-size: 13px;
            margin-top: 8px;
        }}

        /* Decision Tree Rule Cards */
        .dt-rules-container {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 16px;
            margin-top: 12px;
        }}
        .dt-rule-card {{
            background: #131d2e;
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
        }}
        .dt-rule-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .dt-cluster-badge {{
            background: #0284c7;
            color: white;
            font-size: 11px;
            font-weight: 700;
            padding: 3px 8px;
            border-radius: 4px;
        }}
        .dt-rule-num {{ font-size: 12px; color: var(--text-secondary); font-weight: 600; }}
        .dt-rule-body {{ margin-bottom: 14px; }}
        .dt-if-tag {{ font-size: 11px; font-weight: 800; color: var(--accent-orange); margin-bottom: 4px; }}
        .dt-clauses {{ margin-left: 8px; margin-bottom: 8px; }}
        .dt-rule-clause {{ margin-bottom: 4px; }}
        .dt-then-tag {{ font-size: 12px; color: var(--text-primary); }}
        .dt-rule-metrics {{
            display: flex;
            gap: 16px;
            border-top: 1px solid var(--border);
            padding-top: 10px;
            font-size: 11px;
        }}
        .dt-metric {{ flex: 1; }}
        .dt-metric-label {{ color: var(--text-secondary); display: block; }}
        .dt-metric-val {{ font-weight: 700; color: var(--text-primary); }}
        .dt-bar {{
            background: #0f172a;
            height: 4px;
            border-radius: 2px;
            margin-top: 3px;
            overflow: hidden;
        }}
        .dt-fill.prec {{ background: var(--accent-primary); }}
        .dt-fill.rec {{ background: var(--accent-green); }}

        /* Sidecar link button */
        .sidecar-btn {{
            display: inline-block;
            background: #0284c7;
            color: white;
            text-decoration: none;
            padding: 8px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 13px;
            margin-top: 8px;
            margin-bottom: 12px;
        }}
        .sidecar-btn:hover {{ background: #0369a1; }}
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>MolScope Study Dashboard</h1>
            <span class="badge">Run Complete</span>
        </div>
        <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">
            Clean consolidated view of biophysical feature prioritization, conformational states, interpretability rules, and 3D molecular structures.
        </p>
    </header>

    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-title">Frames Analyzed</div>
            <div class="kpi-value">{num_frames}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Replicas</div>
            <div class="kpi-value">{num_replicas}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Extracted Features</div>
            <div class="kpi-value">{num_features}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">SAM-MFP Pareto Candidates</div>
            <div class="kpi-value" style="color: var(--accent-orange);">{num_pareto}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-title">Held-Out CV Accuracy</div>
            <div class="kpi-value" style="color: var(--accent-green);">{cv_acc}</div>
        </div>
    </div>

    <div class="tabs">
        <button class="tab-btn active" onclick="switchTab('tab-prioritization')">🎯 SAM-MFP Prioritization</button>
        <button class="tab-btn" onclick="switchTab('tab-states')">🧬 Conformational States</button>
        <button class="tab-btn" onclick="switchTab('tab-predictive')">📈 Predictive Importance</button>
        <button class="tab-btn" onclick="switchTab('tab-figures')">🖼️ Diagnostic Figures</button>
        <button class="tab-btn" onclick="switchTab('tab-3d')">🧊 3D PyMOL / Sidecar</button>
        <button class="tab-btn" onclick="switchTab('tab-files')">📁 Files & Exports</button>
    </div>

    <!-- TAB 1: SAM-MFP -->
    <div id="tab-prioritization" class="tab-content active">
        <div class="card">
            <h3>System-Aware Multi-Criteria Feature Prioritization (SAM-MFP)</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 12px;">
                Identifies non-dominated Pareto trade-offs across Effective Sample Size (ESS), Replica Consistency, Informational Entropy, and Target-Free PCA Salience.
            </p>
            {sam_mfp_html}
        </div>
        <div class="card">
            <h3>Feature Quality & Reliability Metrics</h3>
            {quality_html}
        </div>
    </div>

    <!-- TAB 2: STATES -->
    <div id="tab-states" class="tab-content">
        <div class="card">
            <h3>Interpretable Decision Tree Rules (If-Then Logic)</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 8px;">
                Sparse, human-interpretable boolean rules explaining what distinguishes each conformational state or cluster.
            </p>
            {dt_rules_html}
        </div>

        {timeline_card}
        {transitions_card}
        {fes_card}

        <div class="card">
            <h3>Conformational State Populations per Trajectory</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 12px;">
                Fraction of simulation time spent in each candidate conformational state across independent replicas.
            </p>
            {state_pop_html}
        </div>
    </div>

    <!-- TAB 3: PREDICTIVE -->
    <div id="tab-predictive" class="tab-content">
        <div class="card">
            <h3>Leakage-Aware Predictive Feature Ranking (LOFO Importance)</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 12px;">
                Evaluated using trajectory-grouped cross-validation to prevent frame autocorrelation leakage.
            </p>
            {pred_ranking_html}
        </div>
    </div>

    <!-- TAB 4: FIGURES -->
    <div id="tab-figures" class="tab-content">
        <div class="grid-2">
            {figures_grid}
        </div>
    </div>

    <!-- TAB 5: 3D VIEW -->
    <div id="tab-3d" class="tab-content">
        <div class="card">
            <h3>Interactive 3D Structure Visualizations & B-Factor Mapping</h3>
            <p style="color: var(--text-secondary); font-size: 14px; margin-bottom: 16px;">
                MolScope maps per-residue importance directly into the Beta-factor column of the reference PDB structure.
            </p>
            
            <h4 style="color: var(--text-primary); margin-top: 12px;">1. Antigravity Biomolecular Structure Viewer (Sidecar)</h4>
            <p style="color: var(--text-secondary); font-size: 13px; margin-top: 4px;">
                Open the interactive Mol* 3D viewer sidecar to inspect residue importance directly in Antigravity:
            </p>
            <a href="sidecar://science/biomolecular_structure_viewer/ui.html" class="sidecar-btn">
                🔬 Open in Biomolecular Structure Viewer
            </a>
            <p style="color: var(--text-secondary); font-size: 12px;">
                Load <code>3d_visualization/molscope_3d.pdb</code> and select coloring by <i>B-factor / Temperature Factor</i>.
            </p>

            <h4 style="color: var(--text-primary); margin-top: 18px;">2. Open in PyMOL</h4>
            <pre>pymol 3d_visualization/molscope_3d.pml</pre>
            <p style="color: var(--text-secondary); font-size: 12px; margin-top: 4px;">
                Loads the structure with cartoon putty, spectrum coloring by feature importance (blue=low, red=high), and labels top discriminative residues.
            </p>

            <h4 style="color: var(--text-primary); margin-top: 18px;">3. Open in UCSF ChimeraX</h4>
            <pre>chimerax 3d_visualization/molscope_3d.cxc</pre>

            <div style="margin-top: 20px;">
                <h4 style="color: var(--text-primary); margin-bottom: 8px;">Top Scored Discriminative Residues</h4>
                {res_scores_html}
            </div>
        </div>
    </div>

    <!-- TAB 6: FILES -->
    <div id="tab-files" class="tab-content">
        <div class="card">
            <h3>Organized Output Directory Structure</h3>
            <p style="color: var(--text-secondary); font-size: 13px; margin-bottom: 16px;">
                All calculation outputs are filed into clean, dedicated subdirectories:
            </p>
            <pre>
run_output/
├── report.html                  <-- This interactive dashboard
├── summary.md                   <-- Executive summary report
│
├── 3d_visualization/           <-- PyMOL & ChimeraX 3D scripts + scored PDB
│   ├── molscope_3d.pdb          <-- PDB with B-factor importance mapping
│   ├── molscope_3d.pml          <-- One-click PyMOL cartoon putty script
│   ├── molscope_3d.cxc          <-- UCSF ChimeraX visualization script
│   └── molscope_3d_residue_scores.csv
│
├── figures/                     <-- Publication-quality PNG plots
│   ├── pca_free_energy_landscape.png
│   ├── cluster_timeline.png
│   ├── state_transitions.png
│   ├── decision_tree_rules.png
│   ├── sam_mfp_tradeoff.png
│   └── predictive_feature_ranking.png
│
├── tables/                      <-- All raw CSV data tables
│   ├── sam_mfp_candidates.csv
│   ├── feature_quality_metrics.csv
│   └── state_assignments.csv
│
└── audit/                       <-- Reproducibility metadata & manifests
    ├── extraction_evidence.json
    └── research_report.json
            </pre>
        </div>
    </div>

    <script>
        function switchTab(tabId) {{
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabId).classList.add('active');
            event.target.classList.add('active');
        }}
    </script>
</body>
</html>
"""
    report_file.write_text(html_content, encoding="utf-8")
    return report_file
