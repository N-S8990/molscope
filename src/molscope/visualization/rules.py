"""Decision tree rule visualization and feature importance plotting module.

Provides publication-quality visual diagrams of if-then state rules, Gini/predictive
feature importance rankings, and interactive HTML cards for reporting.
"""

from __future__ import annotations

import html
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from molscope.analysis.discovery.decision_tree import DecisionRule, DecisionTreeExplainer


def _clean_feature_label(name: str) -> str:
    """Make feature identifiers human-readable for plot labels."""
    s = str(name).strip()
    s = s.replace("__", " ").replace("_", " ")
    return s


def plot_feature_importance(
    importance_df: pd.DataFrame,
    output_path: str | Path,
    feature_col: str | None = None,
    importance_col: str | None = None,
    top_n: int = 15,
    title: str = "Feature Importance Ranking",
    color: str = "#3182ce",
    xlabel: str | None = None,
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Render a horizontal bar plot of top feature importance scores.

    Parameters
    ----------
    importance_df : pd.DataFrame
        DataFrame containing feature identifiers and numerical scores.
    output_path : str or Path
        Destination path for the generated image.
    feature_col : str | None, default=None
        Column holding feature names. Auto-detected from ("Feature", "Feature_Group", "name").
    importance_col : str | None, default=None
        Column holding importance metrics. Auto-detected from Gini, LOFO, or Salience columns.
    top_n : int, default=15
        Number of top features to include in the bar plot.
    title : str, default="Feature Importance Ranking"
        Plot title.
    color : str, default="#3182ce"
        Bar fill color.
    xlabel : str | None, default=None
        Label for horizontal axis.
    dpi : int, default=300
        Image resolution.
    figsize : tuple[float, float] | None, default=None
        Figure dimensions.

    Returns
    -------
    Path
        Resolved path to the saved image.
    """
    df = importance_df.copy()
    if df.empty:
        raise ValueError("importance_df is empty.")

    # Auto-detect feature column
    if feature_col is None:
        for cand in ("Feature", "Feature_Group", "feature", "name", "Variable"):
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
            "Importance",
            "Score",
            "value",
        ):
            if cand in df.columns:
                importance_col = cand
                break
        if importance_col is None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                importance_col = numeric_cols[0]
            else:
                raise KeyError(f"No numeric importance column found in DataFrame: {list(df.columns)}")

    sub = df.dropna(subset=[feature_col, importance_col]).copy()
    sub[importance_col] = pd.to_numeric(sub[importance_col], errors="coerce")
    sub = sub.dropna(subset=[importance_col])

    # Rank and select top N
    top_df = sub.nlargest(top_n, importance_col).iloc[::-1]  # Invert so largest is at the top
    features = [_clean_feature_label(f) for f in top_df[feature_col]]
    scores = top_df[importance_col].to_numpy(dtype=float)

    fig_h = max(3.5, len(features) * 0.38)
    fig_w = 9.0
    fig_size = figsize or (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=fig_size)
    bars = ax.barh(range(len(features)), scores, color=color, alpha=0.85, edgecolor="#2c5282", height=0.65)

    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features, fontsize=9.5)
    ax.set_xlabel(xlabel or importance_col.replace("_", " "), fontsize=10, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)
    ax.grid(True, linestyle="--", alpha=0.35, axis="x")

    # Add numeric labels at bar tips
    max_val = np.max(scores) if len(scores) > 0 else 1.0
    offset = max_val * 0.015
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_width() + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#2d3748",
        )

    ax.set_xlim(0, max_val * 1.15 if max_val > 0 else 1.0)
    fig.tight_layout()

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_p


def plot_decision_tree_rules(
    explainer_or_rules: DecisionTreeExplainer | Sequence[DecisionRule] | pd.DataFrame,
    output_path: str | Path,
    top_n: int = 6,
    title: str = "Conformational State Decision Rules",
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Render a visual diagram summarizing extracted if-then rules with precision/recall metrics.

    Parameters
    ----------
    explainer_or_rules : DecisionTreeExplainer, list of DecisionRule, or DataFrame
        Extracted rules representation.
    output_path : str or Path
        Target destination for output image.
    top_n : int, default=6
        Maximum number of rules to display.
    title : str, default="Conformational State Decision Rules"
        Figure title.
    dpi : int, default=300
        Image resolution.
    figsize : tuple[float, float] | None, default=None
        Figure dimensions.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    # Extract rules into uniform list of dicts
    rules_data: list[dict[str, Any]] = []

    if isinstance(explainer_or_rules, DecisionTreeExplainer):
        all_rules = explainer_or_rules.get_rules()
        for r in all_rules[:top_n]:
            rules_data.append({
                "cluster": str(r.cluster),
                "conditions": r.conditions,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "support": r.support,
                "total": r.total_samples,
            })
    elif isinstance(explainer_or_rules, pd.DataFrame):
        df = explainer_or_rules.head(top_n)
        for _, row in df.iterrows():
            rule_str = str(row.get("Rule", ""))
            conds = [rule_str]
            rules_data.append({
                "cluster": str(row.get("Cluster", "")),
                "conditions": conds,
                "precision": float(row.get("Precision", 0.0)),
                "recall": float(row.get("Recall", 0.0)),
                "f1": float(row.get("F1", 0.0)),
                "support": int(row.get("Support", 0)),
                "total": int(row.get("Total_Samples", 0)),
            })
    else:
        # Sequence of DecisionRule objects
        for r in list(explainer_or_rules)[:top_n]:
            rules_data.append({
                "cluster": str(r.cluster),
                "conditions": r.conditions,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "support": r.support,
                "total": r.total_samples,
            })

    if not rules_data:
        raise ValueError("No decision rules provided to plot.")

    n_rules = len(rules_data)
    fig_h = max(4.0, n_rules * 1.5 + 1.2)
    fig_w = 11.0
    fig_size = figsize or (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.axis("off")

    y_pos = 0.92
    y_step = 0.85 / n_rules

    ax.text(
        0.5,
        0.98,
        title,
        ha="center",
        va="top",
        fontsize=13,
        fontweight="bold",
        color="#1a202c",
    )

    for i, r in enumerate(rules_data, start=1):
        top_y = y_pos
        bot_y = y_pos - y_step + 0.04
        box_h = top_y - bot_y

        # Draw card background
        rect = plt.Rectangle(
            (0.04, bot_y),
            0.92,
            box_h,
            facecolor="#f7fafc",
            edgecolor="#cbd5e0",
            linewidth=1.2,
            transform=ax.transAxes,
            zorder=1,
            clip_on=False,
        )
        ax.add_patch(rect)

        # Draw left accent color bar
        accent_color = "#3182ce" if i % 2 == 1 else "#38a169"
        accent = plt.Rectangle(
            (0.04, bot_y),
            0.015,
            box_h,
            facecolor=accent_color,
            edgecolor="none",
            transform=ax.transAxes,
            zorder=2,
            clip_on=False,
        )
        ax.add_patch(accent)

        # Rule target header
        cl_name = r["cluster"]
        header_text = f"Rule #{i}  →  Conformational State {cl_name}"
        ax.text(
            0.075,
            top_y - 0.035,
            header_text,
            transform=ax.transAxes,
            fontsize=10.5,
            fontweight="bold",
            color="#2d3748",
            va="top",
        )

        # Metrics badges (precision, recall, support)
        prec_text = f"Precision: {r['precision']:.1%}"
        rec_text = f"Recall: {r['recall']:.1%}"
        supp_text = f"Support: {r['support']} / {r['total']} frames"
        metrics_line = f"{prec_text}   |   {rec_text}   |   {supp_text}"
        ax.text(
            0.94,
            top_y - 0.035,
            metrics_line,
            transform=ax.transAxes,
            fontsize=9.0,
            fontweight="medium",
            color="#4a5568",
            ha="right",
            va="top",
        )

        # Conditions
        cond_lines = r["conditions"]
        if cond_lines:
            cond_display = "IF " + " AND\n   ".join(cond_lines)
        else:
            cond_display = "IF (Root Node / Unconstrained)"

        ax.text(
            0.075,
            top_y - 0.08,
            cond_display,
            transform=ax.transAxes,
            fontsize=9.0,
            family="monospace",
            color="#2b6cb0",
            va="top",
        )

        y_pos -= y_step

    fig.tight_layout()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_p


def render_decision_tree_html(
    explainer_or_rules: DecisionTreeExplainer | Sequence[DecisionRule] | pd.DataFrame,
    cluster: Any | None = None,
) -> str:
    """Generate interactive HTML cards displaying decision tree rules for web dashboards.

    Parameters
    ----------
    explainer_or_rules : DecisionTreeExplainer, list of DecisionRule, or DataFrame
        Extracted rules.
    cluster : Any | None, default=None
        Specific cluster to filter for, or None for all rules.

    Returns
    -------
    str
        HTML snippet containing styled rule cards.
    """
    rules_data: list[dict[str, Any]] = []

    if isinstance(explainer_or_rules, DecisionTreeExplainer):
        rules = explainer_or_rules.get_rules(cluster=cluster)
        for r in rules:
            rules_data.append({
                "cluster": str(r.cluster),
                "rule_string": r.rule_string,
                "conditions": r.conditions,
                "precision": r.precision,
                "recall": r.recall,
                "f1": r.f1,
                "support": r.support,
                "total": r.total_samples,
            })
    elif isinstance(explainer_or_rules, pd.DataFrame):
        df = explainer_or_rules.copy()
        if cluster is not None and "Cluster" in df.columns:
            df = df[df["Cluster"].astype(str) == str(cluster)]
        for _, row in df.iterrows():
            rules_data.append({
                "cluster": str(row.get("Cluster", "")),
                "rule_string": str(row.get("Rule", "")),
                "conditions": [str(row.get("Rule", ""))],
                "precision": float(row.get("Precision", 0.0)),
                "recall": float(row.get("Recall", 0.0)),
                "f1": float(row.get("F1", 0.0)),
                "support": int(row.get("Support", 0)),
                "total": int(row.get("Total_Samples", 0)),
            })
    else:
        for r in list(explainer_or_rules):
            if cluster is None or str(r.cluster) == str(cluster):
                rules_data.append({
                    "cluster": str(r.cluster),
                    "rule_string": r.rule_string,
                    "conditions": r.conditions,
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "support": r.support,
                    "total": r.total_samples,
                })

    if not rules_data:
        return "<p class='empty-note'>No decision rules available.</p>"

    cards = []
    for i, r in enumerate(rules_data, start=1):
        conds_html = ""
        for c in r["conditions"]:
            conds_html += f"<div class='rule-clause'><code>{html.escape(c)}</code></div>"

        prec_pct = r["precision"] * 100.0
        rec_pct = r["recall"] * 100.0

        card_html = f"""
        <div class="dt-rule-card">
            <div class="dt-rule-header">
                <span class="dt-cluster-badge">State {html.escape(r['cluster'])}</span>
                <span class="dt-rule-num">Rule #{i}</span>
            </div>
            <div class="dt-rule-body">
                <div class="dt-if-tag">IF</div>
                <div class="dt-clauses">{conds_html}</div>
                <div class="dt-then-tag">THEN <b>Conformational State {html.escape(r['cluster'])}</b></div>
            </div>
            <div class="dt-rule-metrics">
                <div class="dt-metric">
                    <span class="dt-metric-label">Precision</span>
                    <span class="dt-metric-val">{prec_pct:.1f}%</span>
                    <div class="dt-bar"><div class="dt-fill prec" style="width: {prec_pct:.1f}%;"></div></div>
                </div>
                <div class="dt-metric">
                    <span class="dt-metric-label">Recall</span>
                    <span class="dt-metric-val">{rec_pct:.1f}%</span>
                    <div class="dt-bar"><div class="dt-fill rec" style="width: {rec_pct:.1f}%;"></div></div>
                </div>
                <div class="dt-metric">
                    <span class="dt-metric-label">Support</span>
                    <span class="dt-metric-val">{r['support']} / {r['total']} frames</span>
                </div>
            </div>
        </div>
        """
        cards.append(card_html)

    return f"<div class='dt-rules-container'>{''.join(cards)}</div>"
