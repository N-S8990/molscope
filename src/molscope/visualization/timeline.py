"""Cluster timeline and conformational state transition visualization module.

Visualizes cluster membership and state trajectories over simulation time,
and computes and plots empirical Markovian state transition matrices.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


def _resolve_column(df: pd.DataFrame, candidates: Sequence[str], default: str | None = None) -> str | None:
    """Find the first matching column name in df from candidates (case-insensitive)."""
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    return default


def compute_transition_matrix(
    state_assignments: pd.DataFrame,
    state_col: str | None = None,
    trajectory_col: str | None = None,
    normalize: bool = True,
) -> pd.DataFrame:
    """Compute empirical 1-step state transition count or probability matrix from sequential trajectory frames.

    Parameters
    ----------
    state_assignments : pd.DataFrame
        Table with frame-by-frame state or cluster assignments.
    state_col : str | None, default=None
        Column holding state identifiers. If None, auto-detected from ("State", "State_Label", "Cluster", "cluster_id").
    trajectory_col : str | None, default=None
        Column identifying independent trajectories. If None, auto-detected from ("trajectory_id", "replica_id", "trajectory").
    normalize : bool, default=True
        Whether to row-normalize counts into transition probabilities P(S_{t+1}=j | S_t=i).

    Returns
    -------
    pd.DataFrame
        Square matrix of transition probabilities or counts, indexed and columned by state labels.
    """
    df = state_assignments.copy()
    s_col = state_col or _resolve_column(df, ["State", "State_Label", "Cluster", "cluster_id", "state_id"])
    if s_col is None or s_col not in df.columns:
        raise KeyError(f"State column not found in DataFrame. Available: {list(df.columns)}")

    traj_col = trajectory_col or _resolve_column(df, ["trajectory_id", "replica_id", "trajectory", "replica"])

    unique_states = sorted(df[s_col].dropna().unique(), key=lambda v: (str(type(v)), v))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    n_states = len(unique_states)
    counts = np.zeros((n_states, n_states), dtype=float)

    if traj_col is not None and traj_col in df.columns:
        grouped = df.groupby(traj_col, sort=False)
        for _, group in grouped:
            series = group[s_col].dropna().to_numpy()
            if len(series) >= 2:
                from_states = series[:-1]
                to_states = series[1:]
                for f_st, t_st in zip(from_states, to_states):
                    if f_st in state_to_idx and t_st in state_to_idx:
                        counts[state_to_idx[f_st], state_to_idx[t_st]] += 1.0
    else:
        series = df[s_col].dropna().to_numpy()
        if len(series) >= 2:
            from_states = series[:-1]
            to_states = series[1:]
            for f_st, t_st in zip(from_states, to_states):
                if f_st in state_to_idx and t_st in state_to_idx:
                    counts[state_to_idx[f_st], state_to_idx[t_st]] += 1.0

    if normalize:
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero for unvisited or transient states
        with np.errstate(divide="ignore", invalid="ignore"):
            prob_matrix = np.where(row_sums > 0, counts / row_sums, 0.0)
        matrix = prob_matrix
    else:
        matrix = counts

    labels = [f"State {s}" if isinstance(s, (int, np.integer)) else str(s) for s in unique_states]
    return pd.DataFrame(matrix, index=labels, columns=labels)


def plot_cluster_timeline(
    state_assignments: pd.DataFrame,
    output_path: str | Path,
    time_col: str | None = None,
    state_col: str | None = None,
    trajectory_col: str | None = None,
    title: str = "Cluster Membership Timeline & State Transitions",
    time_units: str = "ps",
    palette: str = "tab10",
    dpi: int = 300,
    figsize: tuple[float, float] | None = None,
) -> Path:
    """Plot cluster membership over simulation time across trajectory replicas.

    Parameters
    ----------
    state_assignments : pd.DataFrame
        DataFrame with cluster assignments over time.
    output_path : str or Path
        Target destination for output image (.png, .pdf, .svg).
    time_col : str | None, default=None
        Time or frame index column. Auto-detected from ("time_ps", "time_ns", "time", "frame_index", "frame_order").
    state_col : str | None, default=None
        Cluster/state column. Auto-detected from ("State", "State_Label", "Cluster", "cluster_id").
    trajectory_col : str | None, default=None
        Trajectory ID column. Auto-detected from ("trajectory_id", "replica_id", "trajectory").
    title : str, default="Cluster Membership Timeline & State Transitions"
        Figure title.
    time_units : str, default="ps"
        Label for time axis (e.g. "ps", "ns", or "frames").
    palette : str, default="tab10"
        Matplotlib qualitative colormap name.
    dpi : int, default=300
        Image resolution.
    figsize : tuple[float, float] | None, default=None
        Figure dimensions.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    df = state_assignments.copy()
    s_col = state_col or _resolve_column(df, ["State", "State_Label", "Cluster", "cluster_id", "state_id"])
    if s_col is None or s_col not in df.columns:
        raise KeyError(f"State column not found in DataFrame. Available: {list(df.columns)}")

    t_col = time_col or _resolve_column(df, ["time_ps", "time_ns", "time", "frame_index", "frame_order"])
    if t_col is None or t_col not in df.columns:
        df["_frame_idx"] = np.arange(len(df))
        t_col = "_frame_idx"
        time_units = "frames"
    else:
        # Detect units from column name if default
        if "ns" in t_col.lower():
            time_units = "ns"
        elif "ps" in t_col.lower():
            time_units = "ps"
        elif "frame" in t_col.lower():
            time_units = "frames"

    traj_col = trajectory_col or _resolve_column(df, ["trajectory_id", "replica_id", "trajectory", "replica"])

    unique_states = sorted(df[s_col].dropna().unique(), key=lambda v: (str(type(v)), v))
    n_states = len(unique_states)
    cmap = plt.get_cmap(palette, max(n_states, 2))
    state_colors = {s: cmap(i) for i, s in enumerate(unique_states)}

    trajectories = df[traj_col].dropna().unique() if (traj_col and traj_col in df.columns) else ["All Frames"]
    n_trajs = len(trajectories)

    calc_height = max(4.0, n_trajs * 2.2)
    calc_width = 10.0
    fig_size = figsize or (calc_width, calc_height)

    fig, axes = plt.subplots(
        nrows=n_trajs,
        ncols=1,
        figsize=fig_size,
        sharex=True,
        squeeze=False,
    )

    for idx, traj_id in enumerate(trajectories):
        ax = axes[idx, 0]
        if traj_col and traj_col in df.columns:
            sub = df[df[traj_col] == traj_id].copy()
        else:
            sub = df.copy()

        sub = sub.sort_values(t_col).reset_index(drop=True)
        times = sub[t_col].to_numpy(dtype=float)
        states = sub[s_col].to_numpy()

        # Step line representing state transitions
        y_numeric = np.array([unique_states.index(s) if s in unique_states else np.nan for s in states])
        ax.step(times, y_numeric, where="post", color="#4a5568", alpha=0.55, linewidth=1.2, zorder=2)

        # Color-coded scatter markers at each frame
        for s_idx, st in enumerate(unique_states):
            mask = (states == st)
            if np.any(mask):
                label_txt = f"State {st}" if isinstance(st, (int, np.integer)) else str(st)
                # Compute population percentage
                pct = (np.sum(mask) / len(states)) * 100.0
                ax.scatter(
                    times[mask],
                    y_numeric[mask],
                    color=state_colors[st],
                    s=18,
                    alpha=0.85,
                    edgecolors="none",
                    label=f"{label_txt} ({pct:.1f}%)" if idx == 0 else None,
                    zorder=3,
                )

        ax.set_yticks(range(n_states))
        ax.set_yticklabels([f"State {s}" if isinstance(s, (int, np.integer)) else str(s) for s in unique_states])
        ax.grid(True, linestyle="--", alpha=0.3, axis="y")
        ax.set_ylabel(f"{traj_id}" if n_trajs > 1 else "State", fontsize=10, fontweight="medium")

    axes[-1, 0].set_xlabel(f"Simulation Time ({time_units})", fontsize=11, fontweight="medium")

    # Add shared legend at the top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.98),
            ncol=min(n_states, 6),
            frameon=False,
            fontsize=9,
        )

    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02 if n_trajs == 1 else 1.0)
    fig.tight_layout()

    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_p


def plot_state_transitions(
    transition_matrix_or_assignments: pd.DataFrame,
    output_path: str | Path,
    state_col: str | None = None,
    trajectory_col: str | None = None,
    title: str = "Empirical State Transition Matrix",
    cmap: str = "Blues",
    dpi: int = 300,
    figsize: tuple[float, float] = (7.0, 6.0),
) -> Path:
    """Plot an empirical state transition probability matrix heatmap.

    Parameters
    ----------
    transition_matrix_or_assignments : pd.DataFrame
        Either a precomputed transition probability matrix (square DataFrame)
        or a state_assignments table.
    output_path : str or Path
        Target destination for output image.
    state_col : str | None, default=None
        Column holding state identifiers if state_assignments is provided.
    trajectory_col : str | None, default=None
        Trajectory ID column if state_assignments is provided.
    title : str, default="Empirical State Transition Matrix"
        Plot title.
    cmap : str, default="Blues"
        Heatmap colormap.
    dpi : int, default=300
        Image resolution.
    figsize : tuple[float, float], default=(7.0, 6.0)
        Figure size in inches.

    Returns
    -------
    Path
        Path to the saved figure.
    """
    df = transition_matrix_or_assignments
    # Check if df is already a square transition matrix
    is_square_matrix = (
        df.shape[0] == df.shape[1]
        and df.shape[0] > 1
        and np.all(np.issubdtype(df.dtypes.iloc[0], np.number))
    )

    if is_square_matrix:
        matrix_df = df
    else:
        matrix_df = compute_transition_matrix(
            df,
            state_col=state_col,
            trajectory_col=trajectory_col,
            normalize=True,
        )

    mat_values = matrix_df.to_numpy(dtype=float)
    n = len(matrix_df)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(mat_values, cmap=cmap, vmin=0.0, vmax=1.0)

    # Format ticks and labels
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(matrix_df.columns, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(matrix_df.index, fontsize=10)
    ax.set_xlabel("Transition To (t + 1)", fontsize=11, fontweight="medium")
    ax.set_ylabel("Transition From (t)", fontsize=11, fontweight="medium")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    # Annotate cell probabilities
    for i in range(n):
        for j in range(n):
            val = mat_values[i, j]
            color = "white" if val > 0.55 else "#111111"
            ax.text(
                j,
                i,
                f"{val:.3f}\n({val*100:.1f}%)",
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="medium",
            )

    cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label("Transition Probability P(To | From)", fontsize=10)

    fig.tight_layout()
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_p, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return out_p


def generate_cluster_timeline_plots(
    state_assignments: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "cluster_timeline",
    time_col: str | None = None,
    state_col: str | None = None,
    trajectory_col: str | None = None,
    **kwargs: Any,
) -> dict[str, Path]:
    """Generate both cluster membership timeline and state transition probability matrix plots.

    Parameters
    ----------
    state_assignments : pd.DataFrame
        Table with frame-by-frame state or cluster assignments.
    output_dir : str or Path
        Target folder for generated figures and transition table.
    prefix : str, default="cluster_timeline"
        Base prefix for output filenames.
    time_col : str | None, default=None
        Time column.
    state_col : str | None, default=None
        State column.
    trajectory_col : str | None, default=None
        Trajectory ID column.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys 'timeline', 'transitions', and 'transition_matrix_csv'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    timeline_path = out_dir / f"{prefix}.png"
    plot_cluster_timeline(
        state_assignments=state_assignments,
        output_path=timeline_path,
        time_col=time_col,
        state_col=state_col,
        trajectory_col=trajectory_col,
        **{k: v for k, v in kwargs.items() if k in ("title", "time_units", "palette", "dpi", "figsize")},
    )

    matrix_df = compute_transition_matrix(
        state_assignments=state_assignments,
        state_col=state_col,
        trajectory_col=trajectory_col,
        normalize=True,
    )
    matrix_csv_path = out_dir / f"{prefix}_transition_matrix.csv"
    matrix_df.to_csv(matrix_csv_path)

    transitions_path = out_dir / f"{prefix}_transitions.png"
    plot_state_transitions(
        transition_matrix_or_assignments=matrix_df,
        output_path=transitions_path,
        **{k: v for k, v in kwargs.items() if k in ("cmap", "dpi", "figsize")},
    )

    return {
        "timeline": timeline_path,
        "transitions": transitions_path,
        "transition_matrix_csv": matrix_csv_path,
    }
