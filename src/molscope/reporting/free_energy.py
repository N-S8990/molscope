"""Free Energy Landscape (FES) and Potential of Mean Force (PMF) reporting module.

Computes 2D empirical free energy surfaces from molecular simulation coordinates,
identifies conformational free energy basins (local minima), and generates
publication-quality static (Matplotlib) and interactive (Plotly) contour visualizations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import scipy.ndimage
import scipy.stats

# Boltzmann constant in various thermodynamic energy units
BOLTZMANN_KCAL: float = 0.001987204  # kcal / (mol * K)
BOLTZMANN_KJ: float = 0.008314462    # kJ / (mol * K)

BOLTZMANN_CONSTANTS: dict[str, float] = {
    "kcal/mol": BOLTZMANN_KCAL,
    "kj/mol": BOLTZMANN_KJ,
}


def _normalize_units(units: str) -> tuple[str, float]:
    """Validate and normalize thermodynamic energy units."""
    u = units.strip().lower()
    if u in ("kcal/mol", "kcal"):
        return "kcal/mol", BOLTZMANN_KCAL
    elif u in ("kj/mol", "kj"):
        return "kJ/mol", BOLTZMANN_KJ
    raise ValueError(
        f"Unsupported energy units: '{units}'. MolScope supports 'kcal/mol' and 'kJ/mol'."
    )


def _detect_conformational_basins(
    x_centers: np.ndarray,
    y_centers: np.ndarray,
    density: np.ndarray,
    delta_g: np.ndarray,
    min_peak_density_ratio: float = 0.02,
    footprint_size: int = 5,
    min_distance_bins: int = 4,
    max_energy_cutoff: float | None = None,
) -> pd.DataFrame:
    """Identify local free energy minima (conformational basins) on the 2D surface.

    Parameters
    ----------
    x_centers : np.ndarray
        1D array of bin centers along X.
    y_centers : np.ndarray
        1D array of bin centers along Y.
    density : np.ndarray
        2D probability density matrix, shape (len(y_centers), len(x_centers)).
    delta_g : np.ndarray
        2D potential of mean force matrix, shape (len(y_centers), len(x_centers)).
    min_peak_density_ratio : float, default=0.02
        Minimum peak density relative to p_max to be considered a stable basin.
    footprint_size : int, default=5
        Local neighborhood window size (in bins) for peak detection.
    min_distance_bins : int, default=4
        Minimum Euclidean distance (in grid bins) separating independent basins.
    max_energy_cutoff : float | None, default=None
        Maximum free energy threshold; basins higher than this are omitted.

    Returns
    -------
    pd.DataFrame
        Table of detected conformational basins with columns:
        ["basin_id", "x", "y", "free_energy", "probability"]
    """
    p_max = float(np.nanmax(density)) if density.size > 0 else 0.0
    if p_max <= 0:
        return pd.DataFrame(columns=["basin_id", "x", "y", "free_energy", "probability"])

    # Mild smoothing on density for stable local extrema detection
    p_eval = scipy.ndimage.gaussian_filter(density, sigma=0.8)
    footprint = np.ones((footprint_size, footprint_size), dtype=bool)
    max_filtered = scipy.ndimage.maximum_filter(p_eval, footprint=footprint, mode="constant", cval=0.0)

    # Candidate local maxima in probability density (minima in free energy)
    min_density = p_max * min_peak_density_ratio
    is_candidate = (p_eval == max_filtered) & (p_eval >= min_density) & np.isfinite(delta_g)

    if max_energy_cutoff is not None:
        is_candidate = is_candidate & (delta_g <= max_energy_cutoff)

    candidate_y, candidate_x = np.where(is_candidate)

    # Sort candidates by free energy ascending (lowest free energy / deepest basin first)
    candidates = []
    for cy, cx in zip(candidate_y, candidate_x):
        candidates.append((int(cy), int(cx), float(delta_g[cy, cx]), float(density[cy, cx])))
    candidates.sort(key=lambda item: item[2])

    # Filter out redundant candidate peaks that fall within min_distance_bins of a deeper basin
    kept: list[tuple[int, int, float, float]] = []
    for cy, cx, dg, prob in candidates:
        too_close = False
        for ky, kx, _, _ in kept:
            dist = np.hypot(cy - ky, cx - kx)
            if dist < min_distance_bins:
                too_close = True
                break
        if not too_close:
            kept.append((cy, cx, dg, prob))

    records = []
    for idx, (cy, cx, dg, prob) in enumerate(kept, start=1):
        records.append({
            "basin_id": idx,
            "x": float(x_centers[cx]),
            "y": float(y_centers[cy]),
            "free_energy": float(dg),
            "probability": float(prob),
        })

    return pd.DataFrame(records, columns=["basin_id", "x", "y", "free_energy", "probability"])


@dataclass
class FreeEnergyLandscapeResult:
    """Encapsulates a computed 2D Free Energy Landscape (FES) and its basins.

    Attributes
    ----------
    x_centers : np.ndarray
        1D array of bin centers along X (length Nx).
    y_centers : np.ndarray
        1D array of bin centers along Y (length Ny).
    x_grid : np.ndarray
        2D meshgrid of X coordinates, shape (Ny, Nx).
    y_grid : np.ndarray
        2D meshgrid of Y coordinates, shape (Ny, Nx).
    density : np.ndarray
        2D probability density P(x, y), shape (Ny, Nx).
    delta_g : np.ndarray
        2D potential of mean force Delta G(x, y) = -k_B * T * ln(P/P_max), shape (Ny, Nx).
    temperature_k : float
        Simulation temperature in Kelvin.
    units : str
        Thermodynamic energy units ("kcal/mol" or "kJ/mol").
    basins : pd.DataFrame
        Detected local conformational minima with coordinates, Delta G, and density.
    method : str
        Density estimation method ("histogram" or "kde").
    smooth : float | None
        Gaussian smoothing parameter applied to histogram density, if any.
    bandwidth : float | None
        Kernel bandwidth used for Gaussian KDE, if applicable.
    """

    x_centers: np.ndarray
    y_centers: np.ndarray
    x_grid: np.ndarray
    y_grid: np.ndarray
    density: np.ndarray
    delta_g: np.ndarray
    temperature_k: float
    units: str
    basins: pd.DataFrame
    method: str = "histogram"
    smooth: float | None = None
    bandwidth: float | None = None

    @property
    def min_free_energy(self) -> float:
        """Global minimum free energy (0.0 by construction)."""
        finite = self.delta_g[np.isfinite(self.delta_g)]
        return float(np.min(finite)) if len(finite) > 0 else 0.0

    @property
    def max_free_energy(self) -> float:
        """Maximum finite free energy across the sampled landscape."""
        finite = self.delta_g[np.isfinite(self.delta_g)]
        return float(np.max(finite)) if len(finite) > 0 else 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert result fields to a dictionary."""
        return {
            "x_centers": self.x_centers,
            "y_centers": self.y_centers,
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "density": self.density,
            "delta_g": self.delta_g,
            "temperature_k": self.temperature_k,
            "units": self.units,
            "basins": self.basins,
            "method": self.method,
            "smooth": self.smooth,
            "bandwidth": self.bandwidth,
        }

    def __getitem__(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Key '{key}' not found in FreeEnergyLandscapeResult")

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def plot_static(
        self,
        output_path: str | Path,
        xlabel: str = "CV 1",
        ylabel: str = "CV 2",
        title: str = "Free Energy Landscape",
        **kwargs: Any,
    ) -> Path:
        """Convenience wrapper to render a publication-quality static contour plot."""
        return plot_landscape_static(self, output_path, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)

    def plot_interactive(
        self,
        output_path: str | Path,
        xlabel: str = "CV 1",
        ylabel: str = "CV 2",
        title: str = "Free Energy Landscape",
        **kwargs: Any,
    ) -> Path:
        """Convenience wrapper to render an interactive Plotly HTML contour plot."""
        return plot_landscape_interactive(self, output_path, xlabel=xlabel, ylabel=ylabel, title=title, **kwargs)


def compute_free_energy_landscape(
    x: np.ndarray | Sequence[float],
    y: np.ndarray | Sequence[float],
    temperature_k: float = 300.0,
    bins: int | tuple[int, int] = 50,
    units: str = "kcal/mol",
    method: str = "histogram",
    smooth: float | None = 1.0,
    bandwidth: float | None = None,
    min_density: float = 0.0,
    max_energy_cutoff: float | None = None,
) -> FreeEnergyLandscapeResult:
    """Compute the 2D potential of mean force (PMF) free energy landscape.

    Parameters
    ----------
    x : np.ndarray or Sequence[float]
        Sample coordinates along the first dimension / reaction coordinate (e.g. PC1, TIC_1, torsion).
    y : np.ndarray or Sequence[float]
        Sample coordinates along the second dimension / reaction coordinate (e.g. PC2, TIC_2, distance).
    temperature_k : float, default=300.0
        Thermodynamic temperature in Kelvin.
    bins : int or tuple[int, int], default=50
        Number of grid bins along each dimension.
    units : str, default="kcal/mol"
        Energy units for potential of mean force: "kcal/mol" or "kJ/mol".
    method : str, default="histogram"
        Density estimation method: "histogram" (fast, scalable 2D histogram)
        or "kde" (Gaussian kernel density estimation).
    smooth : float | None, default=1.0
        Gaussian smoothing kernel standard deviation (in bins) applied to the
        histogram density for smooth potential of mean force contours. Set to None or 0 to disable.
    bandwidth : float | None, default=None
        Bandwidth factor for Gaussian KDE if method="kde".
    min_density : float, default=0.0
        Threshold below which probability density is considered unvisited (free energy set to NaN).
    max_energy_cutoff : float | None, default=None
        Optional maximum free energy cutoff for basin detection.

    Returns
    -------
    FreeEnergyLandscapeResult
        Container holding grid coordinates, density, Delta G surface, detected basins, and metadata.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()

    if len(x_arr) != len(y_arr):
        raise ValueError(
            f"Coordinates x and y must have identical lengths; got {len(x_arr)} and {len(y_arr)}."
        )
    if len(x_arr) < 3:
        raise ValueError(f"Free energy estimation requires at least 3 points, got {len(x_arr)}.")

    # Filter out non-finite coordinates
    finite_mask = np.isfinite(x_arr) & np.isfinite(y_arr)
    if not np.all(finite_mask):
        x_arr = x_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        if len(x_arr) < 3:
            raise ValueError("Too few finite (x, y) coordinates for free energy landscape computation.")

    if temperature_k <= 0:
        raise ValueError(f"Temperature must be strictly positive, got {temperature_k} K.")

    unit_name, k_b = _normalize_units(units)

    # Resolve bins
    if isinstance(bins, int):
        if bins < 2:
            raise ValueError(f"bins must be at least 2, got {bins}.")
        bins_x, bins_y = bins, bins
    elif isinstance(bins, (tuple, list)) and len(bins) == 2:
        bins_x, bins_y = int(bins[0]), int(bins[1])
        if bins_x < 2 or bins_y < 2:
            raise ValueError(f"bins dimensions must be at least 2, got {bins}.")
    else:
        raise ValueError(f"bins must be an int or a tuple of two ints, got {bins}.")

    norm_method = method.strip().lower()
    if norm_method == "histogram":
        hist, xedges, yedges = np.histogram2d(x_arr, y_arr, bins=[bins_x, bins_y], density=True)
        x_centers = 0.5 * (xedges[:-1] + xedges[1:])
        y_centers = 0.5 * (yedges[:-1] + yedges[1:])

        # hist shape is (bins_x, bins_y), where hist[i, j] corresponds to x_centers[i], y_centers[j].
        # In meshgrid(x_centers, y_centers), shape is (bins_y, bins_x).
        # We transpose hist so density matches meshgrid orientation: density[j, i] = hist[i, j].
        density = hist.T.copy()

        if smooth is not None and smooth > 0:
            density = scipy.ndimage.gaussian_filter(density, sigma=float(smooth), mode="nearest")
            dx = float(x_centers[1] - x_centers[0]) if len(x_centers) > 1 else 1.0
            dy = float(y_centers[1] - y_centers[0]) if len(y_centers) > 1 else 1.0
            total_prob = np.sum(density) * dx * dy
            if total_prob > 0:
                density = density / total_prob

        x_grid, y_grid = np.meshgrid(x_centers, y_centers)

    elif norm_method == "kde":
        x_min, x_max = float(np.min(x_arr)), float(np.max(x_arr))
        y_min, y_max = float(np.min(y_arr)), float(np.max(y_arr))
        x_margin = (x_max - x_min) * 0.05 if x_max > x_min else 0.1
        y_margin = (y_max - y_min) * 0.05 if y_max > y_min else 0.1

        x_centers = np.linspace(x_min - x_margin, x_max + x_margin, bins_x)
        y_centers = np.linspace(y_min - y_margin, y_max + y_margin, bins_y)
        x_grid, y_grid = np.meshgrid(x_centers, y_centers)

        kde = scipy.stats.gaussian_kde(np.vstack([x_arr, y_arr]), bw_method=bandwidth)
        eval_points = np.vstack([x_grid.ravel(), y_grid.ravel()])
        density = kde(eval_points).reshape(x_grid.shape)

    else:
        raise ValueError(
            f"Unsupported density estimation method: '{method}'. Expected 'histogram' or 'kde'."
        )

    p_max = float(np.nanmax(density))
    if p_max <= 0:
        raise ValueError("Estimated probability density is zero everywhere.")

    # Potential of mean force: Delta G(x, y) = -k_B * T * ln(P(x, y) / P_max)
    delta_g = np.full_like(density, np.nan, dtype=float)
    valid = density > max(min_density, 0.0)
    delta_g[valid] = -k_b * temperature_k * np.log(density[valid] / p_max)

    # Eliminate potential machine-precision negative values (-0.0)
    delta_g[valid] = np.maximum(delta_g[valid], 0.0)

    # Detect conformational basins (local minima in Delta G)
    basins = _detect_conformational_basins(
        x_centers=x_centers,
        y_centers=y_centers,
        density=density,
        delta_g=delta_g,
        max_energy_cutoff=max_energy_cutoff,
    )

    return FreeEnergyLandscapeResult(
        x_centers=x_centers,
        y_centers=y_centers,
        x_grid=x_grid,
        y_grid=y_grid,
        density=density,
        delta_g=delta_g,
        temperature_k=temperature_k,
        units=unit_name,
        basins=basins,
        method=norm_method,
        smooth=smooth,
        bandwidth=bandwidth,
    )


def plot_landscape_static(
    result: FreeEnergyLandscapeResult,
    output_path: str | Path,
    xlabel: str = "CV 1",
    ylabel: str = "CV 2",
    title: str = "Free Energy Landscape",
    levels: int = 30,
    cmap: str = "viridis_r",
    max_energy: float | None = None,
    show_basins: bool = True,
    show_contours: bool = True,
    dpi: int = 300,
    figsize: tuple[float, float] = (8.5, 7.0),
) -> Path:
    """Render a publication-quality static contour plot of the free energy landscape.

    Parameters
    ----------
    result : FreeEnergyLandscapeResult
        Computed free energy surface.
    output_path : str or Path
        Target file path for the exported image (.png, .pdf, or .svg).
    xlabel : str, default="CV 1"
        Label for the horizontal axis.
    ylabel : str, default="CV 2"
        Label for the vertical axis.
    title : str, default="Free Energy Landscape"
        Title displayed above the plot.
    levels : int, default=30
        Number of free energy contour levels.
    cmap : str, default="viridis_r"
        Matplotlib colormap (e.g. "viridis_r", "turbo_r", "plasma_r").
    max_energy : float | None, default=None
        Maximum free energy ceiling to plot. Values above this are masked out.
    show_basins : bool, default=True
        Whether to overlay star markers and labels for detected conformational basins.
    show_contours : bool, default=True
        Whether to overlay thin contour line boundaries on top of filled contours.
    dpi : int, default=300
        Image resolution in dots per inch.
    figsize : tuple[float, float], default=(8.5, 7.0)
        Matplotlib figure dimensions in inches.

    Returns
    -------
    Path
        Resolved path to the saved static image.
    """
    dg_plot = result.delta_g.copy()
    if max_energy is not None:
        dg_plot = np.where(dg_plot > max_energy, np.nan, dg_plot)

    finite_vals = dg_plot[np.isfinite(dg_plot)]
    if len(finite_vals) == 0:
        max_val = 1.0
    else:
        max_val = float(np.max(finite_vals))

    contour_levels = np.linspace(0.0, max_val, max(levels, 5))

    fig, ax = plt.subplots(figsize=figsize)

    # Filled contour landscape
    cf = ax.contourf(
        result.x_grid,
        result.y_grid,
        dg_plot,
        levels=contour_levels,
        cmap=cmap,
        extend="max" if max_energy is not None else "neither",
    )

    if show_contours:
        step = max(1, len(contour_levels) // 10)
        ax.contour(
            result.x_grid,
            result.y_grid,
            dg_plot,
            levels=contour_levels[::step],
            colors="black",
            alpha=0.25,
            linewidths=0.6,
        )

    cbar = fig.colorbar(cf, ax=ax, shrink=0.88, pad=0.03)
    cbar.set_label(rf"$\Delta G$ ({result.units})", fontsize=11, fontweight="medium")

    # Annotate detected conformational basins
    if show_basins and not result.basins.empty:
        ax.scatter(
            result.basins["x"],
            result.basins["y"],
            marker="*",
            color="crimson",
            edgecolors="white",
            linewidths=1.0,
            s=150,
            zorder=6,
            label="Conformational Basins",
        )
        for row in result.basins.itertuples():
            ax.annotate(
                f"B{int(row.basin_id)} ({row.free_energy:.2f})",
                (row.x, row.y),
                xytext=(7, 7),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold",
                color="#111111",
                bbox=dict(
                    boxstyle="round,pad=0.25",
                    facecolor="white",
                    alpha=0.85,
                    edgecolor="#666666",
                    linewidth=0.6,
                ),
                zorder=7,
            )

    ax.set_xlabel(xlabel, fontsize=12, labelpad=8)
    ax.set_ylabel(ylabel, fontsize=12, labelpad=8)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    fig.tight_layout()

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return out_file


def plot_landscape_interactive(
    result: FreeEnergyLandscapeResult,
    output_path: str | Path,
    xlabel: str = "CV 1",
    ylabel: str = "CV 2",
    title: str = "Free Energy Landscape",
    colorscale: str = "Viridis",
    max_energy: float | None = None,
    show_basins: bool = True,
    n_contours: int = 30,
) -> Path:
    """Render an interactive Plotly HTML contour plot of the free energy landscape.

    Parameters
    ----------
    result : FreeEnergyLandscapeResult
        Computed free energy surface.
    output_path : str or Path
        Target file path for the exported HTML file.
    xlabel : str, default="CV 1"
        Label for the horizontal axis.
    ylabel : str, default="CV 2"
        Label for the vertical axis.
    title : str, default="Free Energy Landscape"
        Title displayed on the chart.
    colorscale : str, default="Viridis"
        Plotly colorscale (e.g. "Viridis", "Plasma", "Turbo", "RdBu_r").
    max_energy : float | None, default=None
        Maximum free energy ceiling. Values above this are masked out.
    show_basins : bool, default=True
        Whether to overlay interactive star markers for conformational basins.
    n_contours : int, default=30
        Number of contour lines/levels.

    Returns
    -------
    Path
        Resolved path to the saved interactive HTML file.
    """
    dg_plot = result.delta_g.copy()
    if max_energy is not None:
        dg_plot = np.where(dg_plot > max_energy, np.nan, dg_plot)

    fig = go.Figure()

    # 2D contour trace
    fig.add_trace(
        go.Contour(
            x=result.x_centers,
            y=result.y_centers,
            z=dg_plot,
            colorscale=colorscale,
            ncontours=n_contours,
            colorbar=dict(
                title=dict(text=f"ΔG ({result.units})", side="right"),
            ),
            contours=dict(
                coloring="heatmap",
                showlines=True,
            ),
            hovertemplate=(
                f"<b>{xlabel}</b>: %{{x:.3f}}<br>"
                f"<b>{ylabel}</b>: %{{y:.3f}}<br>"
                f"<b>ΔG</b>: %{{z:.2f}} {result.units}<extra></extra>"
            ),
            name="ΔG Landscape",
        )
    )

    # Basin markers
    if show_basins and not result.basins.empty:
        hover_texts = [
            f"<b>Basin {int(r.basin_id)}</b><br>"
            f"{xlabel}: {r.x:.3f}<br>"
            f"{ylabel}: {r.y:.3f}<br>"
            f"ΔG: {r.free_energy:.2f} {result.units}<br>"
            f"P: {r.probability:.4e}"
            for r in result.basins.itertuples()
        ]
        labels = [f"B{int(r.basin_id)}" for r in result.basins.itertuples()]

        fig.add_trace(
            go.Scatter(
                x=result.basins["x"],
                y=result.basins["y"],
                mode="markers+text",
                text=labels,
                textposition="top center",
                textfont=dict(size=12, color="#111111", family="Arial, sans-serif"),
                marker=dict(
                    size=13,
                    symbol="star",
                    color="#E63946",
                    line=dict(color="white", width=1.5),
                ),
                hoverinfo="text",
                hovertext=hover_texts,
                name="Conformational Basins",
            )
        )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", x=0.5, font=dict(size=16)),
        xaxis=dict(title=xlabel, zeroline=False),
        yaxis=dict(title=ylabel, zeroline=False),
        template="plotly_white",
        width=900,
        height=720,
        margin=dict(l=60, r=60, t=70, b=60),
    )

    out_file = Path(output_path).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(out_file), include_plotlyjs=True, full_html=True)

    return out_file


def compute_pca_free_energy_landscape(
    representation_coordinates: pd.DataFrame | np.ndarray,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    temperature_k: float = 300.0,
    bins: int | tuple[int, int] = 50,
    units: str = "kcal/mol",
    method: str = "histogram",
    smooth: float | None = 1.0,
    bandwidth: float | None = None,
    min_density: float = 0.0,
    max_energy_cutoff: float | None = None,
) -> FreeEnergyLandscapeResult:
    """Compute 2D potential of mean force free energy surface on leading PCA components.

    Parameters
    ----------
    representation_coordinates : pd.DataFrame or np.ndarray
        DataFrame or array containing projection coordinates along principal components.
    pc_x : str, default="PC1"
        Name or column label of the first principal component (horizontal axis).
    pc_y : str, default="PC2"
        Name or column label of the second principal component (vertical axis).
    temperature_k : float, default=300.0
        Simulation temperature in Kelvin.
    bins : int or tuple[int, int], default=50
        Number of grid bins along each dimension.
    units : str, default="kcal/mol"
        Thermodynamic energy units ("kcal/mol" or "kJ/mol").
    method : str, default="histogram"
        Density estimation method: "histogram" or "kde".
    smooth : float | None, default=1.0
        Gaussian smoothing kernel standard deviation (in bins).
    bandwidth : float | None, default=None
        Bandwidth factor for Gaussian KDE if method="kde".
    min_density : float, default=0.0
        Threshold below which probability density is considered unvisited.
    max_energy_cutoff : float | None, default=None
        Optional maximum free energy cutoff for basin detection.

    Returns
    -------
    FreeEnergyLandscapeResult
        Computed free energy surface with coordinates, Delta G grid, and detected basins.
    """
    if isinstance(representation_coordinates, pd.DataFrame):
        df = representation_coordinates
        # Resolve x coordinate
        col_x = None
        for cand in (pc_x, pc_x.upper(), pc_x.lower(), f"PC_{pc_x[-1]}" if pc_x[-1].isdigit() else pc_x):
            if cand in df.columns:
                col_x = cand
                break
        if col_x is None:
            if df.shape[1] >= 1:
                col_x = df.columns[0]
            else:
                raise KeyError(f"Coordinate column '{pc_x}' not found in DataFrame.")

        # Resolve y coordinate
        col_y = None
        for cand in (pc_y, pc_y.upper(), pc_y.lower(), f"PC_{pc_y[-1]}" if pc_y[-1].isdigit() else pc_y):
            if cand in df.columns:
                col_y = cand
                break
        if col_y is None:
            if df.shape[1] >= 2:
                col_y = df.columns[1]
            else:
                raise KeyError(f"Coordinate column '{pc_y}' not found in DataFrame.")

        x = df[col_x].to_numpy(dtype=float)
        y = df[col_y].to_numpy(dtype=float)
    else:
        arr = np.asarray(representation_coordinates, dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"representation_coordinates array must have shape (N, >=2), got {arr.shape}.")
        x = arr[:, 0]
        y = arr[:, 1]

    return compute_free_energy_landscape(
        x=x,
        y=y,
        temperature_k=temperature_k,
        bins=bins,
        units=units,
        method=method,
        smooth=smooth,
        bandwidth=bandwidth,
        min_density=min_density,
        max_energy_cutoff=max_energy_cutoff,
    )


def plot_pca_free_energy(
    representation_coordinates: pd.DataFrame | np.ndarray,
    output_dir: str | Path,
    pc_x: str = "PC1",
    pc_y: str = "PC2",
    prefix: str = "pca_free_energy_landscape",
    temperature_k: float = 300.0,
    units: str = "kcal/mol",
    title: str | None = None,
    interactive: bool = True,
    **kwargs: Any,
) -> dict[str, Path]:
    """Compute and plot 2D Free Energy Landscape on leading PCA coordinates.

    Exports publication-quality static PNG and interactive Plotly HTML contour plots.

    Parameters
    ----------
    representation_coordinates : pd.DataFrame or np.ndarray
        Projection coordinates on PCA components.
    output_dir : str or Path
        Target directory where plot files will be saved.
    pc_x : str, default="PC1"
        Component for horizontal axis.
    pc_y : str, default="PC2"
        Component for vertical axis.
    prefix : str, default="pca_free_energy_landscape"
        File prefix for exported figures.
    temperature_k : float, default=300.0
        Simulation temperature in Kelvin.
    units : str, default="kcal/mol"
        Free energy units ("kcal/mol" or "kJ/mol").
    title : str | None, default=None
        Custom plot title. Defaults to "Free Energy Landscape ({pc_x} vs {pc_y})".
    interactive : bool, default=True
        Whether to generate interactive Plotly HTML file in addition to static PNG.
    **kwargs : Any
        Additional parameters passed to compute_free_energy_landscape or plotting methods.

    Returns
    -------
    dict[str, Path]
        Dictionary with keys 'static', 'interactive' (if generated), and 'basins_csv'.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = compute_pca_free_energy_landscape(
        representation_coordinates=representation_coordinates,
        pc_x=pc_x,
        pc_y=pc_y,
        temperature_k=temperature_k,
        units=units,
        **{k: v for k, v in kwargs.items() if k in ("bins", "method", "smooth", "bandwidth", "min_density", "max_energy_cutoff")},
    )

    plot_title = title or f"Free Energy Landscape ({pc_x} vs {pc_y})"
    static_path = out_dir / f"{prefix}.png"
    result.plot_static(
        output_path=static_path,
        xlabel=pc_x,
        ylabel=pc_y,
        title=plot_title,
        **{k: v for k, v in kwargs.items() if k in ("levels", "cmap", "max_energy", "show_basins", "show_contours", "dpi", "figsize")},
    )

    outputs: dict[str, Path] = {"static": static_path}

    if interactive:
        html_path = out_dir / f"{prefix}.html"
        result.plot_interactive(
            output_path=html_path,
            xlabel=pc_x,
            ylabel=pc_y,
            title=plot_title,
            **{k: v for k, v in kwargs.items() if k in ("colorscale", "max_energy", "show_basins", "n_contours")},
        )
        outputs["interactive"] = html_path

    if not result.basins.empty:
        basins_path = out_dir / f"{prefix}_basins.csv"
        result.basins.to_csv(basins_path, index=False)
        outputs["basins_csv"] = basins_path

    return outputs
