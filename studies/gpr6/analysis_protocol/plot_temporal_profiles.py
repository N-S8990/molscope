"""Plot selected GPR6 measurements across both matched 100 ns windows.

Each thin line is one independent source replica averaged in 10 ns bins;
thick lines and bands are the mean and standard deviation of six replica-bin
means per condition. Bands are not confidence intervals and do not address
full-trajectory convergence.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1] / "runs"
FIRST = ROOT / "public_12_replica_expanded_first_100ns/gpr6_frame_features.csv"
SECOND = ROOT / "public_12_replica_expanded_second_100ns/gpr6_frame_features.csv"
OUTPUT = ROOT / "gpr6_temporal_feature_profiles.png"
MEASUREMENTS = (
    ("CA_DISTANCE_angstrom__X__PHE_152__X__TRP_292", "Phe152–Trp292 Cα distance (Å)"),
    ("CA_DISTANCE_angstrom__X__PHE_152__X__PHE_295", "Phe152–Phe295 Cα distance (Å)"),
    ("SIDECHAIN_chi2__chain_X__PHE_152__sin", "Phe152 χ2 sine"),
)
COLORS = {"no_added_inverse_agonist": "#245f8f", "3h_bound": "#d37b20"}


def main() -> None:
    first = pd.read_csv(FIRST)
    second = pd.read_csv(SECOND)
    if list(first.columns) != list(second.columns) or len(first) != 12_000 or len(second) != 12_000:
        raise ValueError("Expected matched 12,000-row expanded feature tables")
    data = pd.concat([first, second], ignore_index=True)
    if not all(feature in data for feature, _ in MEASUREMENTS):
        raise ValueError("Selected temporal measurements are missing")
    if not (data.groupby("trajectory_id").size() == 2000).all():
        raise ValueError("Each replica must contribute exactly 2000 frames")
    data["time_bin"] = np.floor(data["time_ps"] / 10_000).astype(int)
    data["bin_midpoint_ns"] = data["time_bin"] * 10 + 4.95
    values = [feature for feature, _ in MEASUREMENTS]
    binned = data.groupby(["condition", "trajectory_id", "time_bin", "bin_midpoint_ns"], as_index=False)[values].mean()
    if not (binned.groupby("trajectory_id").size() == 20).all():
        raise ValueError("Each replica must contribute twenty 10 ns bins")
    figure, axes = plt.subplots(3, 1, figsize=(10.5, 10), sharex=True)
    for ax, (feature, label) in zip(axes, MEASUREMENTS, strict=True):
        for condition, color in COLORS.items():
            subset = binned[binned["condition"] == condition]
            for _, replica in subset.groupby("trajectory_id"):
                ax.plot(replica["bin_midpoint_ns"], replica[feature], color=color, alpha=0.18, linewidth=0.8)
            summary = subset.groupby("bin_midpoint_ns")[feature].agg(["mean", "std"])
            ax.plot(summary.index, summary["mean"], color=color, linewidth=2.3,
                    label="3h bound" if condition == "3h_bound" else "No added inverse agonist")
            ax.fill_between(summary.index.to_numpy(dtype=float),
                            (summary["mean"] - summary["std"]).to_numpy(dtype=float),
                            (summary["mean"] + summary["std"]).to_numpy(dtype=float),
                            color=color, alpha=0.14)
        ax.axvline(100, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_ylabel(label)
        ax.grid(alpha=0.2)
    axes[0].legend(loc="upper right", frameon=False)
    axes[-1].set_xlabel("Production time (ns); 10 ns replica means")
    figure.suptitle("GPR6 temporal profiles across six replicas per condition")
    figure.text(0.05, 0.005, "Thin lines: replica means. Bands: ±1 SD across six replicas, not confidence intervals.\n"
                "Second window starts at 100 ns; source full-file MD5 checks remain pending.", fontsize=8)
    figure.tight_layout(rect=[0, 0.04, 1, 0.97])
    figure.savefig(OUTPUT, dpi=180)
    plt.close(figure)
    print(OUTPUT)


if __name__ == "__main__":
    main()
