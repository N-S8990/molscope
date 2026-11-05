"""CLI for the generic, trajectory-aware ML workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from molscope.analysis.predictive.workflow import MLWorkflowConfig, run_grouped_ml_workflow


def _feature_groups(path: str | None) -> dict[str, str] | None:
    if path is None:
        return None
    table = pd.read_csv(path)
    required = {"Feature", "Feature_Group"}
    if not required.issubset(table.columns):
        raise ValueError("Feature-group CSV must contain Feature and Feature_Group columns.")
    return dict(zip(table["Feature"], table["Feature_Group"]))


def _plot_lofo(importance: pd.DataFrame, path: Path, top_n: int = 25) -> None:
    if importance.empty:
        return
    plotted = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.28 * len(plotted))))
    ax.barh(plotted["Feature_Group"], plotted["Trajectory_Mean_Accuracy_Drop_Mean"], color="#1b9e77")
    ax.set(xlabel="Cross-fold trajectory-mean accuracy drop", title="LOFO / group-ablation predictive importance")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run leakage-aware grouped ML and LOFO feature ranking.")
    parser.add_argument("--features", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--target-column", required=True, help="Column in the manifest containing the user-defined class label.")
    parser.add_argument("--out-dir", default="ml_results")
    parser.add_argument("--feature-groups", default=None, help="Optional CSV mapping Feature to Feature_Group for group ablation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--rf-estimators", type=int, default=300)
    parser.add_argument("--bootstrap-samples", type=int, default=1_000)
    parser.add_argument("--skip-lofo", action="store_true")
    args = parser.parse_args()

    outputs = run_grouped_ml_workflow(
        pd.read_csv(args.features),
        pd.read_csv(args.manifest),
        args.target_column,
        MLWorkflowConfig(random_state=args.seed, n_splits=args.folds, rf_estimators=args.rf_estimators, bootstrap_samples=args.bootstrap_samples),
        run_lofo=not args.skip_lofo,
        feature_groups=_feature_groups(args.feature_groups),
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, table in outputs.items():
        table.to_csv(out_dir / f"{name}.csv", index=False)
    if "lofo_importance" in outputs:
        _plot_lofo(outputs["lofo_importance"], out_dir / "lofo_trajectory_accuracy_drop.png")
    report = {"target_column": args.target_column, "configuration": {"random_state": args.seed, "folds": args.folds, "rf_estimators": args.rf_estimators, "bootstrap_samples": args.bootstrap_samples}, "notes": ["Outer splits are trajectory-grouped.", "LOFO ranking uses trajectory-mean accuracy drop as the primary endpoint.", "Accuracy drops show predictive dependence, not causal mechanism."]}
    (out_dir / "ml_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote ML results to {out_dir}")


if __name__ == "__main__":
    main()
