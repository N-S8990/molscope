# SAM-MFP Benchmark and Validation Suite

This directory contains the benchmarking harness, raw validation datasets, and comparative results for **System-Aware Multi-Criteria Feature Prioritization (SAM-MFP)** in MolScope.

## Contents

- `run_benchmarks.ipynb`: Interactive Jupyter benchmark suite and comparative analysis. Evaluates:
  1. Synthetic planted-signal benchmark with known ground-truth features and collinear duplicates.
  2. Real-world 2RJY Villin Headpiece (HP-35) benchmark comparing SAM-MFP against empirical variance, mutual information, decision-tree Gini importance, and random ranking.
  3. Systematic objective ablation studies (ablating Uniqueness, ESS Ratio, Occupancy Entropy, PCA Salience).
- `sam_mfp_benchmark_summary.json`: Complete JSON summary with timings, metrics, and diagnostics.
- `synthetic_baseline_comparison.csv`: Summary metrics across baselines on synthetic planted data.
- `synthetic_planted_evaluation.csv`: Top-3 recall and Jaccard indices for each method on synthetic data.
- `real_2rjy_baseline_comparison.csv`: Multi-threshold ($k=10, 20, 50$) comparison between SAM-MFP and baselines on 2RJY.
- `variance_fallacy_rejected_features.csv`: Demonstration of the Variance Fallacy on 2RJY (top-variance features rejected by SAM-MFP due to low ESS and collinearity).
- `subtle_pareto_features.csv`: Subtle, low-variance collective features on the 2RJY Pareto front (high ESS, high entropy sidechain dihedrals and DSSP fractions).
- `ablation_study_results.csv`: Pareto front size and Jaccard similarity across objective ablations on 2RJY.
- `real_2rjy_pareto_candidates.csv`: Complete table of 340 Pareto front features for 2RJY.

## How to Run

Execute the interactive benchmark suite using Jupyter:
```bash
/Users/nirav/ds-env/bin/jupyter nbconvert --to notebook --execute benchmarks/sam_mfp/run_benchmarks.ipynb --inplace
```

## Key Findings

1. **Synthetic Ground Truth Recovery**: SAM-MFP achieves **1.00 (100%) planted recall without access to the supervised target**, while the naive Variance baseline achieves only 0.33 recall.
2. **Collinearity Handling**: Under collinear duplicates ($r > 0.99$), permutation importance splits importance and decision trees select arbitrarily; SAM-MFP's `Uniqueness` objective strictly dominates the inferior copy.
3. **The Variance Fallacy on 2RJY**: 12 of the top 20 variance features in 2RJY are rejected by SAM-MFP because they exhibit an ESS ratio of only 0.006 (~12 effective frames) and extreme collinearity.
4. **Discovery of Subtle Collective Dynamics**: SAM-MFP surfaces 10 high-quality sidechain dihedrals and secondary structure features with low overall variance ($s^2 < 0.004$) but near-perfect sampling independence ($\text{ESS Ratio} > 0.95$) and high occupancy entropy ($> 0.94$).
5. **Ablation Collapse**: Restricting prioritization to PCA salience alone collapses the Pareto front from 340 diverse trade-offs down to a single scalar feature.
