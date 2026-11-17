# Statistical Machine Learning Rigor and Validation for Molecular Dynamics

## 1. Executive Summary and Foundational Principles

Supervised machine learning and feature attribution methods are increasingly deployed on molecular dynamics (MD) simulations to identify conformational switch mechanisms, ligand-induced allosteric pathways, and functional state determinants. However, applying conventional machine learning validation schemes (e.g., standard randomly shuffled $k$-fold cross-validation, unweighted empirical risk minimization, frame-level bootstrapping) to physical trajectory data introduces severe methodological artifacts:

1. **Pseudo-Replication & Autocorrelation Leakage**: MD frames sampled at picosecond-to-nanosecond intervals are strongly autocorrelated. Shuffling frames across train and test folds leaks identical or near-identical microstates across the partition, generating artificially inflated test performance and spurious feature attributions.
2. **Trajectory Length Imbalance**: Unweighted loss functions allow long trajectories or high-frequency sampling runs to dominate model parameters, biasing feature discovery toward single-trajectory idiosyncratic behaviors rather than reproducible biophysical mechanisms.
3. **Overconfidence via Frame-Level Resampling**: Standard non-parametric bootstrap resampling over individual frames ignores the underlying physical correlation structure, yielding catastrophically narrow confidence intervals.
4. **Lack of Negative Controls**: High test accuracy in complex classifiers often reflects trajectory batch confounding rather than causal physical discrimination. Without permutation-based negative controls, models risk presenting noise fits as physical insights.

The **MolScope Machine Learning Validation Framework** establishes rigorous, physically grounded validation protocols tailored specifically to molecular trajectory ensembles. This document details the mathematical formulation, implementation architecture, and verification standards governing predictive workflows in MolScope.

---

## 2. Theoretical Formulation: Autocorrelation and Effective Sample Size

### 2.1 The Pseudo-Replication Fallacy

Let an MD ensemble comprise $T$ independent simulation trajectories $\mathcal{T} = \{T_1, T_2, \dots, T_M\}$, where each trajectory $T_m$ consists of $N_m$ sequential frames:
$$\mathbf{X}^{(m)} = \left[ \mathbf{x}_1^{(m)}, \mathbf{x}_2^{(m)}, \dots, \mathbf{x}_{N_m}^{(m)} \right]^\top \in \mathbb{R}^{N_m \times D}$$
where $D$ is the dimensionality of the conformational feature space. The total number of frames is $N = \sum_{m=1}^M N_m$.

In physical dynamics governed by Langevin or Newtonian equations of motion, consecutive coordinates $\mathbf{x}_t$ and $\mathbf{x}_{t+\Delta t}$ exhibit strong physical inertia. For any observable feature $f$, the empirical autocorrelation at lag $\tau$ within trajectory $m$ is:
$$\hat{\rho}(\tau) = \frac{\sum_{t=1}^{N_m - \tau} \left( f(\mathbf{x}_t^{(m)}) - \bar{f} \right) \left( f(\mathbf{x}_{t+\tau}^{(m)}) - \bar{f} \right)}{\sum_{t=1}^{N_m} \left( f(\mathbf{x}_t^{(m)}) - \bar{f} \right)^2}$$

The **integrated autocorrelation time** $\tau_{\text{int}}$ characterizes the relaxation timescale:
$$\tau_{\text{int}} = 1 + 2 \sum_{\tau=1}^{\tau_{\max}} \hat{\rho}(\tau)$$

Consequently, the **effective sample size** (ESS) of trajectory $m$ is not $N_m$, but:
$$N_{\text{eff}}^{(m)} = \frac{N_m}{\tau_{\text{int}}^{(m)}} \ll N_m$$

When standard $k$-fold cross-validation randomly assigns frames $i \in \{1, \dots, N\}$ to train and test splits, adjacent frames $t$ and $t+1$ are separated into different folds with probability $\frac{2(k-1)}{k^2}$. Because $\|\mathbf{x}_t - \mathbf{x}_{t+1}\| \approx 0$, the model evaluates on data that is physically identical to its training set. This invalidates out-of-sample error estimates.

---

## 3. Cross-Validation Architectures in MolScope

MolScope provides two distinct, non-overlapping cross-validation architectures depending on the biological level of the target label.

```
+-----------------------------------------------------------------------------+
|                          TARGET LEVEL SEPARATION                            |
+-----------------------------------------------------------------------------+
                                       |
                   Is target constant per trajectory?
                                       |
                 +---------------------+---------------------+
                 | YES                                       | NO
                 v                                           v
+---------------------------------+         +---------------------------------+
|   TRAJECTORY-GROUPED CV         |         | PURGED TEMPORAL BLOCK CV        |
| - Whole trajectory held out     |         | - Contiguous temporal blocks    |
| - Equal trajectory weights      |         | - Embargo buffer (+/- margin)   |
| - Trajectory-mean accuracy      |         | - Zero boundary frame leakage   |
| - Trajectory majority vote      |         | - Eliminates auto-correlation   |
+---------------------------------+         +---------------------------------+
```

### 3.1 Architecture A: Trajectory-Grouped Cross-Validation

Trajectory-grouped cross-validation is mandatory when target labels describe trajectory-level phenomena (e.g., wild-type vs. mutant, apo vs. holo, active vs. inactive condition, temperature/solvent perturbations).

#### 1. Group Partitioning
Trajectories are partitioned into $K$ stratified outer folds $\mathcal{S}_1, \dots, \mathcal{S}_K$ such that:
$$\bigcup_{k=1}^K \mathcal{S}_k = \{1, \dots, M\}, \quad \text{with } \mathcal{S}_j \cap \mathcal{S}_k = \emptyset \quad (\forall j \neq k)$$
All frames belonging to trajectory $m \in \mathcal{S}_k$ are placed strictly in the test partition of fold $k$. Under no circumstances may any frame from trajectory $m$ appear in the training partition of fold $k$.

#### 2. Inverse-Sample Weighting
To prevent long or finely saved trajectories from dominating the empirical risk, MolScope assigns each training frame $i$ from trajectory $m$ an inverse-frequency weight:
$$w_i = \frac{1}{N_m}, \quad \text{normalized such that } \sum_{i \in \text{Train}} w_i = |\text{Train}|$$
This ensures every independent physical simulation trajectory exerts equal mathematical influence during optimization:
$$\mathcal{L}_{\text{weighted}}(\theta) = \sum_{m \in \text{Train}} \frac{1}{N_m} \sum_{t=1}^{N_m} \ell(f_\theta(\mathbf{x}_t^{(m)}), y^{(m)})$$

#### 3. Scoring Endpoints
Model performance is evaluated using two group-aware metrics:
- **Trajectory-Mean Accuracy**:
  $$\text{Acc}_{\text{traj}} = \frac{1}{|\mathcal{S}_k|} \sum_{m \in \mathcal{S}_k} \left[ \frac{1}{N_m} \sum_{t=1}^{N_m} \mathbb{I}\left( \hat{y}_t^{(m)} = y^{(m)} \right) \right]$$
- **Trajectory Majority Vote**:
  $$\hat{Y}^{(m)} = \operatorname{mode}_{t \in \{1, \dots, N_m\}} \left( \hat{y}_t^{(m)} \right), \quad \text{Acc}_{\text{vote}} = \frac{1}{|\mathcal{S}_k|} \sum_{m \in \mathcal{S}_k} \mathbb{I}\left( \hat{Y}^{(m)} = y^{(m)} \right)$$

---

### 3.2 Architecture B: Purged Temporal Block Cross-Validation with Embargo Margin

When target labels vary within a single trajectory (e.g., conformational states discovered via density peaks, open $\leftrightarrow$ closed transitions, dihedral rotamer states), cross-validation across trajectories is not applicable. MolScope applies **Purged Temporal Block Cross-Validation with an Embargo Margin**.

#### 1. Contiguous Temporal Chunking
Each trajectory of $N_m$ frames is segmented into contiguous temporal blocks of length $L_{\text{chunk}}$:
$$\mathcal{B}_c = \left\{ \mathbf{x}_{(c-1) L + 1}, \dots, \mathbf{x}_{c L} \right\}$$
Chunks are allocated to folds using stratified splitting on chunk dominant labels, ensuring that temporal ordering within each block remains contiguous.

#### 2. Embargo Margin Formulation
Because coordinates near the chunk boundaries $\partial \mathcal{B}_c$ remain correlated across the split, an embargo buffer of width $\Delta t_{\text{embargo}}$ is purged from the training set. For every test frame index $t_{\text{test}} \in \mathcal{D}_{\text{test}}$, any frame $t$ in the same trajectory satisfying:
$$|t - t_{\text{test}}| \le \Delta t_{\text{embargo}}$$
is strictly excluded from $\mathcal{D}_{\text{train}}$.

```
Trajectory Timeline:
[ ... Train Frames ... ][  EMBARGO  ][   TEST CHUNK   ][  EMBARGO  ][ ... Train Frames ... ]
                        <-- delta -->[ tf_start, tf_end ]<-- delta -->
```

#### 3. Formal Zero-Leakage Guarantee
MolScope verifies that:
$$\mathcal{D}_{\text{train}} \cap \mathcal{D}_{\text{test}} = \emptyset$$
and for all $t_{\text{tr}} \in \mathcal{D}_{\text{train}}$ and $t_{\text{te}} \in \mathcal{D}_{\text{test}}$ from the same trajectory:
$$|t_{\text{tr}} - t_{\text{te}}| > \Delta t_{\text{embargo}}$$

---

## 4. Uncertainty Estimation and Stability Diagnostics

### 4.1 Group-Level (Trajectory) Bootstrap Resampling

Standard bootstrap resampling that samples individual rows with replacement creates severe pseudo-replication bias by duplicating correlated frames into the resample while omitting others.

In MolScope, the legitimate bootstrap sampling unit is the **independent simulation trajectory**:
1. Given $M$ trajectories $\{T_1, \dots, T_M\}$, draw $M$ trajectories with replacement:
   $$\mathcal{T}^{*(b)} = \left\{ T_{i_1}, T_{i_2}, \dots, T_{i_M} \right\}, \quad i_j \sim \text{Uniform}(\{1, \dots, M\})$$
2. Duplicate trajectories are assigned unique instance identifiers ($T_m\text{\_\_boot\_}j$) to prevent grouped CV fold collisions.
3. The full ML workflow is evaluated across $B$ bootstrap realizations.
4. Empirical confidence intervals are constructed from the bootstrap distribution quantiles:
   $$\text{CI}_{0.95} = \left[ q_{0.025}(\hat{\theta}^*), \, q_{0.975}(\hat{\theta}^*) \right]$$

### 4.2 Descriptive Fold Resampling vs. Independent Replica Confidence Intervals

In Leave-One-Feature-Out (LOFO) importance analysis:
- Cross-validation fold variation measures **instability across data splits**, not sampling uncertainty across physical replicas.
- MolScope explicitly designates fold resampling intervals as descriptive:
  ```python
  "Uncertainty_Qualification": "fold_resampling_is_descriptive_not_independent_replica_confidence_interval"
  ```
- Genuine physical confidence intervals require multiple independent simulation replicas ($M \ge 10$) resampled via `evaluate_group_bootstrap_uncertainty`.

### 4.3 Feature Ranking Stability Across Replicas and Random Seeds

Feature rankings derived from stochastic learners (e.g., Random Forests) or finite sample splits can exhibit high variance. MolScope quantifies ranking stability using three complementary metrics:

1. **Pairwise Spearman Rank Correlation**:
   $$\rho_s = 1 - \frac{6 \sum_{d=1}^D (R_d^{(1)} - R_d^{(2)})^2}{D(D^2 - 1)}$$
2. **Pairwise Kendall's Tau**:
   $$\tau = \frac{C - D_{\text{pairs}}}{\frac{1}{2} D (D - 1)}$$
   where $C$ and $D_{\text{pairs}}$ are concordant and discordant pairs.
3. **Top-$k$ Jaccard Overlap**:
   $$J_k = \frac{|\mathcal{F}_{\text{top-}k}^{(1)} \cap \mathcal{F}_{\text{top-}k}^{(2)}|}{|\mathcal{F}_{\text{top-}k}^{(1)} \cup \mathcal{F}_{\text{top-}k}^{(2)}|}$$

A ranking is classified as stable (`is_stable = True`) only when mean pairwise rank correlation exceeds the stability threshold ($\ge 0.50$).

---

## 5. Negative Control: Permutation Null Hypothesis Testing

### 5.1 Formulation of the Null Test

To guard against spurious predictive associations and subtle data leakage, MolScope provides automated permutation testing (`evaluate_permutation_null_control`).

Under the null hypothesis $H_0$:
$$\mathbb{P}(Y \,|\, \mathbf{X}) = \mathbb{P}(Y)$$
the association between molecular conformational features and target labels is completely absent.

### 5.2 Group-Level vs. Frame-Level Permutation

- **Trajectory-Level Target Shuffling**: When targets are constant across trajectories, labels are shuffled at the **trajectory level**:
  $$\pi: \{1, \dots, M\} \to \{1, \dots, M\}$$
  All frames belonging to trajectory $m$ receive the permuted label $Y_{\pi(m)}$. This perfectly preserves intra-trajectory temporal autocorrelation, frame count distributions, and feature distributions, isolating and destroying only the physical signal.
- **Frame-Level Target Shuffling**: Used for within-trajectory state targets.

### 5.3 Acceptance Criteria for Negative Controls

For a binary classification task under $H_0$:
- Balanced Accuracy must collapse to chance: $\mathbb{E}[\text{Bal\_Acc}] \approx 0.50$ (empirically bounded in $[0.30, 0.70]$ for finite sample ensembles).
- ROC AUC must collapse to chance: $\mathbb{E}[\text{ROC\_AUC}] \approx 0.50$.
- Real model accuracy must decisively exceed the permutation null distribution:
  $$\text{Acc}_{\text{real}} > \bar{\text{Acc}}_{\text{null}} + 2 \cdot \sigma_{\text{null}}$$

If a shuffled-label model achieves high balanced accuracy (e.g., $> 0.75$), the pipeline triggers a failure flag indicating either:
1. Information leakage between train and test splits, or
2. Severe batch-effect confounding correlated with trajectory identities.

---

## 6. Diagnostic API Reference

| Function / Class | Module | Scientific Role |
| :--- | :--- | :--- |
| `MLWorkflowConfig` | `molscope.analysis.predictive.workflow` | Configuration for splits, estimators, chunk size, and embargo margin |
| `run_grouped_ml_workflow` | `molscope.analysis.predictive.workflow` | Executes trajectory-grouped or purged block CV with LOFO ablation |
| `purged_temporal_block_splits` | `molscope.analysis.predictive.workflow` | Partitions contiguous temporal chunks with purged embargo buffers |
| `check_simulation_convergence` | `molscope.analysis.quality.statistical_rigor` | Evaluates integrated autocorrelation time $\tau_{\text{int}}$, ESS, and drift |
| `evaluate_seed_stability` | `molscope.analysis.quality.statistical_rigor` | Pairwise ARI across clustering initializations |
| `evaluate_bootstrap_cluster_stability` | `molscope.analysis.quality.statistical_rigor` | Bootstrap ARI and empirical confidence intervals |
| `evaluate_feature_ranking_stability` | `molscope.analysis.quality.statistical_rigor` | Spearman $\rho$, Kendall $\tau$, and top-$k$ Jaccard across seeds/runs |
| `evaluate_group_bootstrap_uncertainty` | `molscope.analysis.quality.statistical_rigor` | Trajectory-level bootstrap resampling for unbiased uncertainty |
| `evaluate_permutation_null_control` | `molscope.analysis.quality.statistical_rigor` | Shuffled-label negative control confirming collapse to chance |

---

## 7. Recommended Biophysical Reporting Checklist

When reporting ML-derived molecular feature importances, publications should report:

- [ ] **Trajectory Independence**: Number of independent physical replicas ($M \ge 3$ per condition; $M \ge 10$ recommended for robust ranking).
- [ ] **Cross-Validation Type**: Explicit statement of whether Trajectory-Grouped CV or Purged Temporal Block CV was employed.
- [ ] **Embargo Margin**: Explicit buffer width in frames and physical time (e.g., $\Delta t_{\text{embargo}} = 2 \text{ ns} \ge \tau_{\text{int}}$).
- [ ] **Sample Weighting**: Confirmation that inverse-trajectory-length sample weights were used during fitting.
- [ ] **Zero Frame Leakage Assertion**: Verification that train and test partitions share 0 common frames.
- [ ] **Negative Control Results**: Shuffled-label balanced accuracy and ROC AUC confirming chance-level performance under $H_0$.
- [ ] **Ranking Stability**: Spearman $\rho$ and Kendall $\tau$ across distinct seeds and replica subsets.
- [ ] **Causality Qualification**: Explicit notation that predictive importance reflects statistical association, not perturbation-tested physical causality.
