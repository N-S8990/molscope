# System-Aware Multi-Criteria Molecular Feature Prioritization (SAM-MFP)

## 1. Executive Summary and Motivation

High-throughput molecular dynamics (MD) simulations produce high-dimensional trajectory coordinates. Standard featurization protocols typically yield thousands of candidate variables, including inter-atomic distances, backbone and sidechain dihedral angles, contact states, solvent-accessible surface areas (SASA), and secondary structure metrics.

Biophysical feature selection in MD analysis has historically relied on three problematic paradigms:
1. **The Variance Fallacy**: Ranking features by empirical variance ($s^2 = \frac{1}{N-1}\sum (x_i - \bar{x})^2$) rewards noisy, high-amplitude unconstrained loop fluctuations while discarding subtle, biologically crucial collective transitions (e.g., catalytic dyad rearrangements, allosteric hinge bends).
2. **Supervised Target Leakage and Confirmation Bias**: Imposing discrete state or phenotype labels prior to feature prioritization (e.g., via Random Forest Gini importance or logistic regression) introduces arbitrary human categorization, overfits to specific classification boundaries, and masks un-hypothesized conformational intermediates.
3. **Statistical Inadequacy Ignorance**: Features exhibiting high nominal variance or apparent state separation frequently suffer from extreme autocorrelation, low Effective Sample Size ($\text{ESS} \ll N$), or poor reproducibility across independent simulation replicas.

**SAM-MFP (System-Aware Multi-Criteria Feature Prioritization)** reformulates feature discovery as a **target-free multi-objective Pareto optimization** problem. Instead of collapsing heterogeneous evidence into a single heuristic scalar score, SAM-MFP evaluates candidate features simultaneously across intrinsic coordinate salience, statistical sampling independence, replica reproducibility, conformational occupancy entropy, informational uniqueness, and computational cost.

---

## 2. Mathematical Formalism and Objectives

Let $X \in \mathbb{R}^{N \times D}$ be the trajectory feature matrix for $N$ frames and $D$ candidate features. Let $M$ be the trajectory metadata manifest recording trajectory index $t_i \in \{1, \dots, T\}$, replica identifier $r_i$, and frame order for each frame $i \in \{1, \dots, N\}$.

Each candidate feature $f_d \ (d = 1, \dots, D)$ is mapped to an evaluation objective vector $\mathbf{o}(f_d) = (o_1, \dots, o_M) \in \mathbb{R}^M$.

### 2.1 Primary Objectives

#### 1. Representation Salience ($S_{\text{salience}}$) — Maximized
Measures the feature's participation in the primary low-dimensional conformational manifold. For Principal Component Analysis (PCA) with $k$ retained components and explained variance ratios $\lambda_j$:
$$S_{\text{PCA}}(f_d) = \sum_{j=1}^k \lambda_j |v_{j, d}|$$
where $v_{j, d}$ is the eigenvector loading of feature $d$ on principal component $j$. For Time-lagged Independent Component Analysis (TICA) or Kinetic PCA:
$$S_{\text{TICA}}(f_d) = \sum_{j=1}^k \mu_j |w_{j, d}|$$
where $w_{j, d}$ represents the kinetic component loading and $\mu_j$ is weighted by the implied timescale or kinetic eigenvalue.

#### 2. Effective Sample Size Ratio ($\text{ESS Ratio}$) — Maximized
Measures sampling independence and penalizes high autocorrelation:
$$\text{ESS Ratio}(f_d) = \frac{\text{ESS}(f_d)}{N} = \frac{1}{1 + 2 \sum_{\tau=1}^{\tau_{\max}} \hat{\rho}_d(\tau)}$$
where $\hat{\rho}_d(\tau)$ is the sample autocorrelation at lag $\tau$, summed up to the first negative autocorrelation or an automated cut-off. An ESS ratio near $1.0$ indicates independently sampled equilibrium configurations, whereas an ESS ratio $\ll 0.05$ indicates slow, un-equilibrated drift.

#### 3. Replica Consistency ($R_{\text{rep}}$) — Maximized
Quantifies cross-trajectory reproducibility across $T \ge 2$ independent simulation trajectories. In MolScope, this is computed via the pairwise normalized Wasserstein distance:
$$R_{\text{rep}}(f_d) = 1.0 - \frac{1}{\binom{T}{2}} \sum_{1 \le a < b \le T} \frac{W_1\left(P_d^{(a)}, \, P_d^{(b)}\right)}{\text{IQR}(P_d^{\text{pooled}})}$$
where $W_1$ is the 1-Wasserstein (earth-mover) distance between empirical distributions of feature $d$ in trajectories $a$ and $b$, scaled by the interquartile range (IQR) of the pooled distribution. $R_{\text{rep}} \in (0, 1]$, where $1.0$ represents indistinguishable distributions across independent simulation runs.

#### 4. Occupancy Entropy ($H_{\text{occ}}$) — Maximized
Penalizes impulsive, near-binary, or unvisited features (e.g., transient contacts formed in only 1 frame) and rewards continuous, multi-modal conformational exploration:
$$H_{\text{occ}}(f_d) = -\frac{\sum_{b=1}^B p_{d, b} \log_2(p_{d, b})}{\log_2(B)}$$
where $p_{d, b}$ is the normalized occupancy frequency in histogram bin $b$ across $B$ equal-frequency or equal-width bins ($H_{\text{occ}} \in [0, 1]$).

#### 5. Informational Uniqueness ($U$) — Maximized
Prevents the candidate set from collapsing into redundant clusters of near-identical collinear variables (e.g., closely adjacent atom-pair distances $C_\alpha(i, j)$ and $C_\alpha(i, j+1)$):
$$U(f_d) = 1.0 - \max_{d' \neq d, \, d' \in \mathcal{C}} \left|\rho_{\text{Spearman}}(X_{*, d}, X_{*, d'})\right|$$
where $\mathcal{C}$ is the active feature candidate set. When two features exhibit correlation $|\rho| > 0.98$, the one with lower ESS, lower entropy, or higher cost is strictly dominated.

#### 6. Representation Stability ($S_{\text{stability}}$) — Maximized
When $T \ge 2$ independent trajectories are available, stability is assessed via Leave-One-Trajectory-Out (LOTO) cross-validation:
$$S_{\text{stability}}(f_d) = \left[1.0 - \frac{\text{Std}_{\text{LOTO}}(\text{Rank}_d)}{\text{Max Rank}}\right]_{[0, 1]}$$
Features whose coordinate salience rank is stable across fold omissions receive scores close to $1.0$.

### 2.2 Optional and Contextual Objectives

#### 7. Computational Extraction Efficiency ($C_{\text{efficiency}}$) — Maximized
When feature extraction wall-clock times are measured:
$$C_{\text{efficiency}}(f_d) = - C_{\text{cost}}(f_d)$$
where $C_{\text{cost}}$ is the measured computation time in seconds per frame. Maximizing negative cost allows identification of computationally lightweight proxies (e.g., $C_\alpha$ distances instead of full all-atom contact networks).

#### 8. State Explained Variance ($\eta^2$) — Maximized
When discrete conformational states $\mathcal{S}$ are discovered through unsupervised clustering (e.g., Density Peak Advanced clustering on TICA/PCA coordinates):
$$\eta^2(f_d) = \frac{\text{SS}_{\text{between}}}{\text{SS}_{\text{total}}} = \frac{\sum_{s \in \mathcal{S}} N_s (\bar{x}_{s, d} - \bar{x}_d)^2}{\sum_{i=1}^N (x_{i, d} - \bar{x}_d)^2}$$

---

## 3. Pareto Dominance and Ranking Mechanics

### 3.1 Strict Dominance Relation
All objectives in SAM-MFP are formulated such that higher values are strictly preferable:
$$\mathbf{o}(f) = \left(o_1(f), \, o_2(f), \, \dots, \, o_M(f)\right)$$

Feature $A$ strictly **dominates** feature $B$ (denoted $A \succ B$) if and only if:
1. $\forall m \in \{1, \dots, M\}: o_m(A) \ge o_m(B)$
2. $\exists m \in \{1, \dots, M\}: o_m(A) > o_m(B)$

If feature $A$ is at least as good as feature $B$ in all dimensions and strictly superior in at least one, feature $B$ is dominated.

### 3.2 Dominator Count and Pareto Front
For every eligible feature $f_d \in \mathcal{C}$:
$$\text{Dominator Count}(f_d) = \sum_{f_{d'} \in \mathcal{C}, \, d' \neq d} \mathbb{I}\left(f_{d'} \succ f_d\right)$$

The **Pareto Front** $\mathcal{P}$ is defined as the subset of completely non-dominated candidates:
$$\mathcal{P} = \left\{f_d \in \mathcal{C} \;\big|\; \text{Dominator Count}(f_d) = 0\right\}$$

- **Tier 1 (Pareto Front)**: $\text{Dominator Count} = 0$. These features represent optimal, non-dominated trade-offs across the biophysical and statistical criteria.
- **Tier 2 (Near-Optimal Candidates)**: $1 \le \text{Dominator Count} \le 3$. Secondary features that narrowly miss the front due to a minor deficiency in one objective.
- **Dominated Features**: $\text{Dominator Count} > 3$. Strictly inferior candidates with superior alternatives across all criteria.

### 3.3 Why a Pareto Set Instead of a Weighted Scalar Sum?
A linear scalarization $S = \sum w_m o_m$ requires choosing arbitrary heuristic weights (e.g., $w_{\text{salience}} = 0.5$, $w_{\text{ESS}} = 0.3$). Scalarization creates three severe flaws:
1. **Weight Arbitrariness**: No physical principle dictates the relative trade-off between 10% more ESS vs. 10% higher PCA salience.
2. **Convex Hull Blindness**: Linear weighting cannot discover non-convex regions of the trade-off frontier.
3. **Masking of Specialized Profiles**: A feature with exceptional kinetic salience but moderate cost should be presented to the biophysicist alongside an ultra-fast, high-ESS proxy feature, allowing informed scientific selection based on downstream goals.

---

## 4. Configuration and Eligibility Parameters

SAM-MFP is configured through the immutable dataclass `SAMMFPConfig` in `molscope.analysis.prioritization.sam_mfp`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `min_loto_folds` | `int` | `2` | Minimum number of trajectory folds required for cross-replica stability. Must be $\ge 2$. |
| `salience_metric` | `str` | `"PCA_Salience"` | Representation column to maximize (`"PCA_Salience"`, `"TICA_Salience"`, `"Kinetic_Salience"`). |
| `include_state_separation` | `bool` | `False` | Whether to include $\eta^2$ state separation when state clustering is provided. |
| `excluded_objectives` | `tuple[str, ...]` | `()` | Explicit tuple of objective names to exclude (e.g., for ablation experiments). |
| `custom_objectives` | `tuple[str, ...] \| None` | `None` | Explicit tuple of objectives to optimize, overriding default objective selection. |

### Eligibility Filtering Rules:
A feature is marked `SAM_MFP_Eligible = True` only if:
1. `Resolution == "frame"`: Aggregate trajectory-level summary statistics (e.g., whole-trajectory DSSP percentages) are ineligible for frame-level Pareto ranking.
2. `Valid_Fraction >= 0.90`: Features with missing or non-finite values are excluded.
3. Complete finite measurements exist for all active objective columns.

---

## 5. Method Assumptions and Failure Modes

### 5.1 Assumptions
1. **Low-Dimensional Conformational Manifold**: Assumes relevant collective dynamics are captured in the top-$k$ eigenvectors of PCA, TICA, or Contact KPCA.
2. **Ergodic Sampling within Trajectories**: Effective Sample Size estimation assumes trajectories are stationary or quasi-stationary after removal of initial equilibration transients.
3. **Pairwise Monotonicity in Redundancy**: Uniqueness objective uses Spearman rank correlation, which penalizes monotonic collinearity but not high-order non-linear interactions.

### 5.2 Failure Modes and Diagnostic Safeguards

#### 1. Single Trajectory Without Cross-Replica Consistency (Resolved Fallback)
*Problem*: If only 1 trajectory is provided ($T = 1$), cross-trajectory replica consistency ($R_{\text{rep}}$) and LOTO stability cannot be calculated ($R_{\text{rep}} = \text{NaN}$).
*Safeguard*: When `Replica_Consistency` is all-NaN across the dataset, SAM-MFP automatically drops the objective and emits an explicit diagnostic note:
`"Single trajectory available; evaluated Pareto candidates without cross-replica consistency."`
Users must not cite cross-replica validation when only a single simulation run was supplied.

#### 2. Short Trajectories Below Autocorrelation Time
*Problem*: If trajectory duration $N < 10 \times \tau_{\text{corr}}$, the empirical autocorrelation $\hat{\rho}(\tau)$ cannot decay to zero, resulting in biased, artificially elevated ESS ratios.
*Safeguard*: MolScope caps $\tau_{\max}$ and warns when median block counts are $< 5$.

#### 3. Extreme Redundancy Under Collinearity Ablation
*Problem*: If `Uniqueness` is ablated, hundreds of nearly identical distance pairs ($r > 0.99$) from adjacent atom pairs flood the Pareto front, crowding out orthogonal physical degrees of freedom.
*Safeguard*: `Uniqueness` is a mandatory core quality objective in the default configuration.

#### 4. Disagreement with Supervised Baselines
*Problem*: When comparing SAM-MFP against a supervised target (e.g., an arbitrary solvent fluctuation or synthetic label), SAM-MFP Pareto candidates may show low overlap with baseline top-$k$.
*Interpretation*: This divergence is scientifically expected: SAM-MFP isolates **intrinsic dynamical coordinates** of the molecular system, whereas supervised models prioritize features that happen to separate the supplied arbitrary target, regardless of physical or statistical quality.

---

## 6. Empirical Validation and Baseline Benchmarks

The benchmark suite in `benchmarks/sam_mfp/run_benchmarks.py` validates SAM-MFP across both synthetic planted-signal datasets and the 10,001-frame Villin Headpiece (2RJY) benchmark.

### 6.1 Synthetic Planted-Signal Benchmark
To test ground-truth recovery, a 20-feature dataset was generated with 3 planted signal features having complementary multi-objective profiles and 17 strictly inferior noise features ($N=600$ frames, $T=3$ replicas):

| Method | Supervised? | Top-3 Planted Recall | Jaccard Index | Dominator Count (Planted) |
|---|---|---|---|---|
| **SAM-MFP (Pareto Front)** | **No (Target-Free)** | **1.00 (3/3)** | **1.00** | **0 (All Non-Dominated)** |
| **Random Forest (MDI)** | Yes (uses $y$) | 1.00 (3/3) | 1.00 | N/A |
| **Mutual Information** | Yes (uses $y$) | 1.00 (3/3) | 1.00 | N/A |
| **Decision Tree (Gini)** | Yes (uses $y$) | 1.00 (3/3) | 1.00 | N/A |
| **Permutation Importance** | Yes (uses $y$) | 1.00 (3/3) | 1.00 | N/A |
| **Empirical Variance** | No | 0.33 (1/3) | 0.20 | N/A |
| **Random Ranking** | No | 0.33 (1/3) | 0.20 | N/A |

**Key Finding**: SAM-MFP recovered **100% of planted ground-truth features without access to the supervised target**, while purely variance-based selection failed (33% recall), mistaking high-amplitude noise for signal.

#### Collinearity and Permutation Importance Failure:
When two identical signal features ($r = 0.999$) were introduced, permutation importance split the importance evenly between both copies, diluting their rank. In contrast, SAM-MFP's `Uniqueness` objective detected the redundancy and strictly dominated the copy with marginally higher extraction cost, automatically pruning collinear duplicates from the Pareto front.

---

### 6.2 Real-World 2RJY Villin Headpiece Benchmark
The HP-35 Villin Headpiece trajectory (2,000 analyzed frames, 3,818 total extracted features, 2,091 eligible frame features, 5 DPA conformational clusters) was benchmarked against four baselines:
1. **Empirical Variance**
2. **Mutual Information with DPA State Labels**
3. **Decision Tree Gini Importance**
4. **Random Ranking (Null Hypothesis)**

#### Multi-Threshold Comparison Summary:

| Baseline Comparator | Top-$k$ | Overlap with Pareto Front | Top-$k$ Recall | Pareto Precision | Jaccard Index | Spearman $\rho$ vs. Salience |
|---|---|---|---|---|---|---|
| **Empirical Variance** | 10 | 4 | 0.40 | 0.012 | 0.012 | $\rho = 0.776$ ($p < 10^{-15}$) |
| **Empirical Variance** | 20 | 8 | 0.40 | 0.024 | 0.023 | $\rho = 0.776$ ($p < 10^{-15}$) |
| **Empirical Variance** | 50 | 15 | 0.30 | 0.044 | 0.040 | $\rho = 0.776$ ($p < 10^{-15}$) |
| **Mutual Information (DPA)** | 10 | 1 | 0.10 | 0.003 | 0.003 | $\rho = 0.854$ ($p < 10^{-15}$) |
| **Mutual Information (DPA)** | 20 | 1 | 0.05 | 0.003 | 0.003 | $\rho = 0.854$ ($p < 10^{-15}$) |
| **Mutual Information (DPA)** | 50 | 4 | 0.08 | 0.012 | 0.010 | $\rho = 0.854$ ($p < 10^{-15}$) |
| **Decision Tree Gini** | 10 | 0 | 0.00 | 0.000 | 0.000 | $\rho = 0.030$ ($p = 0.17$) |
| **Decision Tree Gini** | 20 | 3 | 0.15 | 0.009 | 0.008 | $\rho = 0.030$ ($p = 0.17$) |
| **Decision Tree Gini** | 50 | 8 | 0.16 | 0.024 | 0.021 | $\rho = 0.030$ ($p = 0.17$) |
| **Random Ranking** | 20 | 1 | 0.05 | 0.003 | 0.003 | $\rho = 0.006$ ($p = 0.77$) |

#### The Variance Fallacy in 2RJY:
Of the top 20 features ranked by empirical variance, **12 were rejected by SAM-MFP** (dominated by other features). Detailed inspection reveals:
- Feature `CA_DISTANCE_ALA_26__ARG_37`: Rank 1 in variance, but has an $\text{ESS Ratio}$ of **0.006** (only ~12 independent effective frames out of 2,000!) and Uniqueness of **0.020** (extreme collinear redundancy). SAM-MFP assigned this feature a **Dominator Count of 7.0**.
- The top-variance features represent high-amplitude, non-ergodic terminal loop drifts rather than concerted structural transitions.

#### Subtle Collective Features Discovered by SAM-MFP:
SAM-MFP identified 10 features on the Pareto front that had **near-zero variance** ($s^2 < 0.004$) and were placed at the very bottom of variance ranking, but exhibited exceptional statistical quality:
- `SIDECHAIN_chi1__PRO_17__cos`: $\text{ESS Ratio} = 0.949$, $\text{Entropy} = 0.937$, $\text{Uniqueness} = 0.225$.
- `SIDECHAIN_chi1__PRO_30__cos`: $\text{ESS Ratio} = 1.000$, $\text{Entropy} = 0.947$, $\text{Uniqueness} = 0.181$.
- `DSSP_sheet_fraction`: $\text{ESS Ratio} = 0.146$, $\text{Uniqueness} = 0.365$.
These represent subtle, highly independent sidechain rotameric flips and secondary structure transitions that traditional variance filtering completely discards.

---

### 6.3 Systematic Objective Ablation Study on 2RJY

To assess the marginal contribution of each objective, ablation experiments were conducted on the 2RJY dataset ($D=2,091$ eligible features):

| Configuration | Objectives Retained | Pareto Front Size | Overlap with Full Pareto | Jaccard Index vs. Full |
|---|---|---|---|---|
| **Full Default Objectives** | $\text{ESS}, H_{\text{occ}}, U, S_{\text{PCA}}, S_{\text{rep}}$ | **340** | **340** | **1.000** |
| **Ablate Uniqueness** | $\text{ESS}, H_{\text{occ}}, S_{\text{PCA}}, S_{\text{rep}}$ | 159 | 159 | 0.468 |
| **Ablate ESS Ratio** | $H_{\text{occ}}, U, S_{\text{PCA}}, S_{\text{rep}}$ | 92 | 92 | 0.271 |
| **Ablate Occupancy Entropy** | $\text{ESS}, U, S_{\text{PCA}}, S_{\text{rep}}$ | 120 | 112 | 0.322 |
| **Ablate PCA Salience** | $\text{ESS}, H_{\text{occ}}, U, S_{\text{rep}}$ | 4 | 4 | 0.012 |
| **Salience Only (Scalar)** | $S_{\text{PCA}}$ | **1** | **1** | **0.003** |

#### Insights from Ablation:
1. **Collapse Under Single-Objective Optimization**: Restricting prioritization to PCA salience alone collapses the Pareto front from 340 rich trade-offs down to a **single feature**. A single scalar rank completely erases alternative hypotheses and orthogonal physical dimensions.
2. **Criticality of Sampling Quality**: Ablating `ESS_Ratio` reduces the Pareto front by 73% (from 340 to 92), as slow-drifting autocorrelated features drown out features whose primary virtue is independent equilibrium sampling.
3. **Information Pruning by Uniqueness**: Ablating `Uniqueness` reduces front diversity (Jaccard = 0.468), allowing redundant collinear clusters to suppress independent degrees of freedom.

---

## 7. Implementation Flags, Code Audit, and Critical Recommendations

During thorough code inspection of `src/molscope/analysis/prioritization/sam_mfp.py`, four notable architectural points and potential traps were identified:

### Flag 1: Single-Trajectory Ineligibility Trap (Resolved)
- **Issue**: In the original implementation, `_QUALITY_OBJECTIVES` included `"Replica_Consistency"`. When single-trajectory data was supplied (such as 2RJY `simulation_run1.xtc`), `Replica_Consistency` was set to `NaN` for all features. The eligibility check `table[objective_columns].notna().all(axis=1)` consequently evaluated to `False` for 100% of candidate features, silently producing an empty Pareto front (`Pareto_Front_Size = 0`).
- **Remediation Applied**: `prioritize_sam_mfp` now inspects whether `Replica_Consistency` is all-NaN across the dataset. If so, it gracefully excludes `Replica_Consistency` from the active Pareto objectives, records an explicit diagnostic note (`"Single trajectory available; evaluated Pareto candidates without cross-replica consistency."`), and evaluates the Pareto front across the remaining valid dimensions.

### Flag 2: Quadratic Scaling of Uniqueness Computation
- **Issue**: Uniqueness requires computing the maximum absolute Spearman rank correlation across all feature pairs $\binom{D}{2}$. For $D > 5,000$ features, this computation takes $O(D^2 N)$ operations.
- **Recommendation**: MolScope already supports `max_correlation_frames` subsampling. For ultra-large libraries ($D > 20,000$), hierarchical clustering or approximate nearest-neighbor correlation should be adopted.

### Flag 3: PCA vs. TICA Salience in Kinetic Studies
- **Issue**: PCA maximizes static geometric variance, which can sometimes be dominated by large flexible loops. TICA maximizes autocorrelation at a user-defined lag time $\tau$.
- **Recommendation**: When analyzing kinetic processes (folding, binding kinetics, state transitions), users should pass `salience_metric="TICA_Salience"` or `"Kinetic_Salience"` to `SAMMFPConfig`.

### Flag 4: Cost Objective Asymmetry
- **Issue**: When feature extraction time is measured at the plugin level (e.g., all 2,000 $C_\alpha$ distances extracted in a single vectorized pass of 0.05s), features within the same family share identical cost values.
- **Behavior**: Because costs are tied, the cost objective does not differentiate between individual features of the same plugin; it acts as a group-level trade-off between whole feature classes (e.g., distances vs. PLIF fingerprints).

---

## 8. Summary Comparison: SAM-MFP vs. Traditional Baselines

| Criterion | Empirical Variance | Random Forest MDI | Decision Tree Split | Mutual Information | SAM-MFP |
|---|---|---|---|---|---|
| **Supervision Required** | Target-Free | Requires Class/Target | Requires Class/Target | Requires Class/Target | **Target-Free** |
| **Sensitivity to Noise Drift** | **Severe (Fails)** | Moderate | Moderate | Moderate | **Resistant (via ESS Ratio)** |
| **Collinearity Pruning** | None | Feature Dilution | Arbitrary Greedy Pick | None | **Strict Dominance (via Uniqueness)** |
| **Replica Consistency** | None | None | None | None | **Evaluated (via Wasserstein JSD)** |
| **Conformational Exploration** | None | None | None | None | **Evaluated (via Occupancy Entropy)** |
| **Output Type** | Single Scalar Ranking | Single Scalar Ranking | Single Scalar Ranking | Single Scalar Ranking | **Multi-Objective Pareto Set** |
