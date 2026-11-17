# System-Aware Multi-Criteria Molecular Feature Prioritization (SAM-MFP): Mathematical Specification

## 1. Motivation and Problem Formulation

Molecular dynamics (MD) trajectories generate high-dimensional conformational data. Standard featurization schemes yield hundreds to thousands of candidate features (e.g., $C_\alpha$ pair distances, backbone/sidechain torsions, solvent-accessible surface areas, hydrogen bonds, and contact networks).

Traditional feature selection in MD analysis suffers from three foundational deficiencies:
1. **Variance Fallacy**: Filtering features solely by variance ($s^2$) rewards noisy, high-amplitude unconstrained loop motions over subtle, biophysically decisive collective shifts (e.g., hinge bends, catalytic dyad distances).
2. **Supervised Target Leakage**: Supervised feature importance (Random Forest MDI, Gini impurity, logistic regression) requires discrete state labels. Imposing labels prior to dimensionality reduction bakes arbitrary human categorization into discovery, causing selection bias.
3. **Statistical Inadequacy Ignorance**: Features with high apparent discrimination often exhibit strong autocorrelation, low Effective Sample Size (ESS), or poor replica reproducibility across independent trajectories.

**SAM-MFP (System-Aware Multi-Criteria Feature Prioritization)** resolves these challenges by framing feature discovery as a **target-free multi-objective Pareto optimization** problem over intrinsic dynamical salience, statistical sampling quality, informational uniqueness, and computational cost.

---

## 2. Method Formalism

### 2.1 Inputs
- **Feature Matrix** $X \in \mathbb{R}^{N \times D}$, where $N$ is the total number of frames and $D$ is the number of candidate molecular features.
- **Trajectory Metadata Manifest** $M$, specifying trajectory ID $t_i \in \{1, \dots, T\}$, replica ID $r_i$, and frame index/time for each frame $i \in \{1, \dots, N\}$.
- **Dimensionality Reduction Model** (PCA or TICA) fitted on $X$ yielding top-$k$ coordinate projections.

### 2.2 Evaluation Criteria (Objectives)

Each candidate feature $f_d \ (d = 1, \dots, D)$ is mapped to an objective vector $\mathbf{o}(f_d) \in \mathbb{R}^M$:

#### 1. Representation Salience ($S_{\text{salience}}$) — To Maximize
Quantifies the feature's participation in the primary low-dimensional conformational coordinate space (PCA or TICA). For PCA with $k$ retained components and explained variance ratios $\lambda_j$:
$$S_{\text{PCA}}(f_d) = \sum_{j=1}^k \lambda_j |v_{j, d}|$$
where $v_{j, d}$ is the loading of feature $d$ on eigenvector $j$. For TICA, $v_{j, d}$ represents the kinetic eigenvector projection weighted by kinetic timescales or eigenvalues.

#### 2. Effective Sample Size Ratio ($\text{ESS Ratio}$) — To Maximize
Measures sampling independence and penalizes high autocorrelation:
$$\text{ESS Ratio}(f_d) = \frac{\text{ESS}(f_d)}{N} = \frac{1}{1 + 2 \sum_{\tau=1}^{\tau_{\max}} \hat{\rho}_d(\tau)}$$
where $\hat{\rho}_d(\tau)$ is the empirical autocorrelation at lag $\tau$. A low ESS ratio indicates that high variance is driven by slow, un-equilibrated drift rather than ergodic equilibrium sampling.

#### 3. Replica Consistency ($R_{\text{rep}}$) — To Maximize
Requires at least $T \ge 2$ independent simulation trajectories. Evaluates the cross-trajectory reproducibility of the feature's empirical distribution:
$$R_{\text{rep}}(f_d) = 1.0 - \frac{1}{\binom{T}{2}} \sum_{1 \le a < b \le T} \text{JSD}\left(P_d^{(a)} \,||\, P_d^{(b)}\right)$$
where $\text{JSD}$ is the Jensen-Shannon divergence between the discretized empirical probability densities $P_d^{(a)}$ and $P_d^{(b)}$ of feature $d$ in trajectories $a$ and $b$.

#### 4. Occupancy Entropy ($H_{\text{occ}}$) — To Maximize
Penalizes pathological binary or impulse features (e.g., an interaction formed for only 1 frame) and rewards rich conformational exploration:
$$H_{\text{occ}}(f_d) = -\sum_{b=1}^B p_{d, b} \log_2(p_{d, b}) \Big/ \log_2(B)$$
normalized to $[0, 1]$ over $B$ histogram bins.

#### 5. Uniqueness ($U$) — To Maximize
Prevents the Pareto front from collapsing into redundant clusters of near-identical collinear features (e.g., adjacent atom pairs):
$$U(f_d) = 1.0 - \max_{d' \neq d, \, d' \in \mathcal{C}} |\text{Corr}(X_{*, d}, X_{*, d'})|$$
where $\mathcal{C}$ is the active candidate feature set. When two features have correlation $|\rho| > 0.95$, the one with lower sampling quality or higher cost is dominated.

#### 6. Computational Extraction Cost ($C_{\text{cost}}$) — To Minimize
Measured wall-clock extraction time per frame (seconds/frame) for the feature family. This allows identifying lightweight proxies that capture comparable physical motions at $10\times$ lower computational cost.

#### 7. Optional: State Separation ($\eta^2$) — To Maximize
When conformational states $\mathcal{S}$ are discovered (e.g., via Density Peak Advanced clustering), state separation is measured via ANOVA effect size:
$$\eta^2(f_d) = \frac{\text{SS}_{\text{between}}}{\text{SS}_{\text{total}}} = \frac{\sum_{s \in \mathcal{S}} N_s (\bar{x}_{s, d} - \bar{x}_d)^2}{\sum_{i=1}^N (x_{i, d} - \bar{x}_d)^2}$$

---

## 3. Pareto Dominance and Candidate Filtering

### 3.1 Strict Dominance Relation
Let $\mathcal{O}_{\max}$ be the set of objectives to maximize and $\mathcal{O}_{\min}$ the set of objectives to minimize.
Feature $A$ **dominates** feature $B$ (denoted $A \succ B$) if and only if:
1. $\forall m \in \mathcal{O}_{\max}: o_m(A) \ge o_m(B)$ and $\forall m \in \mathcal{O}_{\min}: o_m(A) \le o_m(B)$
2. $\exists m \in \mathcal{O}_{\max} \cup \mathcal{O}_{\min}$ such that the inequality is strict ($o_m(A) > o_m(B)$ or $o_m(A) < o_m(B)$).

### 3.2 Dominator Count & Pareto Ranking
For each candidate feature $f_d$:
$$\text{Dominator Count}(f_d) = |\{f_{d'} \in \mathcal{C} \mid f_{d'} \succ f_d\}|$$
The **Pareto Front** is the non-dominated set:
$$\mathcal{P} = \{f_d \in \mathcal{C} \mid \text{Dominator Count}(f_d) = 0\}$$

Features with low dominator counts ($1 \le \text{Dominator Count} \le 3$) represent near-optimal trade-offs and are retained as secondary candidates.

---

## 4. Benchmark Comparisons Against Supervised Baselines

To validate SAM-MFP against standard machine-learning baselines, MolScope implements four comparative baseline estimators in `molscope.analysis.prioritization.baselines`:

| Baseline Method | Mathematical Formulation | Characteristics |
|---|---|---|
| **Random Forest MDI** | Mean Decrease in Impurity across ensemble trees | Sensitive to feature scales, favors high-cardinality features |
| **Decision Tree Importance** | Normalized Gini/MSE reduction from root to leaves | Greedy split selection; under collinearity, picks 1 feature and ignores duplicates |
| **Mutual Information** | $I(X; Y) = \iint p(x, y) \log \frac{p(x, y)}{p(x)p(y)} dx dy$ | Non-parametric, captures non-linear relationships, scale-invariant |
| **Permutation Importance** | $P_d = \text{Score}(X) - \text{Score}(X^{\text{perm } d})$ | Evaluates test score drop; fails under extreme collinearity (correlated features substitute) |

### Comparative Metrics:
1. **Top-$k$ Recall**: Fraction of baseline top-$k$ features present in the SAM-MFP Pareto front:
   $$\text{Recall}_k = \frac{|\mathcal{P} \cap \text{Top}_k|}{k}$$
2. **Jaccard Index**:
   $$J(\mathcal{P}, \text{Top}_k) = \frac{|\mathcal{P} \cap \text{Top}_k|}{|\mathcal{P} \cup \text{Top}_k|}$$
3. **Rank Correlation**: Spearman $\rho$ between continuous baseline scores and SAM-MFP representation salience.

---

## 5. Failure Modes and Scope Boundaries

SAM-MFP is designed with strict mathematical guards. Users and automated pipelines must not trust outputs under the following conditions:

1. **Single Trajectory Without LOTO Stability**: When only 1 trajectory is supplied, Leave-One-Trajectory-Out (LOTO) cross-validation cannot be computed. Replica consistency defaults to constant or is unavailable. Pareto selection must not be claimed as cross-validated.
2. **Short Trajectories Below Autocorrelation Time**: If trajectory length $N < 10 \times \tau_{\text{corr}}$, the estimated Effective Sample Size is unreliably biased upward.
3. **Unchecked Collinearity**: If the `Uniqueness` objective is excluded (in ablation), groups of mutually collinear distance pairs ($r > 0.99$) flood the Pareto front, crowding out orthogonal physical degrees of freedom.
4. **Target Mismatch in Supervised Comparisons**: If comparing SAM-MFP against a supervised baseline whose target variable is uncoupled to global conformational dynamics (e.g., an arbitrary solvent fluctuation), SAM-MFP will intentionally diverge from the supervised ranking. This divergence is a feature, not a bug: SAM-MFP reflects intrinsic physical dynamics rather than spurious target correlations.
