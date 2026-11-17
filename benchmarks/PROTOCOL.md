# MolScope Shared Evaluation & Benchmark Protocol

**Version:** 1.0.0  
**Effective Date:** 2026-11-17  
**Applies to:** Agent A (mdxplain comparison), Agent B (SAM-MFP validation), Agent D (ML rigor)

---

## 1. Objective and Principles

This protocol standardizes benchmarking datasets, preprocessing, feature extraction parameters, cross-validation splits, evaluation metrics, and reporting across all validation experiments.
All reported metrics must originate from reproducible commands executed in the target environment, with raw CSV/JSON outputs preserved under `benchmarks/` or results run directories.

---

## 2. Standard Evaluation Datasets

| Dataset Identifier | System Description | Frames | Atoms | Trajectories / Replicas | Ground Truth / Target Reference |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`2rjy_benchmark`** | Villin Headpiece (HP35, PDB: 2RJY) | 2,000 | 596 | 1 continuous trajectory (20 ns) | Discovered folding/unfolding intermediate macrostates |
| **`synthetic_planted`** | Synthetic feature matrix with known planted signal | 1,000 | N/A | 4 independent groups (250 frames each) | Exact planted ground-truth features (signal vs noise) |
| **`gpr6_ensemble`** | Human G-protein coupled receptor 6 (GPR6) | 500 | ~4,500 | 6 independent trajectories (3 active, 3 inactive) | Biological activation state (active vs inactive) |

---

## 3. Preprocessing and Feature Alignment

1. **Feature Scope:**
   - C-alpha pairwise distances ($d \le 12.0$ Å cutoff).
   - Backbone dihedral angles ($\phi, \psi, \omega$) with circular trigonometric encoding ($\sin, \cos$).
   - Sidechain rotamer dihedrals ($\chi_1\text{--}\chi_4$).
   - Hydrogen bonds ($d_{\text{donor-acceptor}} \le 3.5$ Å, $\theta \ge 120^\circ$).
   - Secondary structure composition via DSSP.
2. **Missing Feature Handling:**
   - Inapplicable features (e.g., Glycine $\chi_1$, terminal dihedrals) are marked as unavailable with explicit metadata, **never silently zero-filled**.

---

## 4. Cross-Validation and Autocorrelation Guardrails

1. **Independent Replicas Available ($\ge 2$ trajectories per class):**
   - Stratified Trajectory-Grouped Cross-Validation (`GroupKFold` or trajectory-stratified K-fold).
   - Entire trajectories are assigned strictly to train or test. **Zero frame leakage across folds**.
2. **Single Continuous Trajectory (Conformational State Classification):**
   - **Purged Temporal Block Cross-Validation:** The timeline is segmented into contiguous chunks of 25–50 frames.
   - An embargo buffer of 2 frames is excised at train/test boundaries, eliminating adjacent-frame autocorrelation leakage.

---

## 5. Standard Evaluation Metrics

1. **Clustering & State Discovery Agreement:**
   - Adjusted Rand Index (ARI): $ARI = \frac{\text{Index} - \text{Expected Index}}{\text{Max Index} - \text{Expected Index}}$
   - Normalized Mutual Information (NMI): $NMI(U, V) = \frac{2 I(U, V)}{H(U) + H(V)}$
   - Silhouette Score & Davies-Bouldin Index.
2. **Feature Prioritization & Recovery:**
   - Top-$k$ Feature Overlap ($k \in \{5, 10, 20\}$): $Jaccard(A, B) = \frac{|A \cap B|}{|A \cup B|}$
   - Precision and Recall against planted synthetic ground truth.
3. **Generalization Performance:**
   - Out-of-sample Trajectory-Mean Balanced Accuracy.
   - Out-of-sample ROC-AUC for binary comparisons.
   - Leave-One-Feature-Out (LOFO) Accuracy Drop: $\Delta Acc_j = Acc_{\text{full}} - Acc_{-j}$.

---

## 6. Execution Environment & Seeds

- **Python Interpreter:** `/Users/nirav/ds-env/bin/python` (Python 3.13 / 3.10+ compatible).
- **Default Random Seeds:** 42 (primary), 123 (seed stability test), 999 (control).
- **Compute Constraints:** Single-node workstation execution; recorded wall-clock time and peak RSS memory.
