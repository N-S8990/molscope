# MolScope: System-Aware Molecular Dynamics Feature Discovery & Core Machine Learning

[![CI Matrix](https://img.shields.io/badge/CI-Passing%20(360%20tests)-10b981.svg)](https://github.com/N-S8990/molscope)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-3776ab.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Version: 0.2.0](https://img.shields.io/badge/version-0.2.0-orange.svg)](https://github.com/N-S8990/molscope)

MolScope is an end-to-end Python platform for Molecular Dynamics (MD) simulation, multi-family biophysical featurization, non-parametric conformational state discovery, and statistically rigorous Machine Learning feature prioritization (**SAM-MFP**).

---

## Graphical Abstract

![MolScope Graphical Abstract](docs/images/molscope_graphical_abstract.png)

*Figure 1: The MolScope 3-Stage Architecture. (Top) Flexible input: raw PDB structures (automated OpenMM simulation) or existing multi-replica MD trajectories with zero-leakage cross-validation splits. (Middle) Integrated engine: multi-family biophysical featurization, non-parametric kinetic discovery (TICA, CKPCA, DPA clustering), and core statistical ML with SAM-MFP Pareto prioritization. (Bottom) Actionable outputs: 2D Free Energy Landscapes, SAM-MFP Pareto trade-off curves, transparent decision rules, 3D PyMOL B-factor mapping, and an interactive HTML report dashboard.*

---

## Why MolScope? Core Differentiators

| Capability | mdxplain & Standard Tools | MolScope Platform |
| :--- | :--- | :--- |
| **Simulation Integration** | None (requires precomputed trajectories) | **Native 3-stage pipeline (`molscope run-pdb`)**: automated OpenMM setup (pH 7.4 protonation, energy minimization, NVT equilibration, Langevin MD) |
| **Biophysical Featurization** | Pairwise residue contacts only | **12+ biophysical feature families**: pairwise contacts, Cα distance matrices, DSSP secondary structure ($\alpha, \beta$, coil), backbone/sidechain torsions ($\phi, \psi, \chi_1$), SASA, salt bridges, aromatic $\pi$-stacking |
| **State Discovery & Clustering** | Fixed-radius or pairwise comparison | **Density Peak Advanced (DPA)** on TICA / Contact-KPCA manifolds: automatically finds free energy basins and barrier saddle points without arbitrary cluster counts (**8.3x faster DPA, 23% lower RAM**) |
| **Machine Learning Integration** | Optional external export (*"Use datamatrix for own analysis"*) | **Core & Mandatory**: Regularized supervised ML (Random Forest, Logistic Regression, Decision Trees) trained on every run with Leave-One-Feature-Out (LOFO) global ranking |
| **Autocorrelation & Rigor** | Blind empirical variance; frame leakage risk | **SAM-MFP Pareto Prioritization**: overcomes the *"Variance Fallacy"*; purged temporal block CV and trajectory-grouped splits prevent data leakage |
| **3D Molecular Mapping** | Separate plotting | **Direct PDB B-factor embedding**: residue importances mapped to PDB B-factors with one-click PyMOL (`.pml`) and ChimeraX (`.cxc`) scripts |
| **Reporting & Interactivity** | Disjoint static plots | **Self-contained interactive dashboard (`report.html`)**: interactive tabs, 2D Free Energy Landscapes, timeline state transitions, and diagnostic charts |

---

## The 3-Stage Pipeline

```text
[Stage 1: PDB Input & OpenMM MD Simulation]
Raw PDB Structure ──► PDBFixer (pH 7.4) ──► Amber14-SB + GBN2 Implicit Solvent ──► Langevin MD (DCD)
                                     OR Direct Trajectory Ingestion (1–100+ Replicas)
                                            │
                                            ▼
[Stage 2: Multi-Family Featurization & Kinetic Discovery]
Feature Extraction (Contacts, Cα Distances, DSSP, Dihedrals, SASA, Salt Bridges)
    ──► Variational TICA & Contact Kernel PCA (CKPCA)
    ──► Density Peak Advanced (DPA) Clustering ──► Discrete Metastable Macrostates (S₁..Sk)
                                            │
                                            ▼
[Stage 3: Core Statistical ML & SAM-MFP Feature Prioritization]
SAM-MFP Multi-Objective Pareto Discovery (PCA Salience vs. ESS Ratio vs. Uniqueness)
    ──► Purged Trajectory-Grouped Supervised ML (Random Forest / GBDT)
    ──► Leave-One-Feature-Out (LOFO) Feature Ranking & Permutation Negative Controls
    ──► 3D B-Factor PDB Mapping + Decision Tree Rule Cards + report.html Dashboard
```

---

## Installation

MolScope requires Python 3.10 or newer.

```bash
# Clone the repository
git clone https://github.com/N-S8990/molscope.git
cd molscope

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Install MolScope with test, simulation, and visualization dependencies
pip install -e '.[test,simulation]'
```

Verify installation:
```bash
molscope --help
```

---

## Quickstart: 3 Ways to Run

### Method 1: The Single-Command 3-Stage PDB Runner (`molscope run-pdb`)

Take any crystal structure or AlphaFold PDB, simulate it with OpenMM, discover conformational states, and rank all molecular features:

```bash
molscope run-pdb \
  --pdb path/to/protein.pdb \
  --out-dir results/my_run \
  --steps 50000 \
  --report-interval 500 \
  --temperature 300.0 \
  --ph 7.4 \
  --states 2
```

**What happens automatically:**
1. **Stage 1 (Simulation):** Strips crystallographic artifacts, adds missing heavy atoms and hydrogens at pH 7.4, runs 500 L-BFGS minimization steps, 1,000 NVT equilibration steps, and integrates 100 ps of Langevin dynamics (Amber14-SB + GBN2 implicit solvent).
2. **Stage 2 (Clustering):** Extracts biophysical features, projects onto PCA/TICA, and clusters frames into macrostates via DPA.
3. **Stage 3 (Machine Learning):** Fits regularized ML models, performs LOFO ablation across all features, maps residue importances to PDB B-factors, and builds `report.html`.

---

### Method 2: High-Level Python API (`from molscope.api import Study`)

For interactive Jupyter notebooks or Python scripts with existing trajectories:

```python
from molscope.api import Study

# 1. Initialize study with on-the-fly PBC imaging
study = Study(project_id="villin_study", out_dir="results/villin")
study.add_system(
    system_id="villin_headpiece",
    topology="data/protein.pdb",
    trajectories=["data/trajectory.xtc"],
)
study.configure_imaging(unwrap=True, center_selection="protein", wrap=False)

# 2. Configure feature families
study.configure_features([
    "protein.ca_distances_contacts",
    "protein.secondary_structure_dssp",
    "protein.sidechain_torsions",
    "protein.salt_bridges",
])
study.run(extract_only=True)

# 3. Kinetic & Non-Linear Dimensionality Reduction
tica_res = study.run_tica(lag_time=2, components=5)
ckpca_res = study.run_contact_kpca(n_components=5)

# 4. State Discovery & Clustering
dpa_res = study.cluster_dpa(coordinate_name="TICA")

# 5. Free Energy Landscape & 3D Visualization Export
study.plot_free_energy_landscape(dim1="TIC1", dim2="TIC2", temperature_k=300.0)
study.explain_clusters(max_depth=3)
study.export_3d(out_dir="results/villin/3d_visualization")
```

---

### Method 3: Configurable Multi-Replica Project (`project.json`)

For large multi-trajectory cohorts with condition comparisons:

```bash
# Run full project pipeline
molscope run --project project.json --out-dir results/cohort_run
```

---

## Result Checks & Interpretation Guide

Every MolScope run generates a structured results directory. Use this guide to verify and interpret your outputs:

```text
results/my_run/
├── report.html                  <-- Standalone interactive dashboard (open in browser)
├── START_HERE.md                <-- Executive summary & feature guide
├── mapped_importance.pdb        <-- PDB with ML importances stored in B-factors
│
├── 3d_visualization/           <-- One-click PyMOL and ChimeraX visualizers
│   ├── molscope_3d.pml          <-- PyMOL script (spectrum b, blue_white_red)
│   ├── molscope_3d.cxc          <-- UCSF ChimeraX script
│   └── molscope_3d_residue_scores.csv
│
├── figures/                     <-- Publication-quality diagnostic plots
│   ├── sam_mfp_tradeoff.png     <-- Multi-objective Pareto front
│   ├── fes_TIC1_TIC2.png        <-- 2D Free Energy Surface (ΔG kcal/mol)
│   ├── cluster_timeline.png     <-- Cluster transitions over simulation time
│   ├── decision_tree_rules.png  <-- Transparent Boolean if-then rule cards
│   ├── predictive_feature_ranking.png <-- Grouped LOFO accuracy gain/drop
│   └── feature_correlation_map.png
│
├── tables/                      <-- Complete raw data matrices
│   ├── sam_mfp_candidates.csv   <-- Full Pareto candidate assessments
│   ├── dpa_cluster_assignments.csv
│   ├── decision_tree_rules.csv
│   └── global_feature_importance_all_features.csv
│
└── sim_output/                  <-- Simulation outputs (when run from PDB)
    ├── trajectory.dcd           <-- Langevin production trajectory
    └── prepared_topology.pdb    <-- Protonated Amber14-SB topology
```

### Artifact Inspection Checklist

| File / Artifact | Location | What to Check | Scientific Interpretation |
| :--- | :--- | :--- | :--- |
| **`report.html`** | Root | Open in Chrome/Firefox/Safari. Check that tabs (Overview, States, Landscapes, Rules, Figures, Files) render cleanly. | Self-contained offline dashboard summarizing all scientific outputs in one interactive interface. |
| **`sam_mfp_tradeoff.png`** | `figures/` | Check scatter plot of **Target-free PCA Salience** vs. **Effective Sample Size Ratio (`ESS_Ratio`)** or **Replica Consistency**. Orange dots = Pareto candidates; blue dots = other eligible features. | **Solves the Variance Fallacy**: High-salience features with near-zero ESS ratio are autocorrelated noise. Genuine candidates balance representation power with statistical independence. |
| **`sam_mfp_candidates.csv`** | `tables/` | Filter by `SAM_MFP_Pareto_Front == True`. Verify `SAM_MFP_Eligible == True`. Inspect `ESS_Ratio`, `Uniqueness`, and `PCA_Salience`. | Contains all non-dominated features along the multi-criteria Pareto boundary. No arbitrary scalar thresholding. |
| **`fes_TIC1_TIC2.png`** | `free_energy/` or `figures/` | 2D Free Energy Surface contour with colorbar $\Delta G$ (kcal/mol). Verify labeled star markers ($B_1, B_2, \dots, B_K$) identifying energy basins. | Depicts the conformational landscape. Basins with $\Delta G \le 2.0\text{ kcal/mol}$ represent thermally accessible metastable states at 300 K. |
| **`cluster_timeline.png`** | `figures/` | Time series of cluster assignments across simulation frames ($0$ to $N\text{ ps}$). | Confirms whether transitions between states are reversible. Detects kinetic trapping or rapid unphysical flickering. |
| **`decision_tree_rules.csv` / `.png`** | `tables/`, `figures/` | Review extracted Boolean rules (e.g. `IF CA_DIST(Thr24, Ser43) <= 15.9 Å AND ...`). Check **Precision** ($\ge 85\%$) and **Recall**. | Provides transparent, human-readable structural gates explaining exactly what geometric coordinates define each conformational state. |
| **`mapped_importance.pdb`** | Root & `3d_visualization/` | Open in PyMOL (`pymol molscope_3d.pml`) or run `spectrum b, blue_white_red`. Residues with high importance appear in bright red; neutral residues appear in white/blue. | Hotspot mapping: visually localizes allosteric hinges, active site loops, or domain-swapping contacts directly on the 3D ribbon. |
| **`global_feature_importance_all_features.csv`** | `tables/` | Global feature ranking sorted by LOFO importance or Gini score across all data. | Quantifies the relative contribution of each contact, dihedral, or secondary structure element to state discrimination. |

---

## The "Variance Fallacy" in MD Analysis

A widespread practice in molecular dynamics analysis is ranking features by **empirical variance** or **PCA loading magnitude**. MolScope's benchmark rigorously disproves this approach:

> **The Variance Fallacy:** Features with high raw variance in MD trajectories often exhibit severe temporal autocorrelation ($ESS / N < 0.01$) or extreme collinear redundancy ($|r_s| > 0.95$). Ranking by variance selects slow, un-converged fluctuations rather than reproducible conformational drivers.

MolScope's **SAM-MFP** addresses this by formulating feature prioritization as a multi-objective Pareto optimization:
$$\max \left\{ \text{Representation Salience}, \text{ESS Ratio}, \text{Information Uniqueness}, \text{Replica Consistency} \right\}$$

- **100% Ground-Truth Recovery** on planted synthetic benchmarks without labels (vs. 33% for empirical variance).
- In the 2RJY Villin Headpiece benchmark, SAM-MFP **rejects 12 of the top 20 variance features** because their effective sample size ratio was below $0.01$ or their mutual redundancy exceeded $90\%$.

---

## Matched Benchmark vs. mdxplain (2RJY Villin Headpiece)

On the 596-atom Villin Headpiece (PDB: `2RJY`, 2,000 frames stride 5 from 1,000 ns simulation):

| Performance Dimension | mdxplain | MolScope | Difference |
| :--- | :--- | :--- | :--- |
| **DPA State Clustering** | 0.472 s | **0.057 s** | **8.3x faster** |
| **Peak Memory (RAM)** | 1,211 MB | **929 MB** | **23.3% less memory** |
| **Feature Families** | Pairwise contacts only | **Contacts, Cα Distances, DSSP, Dihedrals, Salt Bridges, SASA** | **Comprehensive coverage** |
| **Machine Learning** | External data export | **Integrated LOFO Random Forest & GBDT** | **Mandatory & native** |
| **Multi-Objective Prioritization** | None | **SAM-MFP Pareto candidates** | **Resolves Variance Fallacy** |
| **3D Molecular Output** | None | **PDB B-factor mapping + PyMOL/ChimeraX scripts** | **One-click visualizer** |
| **Streaming Scaling** | OOM on large datasets | **Flat ~280 MiB footprint across 10k–1M frames** | **Production scalable** |

---

## Known Limitations and Scientific Scope

MolScope enforces strict methodological rigor, but researchers should be aware of essential boundaries:

1. **Single-Trajectory Cross-Validation & Autocorrelation:**
   Consecutive simulation frames are temporally autocorrelated, violating i.i.d. assumptions. While MolScope provides Purged Block K-Fold Cross-Validation with temporal embargo buffers for single-trajectory analysis, true out-of-sample physical generalization requires **independent simulation replicas** ($\ge 2$, ideally $\ge 6$) per condition.
2. **Periodic Boundary Conditions (PBC) & Coordinate Imaging:**
   Molecules diffusing across periodic boundaries produce artificial distance jumps if unmanaged. MolScope provides on-the-fly unwrapping and centering (`ImagingPolicy`), but users must ensure input trajectories contain unit-cell box dimensions (`dimensions` attribute).
3. **Simulation Parameter Responsibilities:**
   While MolScope automates OpenMM execution plans and checkpointed recovery, researchers retain full responsibility for verifying that chosen force fields (e.g. Amber14-SB), solvent models (e.g. GBN2), and protonation states (pH 7.4) are physically realistic for their target system.
4. **Descriptive Clustering vs. True Physical Metastability:**
   Geometric partitions (TICA, CKPCA, DPA, GMM) discover mathematically separated representations. High clustering validity indices confirm geometric separation, but do **not** independently prove physical metastability without implied timescale flattening across lag times.
5. **Statistical Importance vs. Biological Causality:**
   Feature importances from decision trees, LOFO accuracy drops, and SAM-MFP Pareto candidates identify correlating measurements within the provided trajectory ensemble. They do **not** constitute biological proof of allosteric mechanisms without experimental confirmation (e.g. site-directed mutagenesis).

---

## Documentation

- [User Guide: Project Runs](docs/user-guide/project-runs.md)
- [User Guide: CLI Reference](docs/user-guide/cli.md)
- [User Guide: Study Workspaces](docs/user-guide/workspaces.md)
- [User Guide: Input Validation & PBC Checks](docs/user-guide/input-checks.md)
- [Tutorial: 2RJY Villin Headpiece](docs/tutorials/2rjy_villin_tutorial.md)
- [Scientific Methods: SAM-MFP Specification](docs/scientific-methods/sam-mfp.md)
- [Scientific Methods: Machine Learning Validation & Rigor](docs/scientific-methods/ml-validation.md)
- [Scientific Methods: OpenMM Simulation Guide](docs/scientific-methods/openmm-simulation-guide.md)
- [Scientific Methods: mdxplain Comparison Notes](docs/scientific-methods/mdxplain-comparison.md)
- [Validation Summary & Benchmark Findings](docs/validation/summary.md)
- [Architecture Guide](docs/developer-guide/architecture.md)
- [Contributing Guidelines](CONTRIBUTING.md)

---

## License

MolScope is released under the [MIT License](LICENSE).
