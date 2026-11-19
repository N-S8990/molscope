# MolScope Interactive Jupyter Notebooks

This directory contains official, interactive Jupyter tutorial notebooks demonstrating the full continuum of MolScope's capabilities—from raw crystal structures and automated OpenMM simulations to multi-family biophysical featurization, non-parametric kinetic discovery, statistical machine learning, and 3D molecular visualization.

---

## Tutorial Suite Overview

| Notebook | Focus Area | Key Concepts Covered |
| :--- | :--- | :--- |
| **[`01_Quickstart_Villin_Headpiece.ipynb`](01_Quickstart_Villin_Headpiece.ipynb)** | **End-to-End Analysis Workflow** | `Study` API, on-the-fly PBC imaging, Cα distances/contacts, DSSP secondary structure, sidechain torsions, TICA, Contact KPCA, Density Peak Advanced (DPA) clustering, 2D Free Energy Surfaces (FES), Boolean rule cards, 3D PyMOL/ChimeraX export. |
| **[`02_Simulation_to_Discovery_Pipeline.ipynb`](02_Simulation_to_Discovery_Pipeline.ipynb)** | **Automated OpenMM Continuum** | Native 3-stage continuum (`molscope run-pdb`), pH 7.4 protonation, Amber14-SB + GBN2 implicit solvent, dry-run plan inspection, byte/frame size estimation, checkpoint recovery, immutable segment resumption. |
| **[`03_Statistical_ML_and_SAM_MFP_Rigor.ipynb`](03_Statistical_ML_and_SAM_MFP_Rigor.ipynb)** | **Statistical ML & Rigor** | The "Variance Fallacy", multi-objective SAM-MFP Pareto optimization, temporal autocorrelation & effective sample size ($ESS/N$), purged block cross-validation with embargo buffers, Leave-One-Feature-Out (LOFO) predictive ranking. |
| **[`04_Multi_Family_Biophysical_Featurization.ipynb`](04_Multi_Family_Biophysical_Featurization.ipynb)** | **Biophysical Descriptors** | 12+ feature families beyond simple contacts: Cα pair distances, DSSP helix/sheet/coil fractions, circular $\sin/\cos$ backbone and sidechain dihedrals, FreeSASA solvent exposure, salt-bridge geometry, aromatic $\pi$-stacking. |
| **[`05_Physical_Correctness_and_PBC_Imaging.ipynb`](05_Physical_Correctness_and_PBC_Imaging.ipynb)** | **Input Checks & PBC Imaging** | Pre-flight trajectory validation, missing atom/residue detection, unwrapped coordinate jump diagnosis, on-the-fly PBC imaging modes (`auto`, `center_and_wrap`, `no_wrap`), unit cell box verification, and structural profiling. |
| **[`06_Large_Scale_Streaming_Analysis.ipynb`](06_Large_Scale_Streaming_Analysis.ipynb)** | **Out-of-Core Streaming** | Constant-memory trajectory extraction, out-of-core chunked featurization, columnar Parquet persistence, streaming statistics aggregation, memory profiling, and linear scaling over multi-gigabyte trajectories. |
| **[`07_ML_Validation_and_Leakage_Prevention.ipynb`](07_ML_Validation_and_Leakage_Prevention.ipynb)** | **Leakage-Free Statistical ML** | Autocorrelation time ($\tau_{int}$) estimation, purged temporal block cross-validation, temporal embargo buffers, preventing optimistic data leakage, block permutation null controls, and simulation convergence checks. |
| **[`08_Configurable_Projects_and_Study_Workspaces.ipynb`](08_Configurable_Projects_and_Study_Workspaces.ipynb)** | **Declarative Workspaces & Studies** | Production workspace initialization (`initialize_study`), declarative `project.json` schema configuration, run lifecycle management (`begin_run`), run provenance tracking, and reproducible artifact trees. |

---

## Quickstart: Running the Notebooks

### 1. Environment Setup

Ensure you have MolScope and Jupyter installed in your virtual environment:

```bash
# Clone and navigate to MolScope
git clone https://github.com/N-S8990/molscope.git
cd molscope

# Activate your virtual environment
source .venv/bin/activate

# Install MolScope with full extras
pip install -e '.[dssp,simulation,parquet]'
pip install jupyterlab matplotlib ipywidgets
```

### 2. Launching JupyterLab

From the repository root:

```bash
jupyter lab notebooks/
```

Or start with the quickstart notebook directly:

```bash
jupyter lab notebooks/01_Quickstart_Villin_Headpiece.ipynb
```

---

## System Requirements & Datasets

- **Python**: 3.10, 3.11, 3.12, or 3.13.
- **HP35 Villin Headpiece Dataset (2RJY)**: The quickstart and featurization notebooks use the canonical 2RJY folding trajectory (`protein.pdb` and `simulation_run1.xtc`). The notebooks automatically detect the dataset if placed in `data/2RJY/` or adjacent folders.
- **Simulation Dependencies**: Notebook 02 requires `openmm` and `pdbfixer` (installed via `pip install -e '.[simulation]'`).
