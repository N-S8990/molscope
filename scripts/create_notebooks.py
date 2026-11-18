"""Generate official tutorial Jupyter notebooks for MolScope."""

from __future__ import annotations

import json
from pathlib import Path


def make_notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3 (ipykernel)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbformat_minor": 5,
                "pygments_lexer": "ipython3",
                "version": "3.10.12",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md_cell(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip().split("\n")],
    }


def build_notebook_01() -> dict:
    """Notebook 01: Quickstart Villin Headpiece."""
    cells = [
        md_cell(
            r"""# MolScope Quickstart: End-to-End Conformational Analysis

[![Open in JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange.svg)](https://jupyter.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Welcome to **MolScope**! This interactive tutorial walks you through an end-to-end conformational analysis of the canonical fast-folding **HP35 Villin Headpiece** (PDB: `2RJY`).

### What You Will Learn
1. **Initialize a Study**: Set up a stateful study workspace with on-the-fly Periodic Boundary Condition (PBC) imaging.
2. **Multi-Family Featurization**: Extract residue-residue contacts, Cα distances, DSSP secondary structure, and sidechain dihedrals.
3. **Dimensionality Reduction**: Project conformational dynamics onto **TICA** (slow kinetic transitions) and **Contact Kernel PCA** (CKPCA non-linear RKHS embeddings).
4. **State Discovery (DPA)**: Discover metastable conformational free energy basins without arbitrary cluster counts using **Density Peak Advanced**.
5. **Free Energy Landscapes**: Compute and visualize 2D Potential of Mean Force ($\Delta G$ in kcal/mol) surfaces with detected energy minima.
6. **Explainable AI**: Extract transparent, human-readable Boolean decision rules that structurally define each macrostate.
7. **3D Molecular Hotspots**: Map residue importances directly onto PDB B-factors with one-click PyMOL and ChimeraX visualizers.
8. **Interactive Dashboard**: Explore the self-contained offline HTML report (`report.html`)."""
        ),
        md_cell("## 1. Imports and Environment Setup"),
        code_cell(
            """import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Import MolScope's high-level interactive Study API
from molscope.api import Study

print(f"Python: {sys.version.split()[0]}")"""
        ),
        md_cell(
            """## 2. Locate System Data and Initialize Study

We use the HP35 Villin Headpiece (596 atoms, 35 residues, 3 $\\alpha$-helices forming a hydrophobic core).
We will locate the trajectory (`simulation_run1.xtc`) and topology (`protein.pdb`)."""
        ),
        code_cell(
            """# Automatically locate tutorial dataset
data_candidates = [
    Path("../mdxplain study/mdxplain/data/2RJY"),
    Path("data/2RJY"),
    Path("../data/2RJY"),
    Path("/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY"),
]

data_dir = next((p for p in data_candidates if (p / "protein.pdb").is_file()), None)

if data_dir is None:
    raise FileNotFoundError(
        "Could not find 2RJY dataset. Please ensure protein.pdb and simulation_run1.xtc are present."
    )

topology_file = data_dir / "protein.pdb"
trajectory_file = data_dir / "simulation_run1.xtc"

print(f"Topology:   {topology_file}")
print(f"Trajectory: {trajectory_file}")"""
        ),
        md_cell(
            """Now we initialize a stateful `Study` workspace.
The study manages data provenance, cached feature extractions, kinetic coordinates, and export directories."""
        ),
        code_cell(
            """out_dir = Path("results/quickstart_2rjy")
out_dir.mkdir(parents=True, exist_ok=True)

study = Study(
    project_id="2rjy_quickstart",
    out_dir=out_dir,
)

# Register the Villin Headpiece system
study.add_system(
    system_id="villin_headpiece",
    topology=topology_file,
    trajectories=[trajectory_file],
    condition_label="wild_type_300K",
)

# Configure Periodic Boundary Condition (PBC) unwrapping and centering
study.configure_imaging(unwrap=True, center_selection="protein", wrap=False)
print("Study initialized successfully.")"""
        ),
        md_cell(
            r"""## 3. Multi-Family Biophysical Featurization

Unlike tools limited to simple pairwise contacts, MolScope extracts multiple complementary biophysical feature families:
- **`protein.ca_distances_contacts`**: Pairwise Cα distances and binary contact graphs ($\le 8.0$ Å).
- **`protein.secondary_structure_dssp`**: Fractions of $\alpha$-helix, $\beta$-sheet, and coil.
- **`protein.sidechain_torsions`**: Circular $\sin/\cos$ rotamers for key hydrophobic core residues (Phe47, Phe51, Phe58)."""
        ),
        code_cell(
            """study.configure_features(
    plugins=[
        "protein.ca_distances_contacts",
        "protein.secondary_structure_dssp",
        "protein.sidechain_torsions",
    ],
    plugin_parameters={
        "protein.ca_distances_contacts": {
            "selection": "protein and name CA",
            "contact_cutoff_angstrom": 8.0,
            "max_pairs": 3000,
        },
        "protein.secondary_structure_dssp": {
            "per_residue": False,
        },
        "protein.sidechain_torsions": {
            "residue_ids": [47, 51, 58],  # Hydrophobic core rotamers
        },
    },
    generic_parameters={
        "frame_stride": 5,   # Subsample stride 5 (2,000 frames from 10,001)
        "max_frames": 2000,
    },
)

# Extract features
print("Extracting multi-family biophysical features...")
study.run(extract_only=True)

features = study.get_features()
manifest = study.get_manifest()

print(f"Extracted {features.shape[1]} features across {features.shape[0]} trajectory frames.")
features.iloc[:5, :6]"""
        ),
        md_cell(
            """## 4. Dimensionality Reduction: Variational TICA & Contact KPCA

To understand conformational transitions, we project high-dimensional coordinates onto:
1. **TICA (Time-lagged Independent Component Analysis)**: Finds the slowest kinetic relaxation modes by solving the generalized eigenvalue problem $C_\\tau v_i = \\lambda_i C_0 v_i$.
2. **Contact Kernel PCA (CKPCA)**: Non-linear kernel projection ($k(x, y) = \\exp(-\\gamma \\|x-y\\|^2)$) capturing non-linear contact manifolds."""
        ),
        code_cell(
            """# 1. Fit Variational TICA (lag time tau = 2 frames = 1.0 ns)
tica_result = study.run_tica(lag_time=2, components=5)

print("TICA Implied Timescales (ps):")
for i, ts in enumerate(tica_result.implied_timescales[:4], 1):
    print(f"  Mode {i}: {ts:.1f} ps")

# 2. Fit Contact Kernel PCA
ckpca_result = study.run_contact_kpca(n_components=4)
print(f"Contact KPCA computed {ckpca_result.coordinates.shape[1]} non-linear components.")"""
        ),
        code_cell(
            """# Inspect kinetic coordinates
tica_coords = tica_result.coordinates
tica_coords.head()"""
        ),
        md_cell(
            r"""## 5. Non-Parametric State Discovery: Density Peak Advanced (DPA)

MolScope employs **Density Peak Advanced (DPA)** clustering directly on the TICA kinetic manifold.
DPA automatically finds high-density free energy basins ($\rho$) and saddle-point barriers ($\delta$) **without requiring the user to specify an arbitrary number of clusters ($k$)**."""
        ),
        code_cell(
            """# Cluster on the slowest kinetic coordinates (TIC1 vs TIC2)
dpa_result = study.cluster_dpa(coordinate_name="TICA", z_score=2.0)

print(f"Discovered {dpa_result.n_clusters} conformational macrostates.")
for state_id, pop in dpa_result.state_populations.items():
    pct = (pop / len(features)) * 100
    print(f"  State {state_id}: {pop:4d} frames ({pct:5.1f}%)")"""
        ),
        code_cell(
            """# Plot DPA clusters on the TICA reaction coordinates
plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    tica_coords["TIC1"],
    tica_coords["TIC2"],
    c=dpa_result.assignments["Cluster"],
    cmap="tab10",
    s=12,
    alpha=0.7,
)
plt.colorbar(scatter, label="Discovered Macrostate")
plt.xlabel("TIC 1 (Slowest Kinetic Coordinate)")
plt.ylabel("TIC 2")
plt.title("MolScope DPA Conformational State Discovery on TICA Manifold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()"""
        ),
        md_cell(
            """## 6. 2D Free Energy Surface (FES) Landscape

We compute the potential of mean force:
$$\\Delta G(x, y) = -k_B T \\ln P(x, y)$$
and identify local thermodynamic energy basins ($B_1, B_2, \\dots$)."""
        ),
        code_cell(
            """fes_out = study.plot_free_energy_landscape(
    dim1="TIC1",
    dim2="TIC2",
    temperature_k=300.0,
    units="kcal/mol",
    bins=50,
    smooth=1.0,
)

print(f"Static FES plot:      {fes_out['static_plot']}")
print(f"Interactive HTML FES: {fes_out['interactive_plot']}")
print(f"Detected {len(fes_out['basins'])} free energy basins:")
fes_out['basins']"""
        ),
        md_cell(
            """## 7. Explainable AI: Decision Tree Boolean Rule Extraction

Rather than treating conformational states as black boxes, MolScope extracts human-interpretable Boolean decision rules:
$$\\text{IF } [\\text{Feature}_A \\le \\theta_A] \\text{ AND } [\\text{Feature}_B > \\theta_B] \\implies \\text{Macrostate } S_k$$"""
        ),
        code_cell(
            """explainer = study.explain_clusters(
    clustering_name="dpa",
    max_depth=3,
    min_samples_leaf=10,
)

print("Extracted Boolean Biophysical Rules for Discovered States:")
for rule in explainer.rules:
    print(f"\\n--- State {rule.cluster_id} ({rule.cluster_name}) ---")
    print(f"  Precision: {rule.precision * 100:.1f}% | Recall: {rule.recall * 100:.1f}% | Support: {rule.support_fraction * 100:.1f}%")
    print(f"  Rule: {rule.simplified_rule_text()}")"""
        ),
        md_cell(
            """## 8. 3D Hotspot Mapping for PyMOL & ChimeraX

MolScope writes residue-level ML importance scores directly into the crystallographic B-factor field of a prepared PDB, allowing instant 3D visualization."""
        ),
        code_cell(
            """vis_exports = study.export_3d(
    score_column="TICA_Salience",
    top_labels=6,
)

print("Generated 3D Visualization Assets:")
for name, path in vis_exports.items():
    print(f"  {name:15s}: {path}")"""
        ),
        md_cell(
            """To visualize in PyMOL:
```bash
pymol results/quickstart_2rjy/3d_visualization/2rjy_quickstart_importance.pml
```
Residues critical for state transitions will be highlighted with a red-white-blue heat spectrum."""
        ),
        md_cell(
            r"""## 9. Conclusion and Next Steps

In this quickstart tutorial, we used MolScope to:
1. Ingest raw MD trajectories with zero-leakage PBC imaging.
2. Extract multi-family biophysical descriptors (contacts, secondary structure, dihedrals).
3. Compute variational TICA and non-linear Contact KPCA manifolds.
4. Discover free energy basins using non-parametric Density Peak Advanced (DPA) clustering.
5. Compute the 2D Free Energy Surface ($\Delta G$ kcal/mol) with basin detection.
6. Extract transparent Boolean rules explaining the physical determinants of each state.
7. Export 3D PyMOL and ChimeraX structural visualizations.

**Next Tutorials:**
- `02_Simulation_to_Discovery_Pipeline.ipynb`: Running automated OpenMM simulations directly from PDB files.
- `03_Statistical_ML_and_SAM_MFP_Rigor.ipynb`: Machine Learning rigor, purged cross-validation, and solving the Variance Fallacy.
- `04_Multi_Family_Biophysical_Featurization.ipynb`: Deep dive into 12+ biophysical feature families."""
        ),
    ]
    return make_notebook(cells)


def build_notebook_02() -> dict:
    """Notebook 02: OpenMM Simulation to Discovery Pipeline."""
    cells = [
        md_cell(
            """# From Crystal Structure to Conformation: Automated OpenMM Simulation & Discovery Pipeline

[![Open in JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange.svg)](https://jupyter.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

One of MolScope's most distinctive capabilities is its **native 3-stage simulation-to-discovery continuum**. Researchers can supply a raw crystal structure or AlphaFold PDB and MolScope will:
1. Automatically fix, protonate at pH 7.4, and parameterize the system in OpenMM.
2. Generate transparent, reviewable execution plans (`openmm_plan.json`) with dry-run byte and frame estimates.
3. Integrate Langevin molecular dynamics with checkpointing and provenance tracking.
4. Chain directly into multi-family featurization, DPA state discovery, and ML feature prioritization."""
        ),
        md_cell("## 1. Imports and Verification"),
        code_cell(
            """import json
from pathlib import Path
import tempfile
import matplotlib.pyplot as plt
import pandas as pd

from molscope.core.project import (
    ProjectSpec, SystemSpec, EnsembleSpec, ReplicaSpec, OpenMMSimulationSpec
)
from molscope.simulation.openmm import (
    write_openmm_plan, inspect_openmm_plan, execute_openmm_plan
)

print("OpenMM simulation planning modules imported successfully.")"""
        ),
        md_cell(
            """## 2. Defining an OpenMM Simulation Protocol

We specify a complete simulation protocol declaratively:
- **Force field**: Amber14-SB with GBN2 implicit solvent (`implicit/gbn2.xml`).
- **Thermostat**: Langevin integrator at 300 K with 1.0 ps⁻¹ friction.
- **Timestep**: 2.0 fs with hydrogen bonds constrained.
- **Phases**: Energy minimization (L-BFGS), NVT thermal equilibration, and Langevin production dynamics."""
        ),
        code_cell(
            """protocol = OpenMMSimulationSpec(
    timestep_femtoseconds=2.0,
    temperature_kelvin=300.0,
    friction_per_picosecond=1.0,
    nonbonded_method="NoCutoff",
    constraints="HBonds",
    minimization_max_iterations=100,
    equilibration_steps=500,
    production_steps=5000,          # 10 ps production for demonstration
    report_interval_steps=100,       # Save every 0.2 ps (50 frames total)
    platform="CPU",                  # Use 'CUDA' or 'OpenCL' on GPU instances
    forcefield_files=("amber14-all.xml", "implicit/gbn2.xml"),
    ph=7.4,
    seed=42,
)

print("Simulation Protocol configured:")
print(f"  Temperature: {protocol.temperature_kelvin} K")
print(f"  Production:  {protocol.production_steps} steps ({protocol.production_duration_ns:.3f} ns)")
print(f"  Save Rate:   Every {protocol.report_interval_steps} steps ({protocol.save_interval_ps:.2f} ps)")"""
        ),
        md_cell(
            """## 3. Creating a Project Specification and Generating the Execution Plan

We create a `ProjectSpec` referencing our input structure and generate a portable, reviewable plan."""
        ),
        code_cell(
            """# Locate test protein fixture
fixture_pdb = Path("tests/fixtures/1A22.pdb")
if not fixture_pdb.is_file():
    fixture_pdb = Path("../tests/fixtures/1A22.pdb")

run_dir = Path("results/simulation_demo")
run_dir.mkdir(parents=True, exist_ok=True)

system = SystemSpec(
    system_id="growth_factor_hgh",
    topology=fixture_pdb,
    simulation_protocol=protocol,
    ensembles=(
        EnsembleSpec(
            ensemble_id="nvt_production",
            replicas=(
                ReplicaSpec(replica_id="rep_01", trajectory=run_dir / "sim_output/trajectory.dcd"),
            ),
        ),
    ),
)

project = ProjectSpec(project_id="demo_sim_to_discovery", systems=(system,))

plan_path, launcher_path = write_openmm_plan(project, run_dir / "sim_output")
print(f"Plan generated:     {plan_path}")
print(f"Launcher script:    {launcher_path}")"""
        ),
        md_cell(
            """## 4. Dry-Run Plan Inspection & Byte/Frame Estimation

Before spending expensive GPU/CPU compute, MolScope inspects the plan without importing heavyweight simulation dependencies:
- Verifies input file existence and SHA-256 digests.
- Calculates exact expected frame counts.
- Estimates trajectory file size in bytes."""
        ),
        code_cell(
            """with open(plan_path) as f:
    plan = json.load(f)

preview = inspect_openmm_plan(plan)
job = preview["jobs"][0]

print(f"Job ID:             {job['job_id']}")
print(f"Status:             {job['status']}")
print(f"Input Structure:    {job['input_structure']}")
print(f"Expected Frames:    {job['estimate']['expected_frames']}")
print(f"Estimated Atom Count: {job['estimate']['input_atom_count']}")
print(f"Approx DCD Bytes:   {job['estimate']['approximate_dcd_bytes']:,} bytes")"""
        ),
        md_cell(
            """## 5. Checkpointed Execution & Segment Resumption

MolScope guarantees reproducible simulations:
- Generates an immutable `execution_provenance.json` containing SHA-256 hashes of the topology, force field, serialized system XML, and protocol parameters.
- Writes binary checkpoints (`checkpoint.chk`) periodically.
- Interrupted or crashed runs can be resumed into immutable child segments without altering parent replica history."""
        ),
        code_cell(
            """print("To execute this simulation plan on your machine:")
print(f"  python -m molscope.simulation.openmm {plan_path}")
print("\\nOr use the single-command CLI:")
print(f"  molscope run-pdb --pdb {fixture_pdb} --out-dir results/my_pdb_run --steps 10000 --states 2")"""
        ),
        md_cell(
            """## 6. Chaining Directly into MolScope Discovery

Once the simulation completes, the output trajectory (`trajectory.dcd`) and prepared topology (`prepared.pdb`) are loaded into a `Study` with zero manual intervention:

```python
study = Study(project_id="sim_discovery", out_dir="results/discovery_results")
study.add_system(
    system_id="hgh_system",
    topology=run_dir / "sim_output/prepared_topology.pdb",
    trajectories=[run_dir / "sim_output/trajectory.dcd"],
)
study.configure_imaging(unwrap=True, center_selection="protein")
study.configure_features(["protein.ca_distances_contacts", "protein.secondary_structure_dssp"])
study.run()
```"""
        ),
    ]
    return make_notebook(cells)


def build_notebook_03() -> dict:
    """Notebook 03: Statistical ML Rigor & SAM-MFP."""
    cells = [
        md_cell(
            """# Statistical Machine Learning Rigor & SAM-MFP Feature Prioritization

[![Open in JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange.svg)](https://jupyter.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

A critical differentiator of MolScope is its mathematical and statistical rigor. This notebook explores:
1. **The Variance Fallacy**: Why ranking molecular features by empirical variance or PCA loadings selects autocorrelated noise.
2. **SAM-MFP Multi-Objective Optimization**: Finding non-dominated features along the 4-criteria Pareto front.
3. **Temporal Data Leakage Prevention**: Purged Block K-Fold Cross-Validation and trajectory-grouped splits.
4. **Supervised ML & LOFO Ranking**: Regularized Random Forests and Leave-One-Feature-Out feature importance."""
        ),
        md_cell("## 1. The 'Variance Fallacy' in Molecular Dynamics"),
        md_cell(
            """> **The Variance Fallacy:**
> In molecular dynamics, consecutive simulation frames are strongly temporally correlated. Features with the largest raw sample variance frequently correspond to slow, un-converged fluctuations with Effective Sample Size ratios ($ESS / N < 0.01$) or extreme collinear redundancy ($|r| > 0.95$). Ranking by variance fails to recover true conformational transition drivers.

Let's inspect the benchmark findings from the HP35 Villin Headpiece where variance was compared against SAM-MFP:"""
        ),
        code_cell(
            """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load benchmark Pareto data
bench_path = Path("benchmarks/sam_mfp/variance_fallacy_rejected_features.csv")
if not bench_path.is_file():
    bench_path = Path("../benchmarks/sam_mfp/variance_fallacy_rejected_features.csv")

if bench_path.is_file():
    rejected_df = pd.read_csv(bench_path)
    print(f"Loaded {len(rejected_df)} high-variance features rejected by SAM-MFP:")
    print(rejected_df[["Feature", "Empirical_Variance", "ESS_Ratio", "Rejection_Reason"]].head(8))
else:
    print("Benchmark tables located in benchmarks/sam_mfp/")"""
        ),
        md_cell(
            """## 2. SAM-MFP Formulation

MolScope formulates feature discovery as a multi-objective Pareto optimization problem:
$$\\max \\left\\{ \\text{Representation Salience}, \\text{ESS Ratio}, \\text{Information Uniqueness}, \\text{Replica Consistency} \\right\\}$$

- **Representation Salience**: Variance captured along PCA / TICA / CKPCA projection manifolds.
- **ESS Ratio ($N_{\\text{eff}} / N$)**: Statistical independence accounting for temporal autocorrelation (via initial positive sequence autocorrelation time $\\tau_{\\text{int}}$).
- **Information Uniqueness**: Orthogonality to already-selected features ($1 - \\max |r_s|$).
- **Replica Consistency**: Cross-replica Rank Cosine Similarity (RCS) across independent simulation trajectories."""
        ),
        code_cell(
            """# Load Pareto front candidates from real 2RJY benchmark
pareto_path = Path("benchmarks/sam_mfp/real_2rjy_pareto_candidates.csv")
if not pareto_path.is_file():
    pareto_path = Path("../benchmarks/sam_mfp/real_2rjy_pareto_candidates.csv")

if pareto_path.is_file():
    pareto_df = pd.read_csv(pareto_path)
    print(f"Identified {len(pareto_df)} non-dominated Pareto features:")
    print(pareto_df.head(6))"""
        ),
        md_cell(
            """## 3. Visualizing the SAM-MFP Pareto Trade-Off Curve

The plot below shows how MolScope separates genuine conformational drivers from autocorrelated noise:"""
        ),
        code_cell(
            """# Plot trade-off curve
tradeoff_png = Path("docs/images/sam_mfp_tradeoff.png")

fig, ax = plt.subplots(figsize=(8, 6))

if pareto_path.is_file():
    # If table is available, generate dynamic scatter
    ax.scatter(
        pareto_df["PCA_Salience"],
        pareto_df["ESS_Ratio"],
        c="darkorange",
        s=80,
        edgecolor="black",
        label="SAM-MFP Pareto Front (Optimal)",
    )
    ax.set_xlabel("Target-Free PCA Salience (Representation Power)")
    ax.set_ylabel("Effective Sample Size Ratio (Statistical Independence)")
    ax.set_title("SAM-MFP Multi-Objective Pareto Frontier")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.show()
else:
    print("Run benchmarks/sam_mfp/run_benchmarks.py to regenerate live Pareto data.")"""
        ),
        md_cell(
            """## 4. Preventing Data Leakage in MD Machine Learning

A common pitfall in biophysical machine learning is applying standard $K$-fold cross-validation to time-series trajectories. Consecutive frames $t$ and $t+1$ are near-identical; if $t$ is in train and $t+1$ is in test, models achieve artificially inflated accuracy via interpolation.

MolScope prevents this through:
1. **Purged Block Cross-Validation**: Divides trajectories into contiguous blocks and discards an *embargo window* around test splits.
2. **Trajectory-Grouped Splitting**: Enforces that entire independent simulation replicas are strictly segregated into train or validation folds."""
        ),
        code_cell(
            """from molscope.analysis.quality.statistical_rigor import PurgedBlockTimeSeriesSplit

splitter = PurgedBlockTimeSeriesSplit(n_splits=5, embargo_pct=0.02)
print("Purged Block Cross-Validation configured with 2% temporal embargo.")"""
        ),
        md_cell(
            """## 5. Supervised Feature Ranking: Grouped LOFO

MolScope uses **Leave-One-Feature-Out (LOFO)** ablation rather than Gini impurity to rank feature predictive importance:
$$\\Delta \\text{Score}_j = \\text{Score}(\\mathcal{F}) - \\text{Score}(\\mathcal{F} \\setminus \\{j\\})$$
This directly measures the unique predictive value of each coordinate while controlling for collinearity."""
        ),
    ]
    return make_notebook(cells)


def build_notebook_04() -> dict:
    """Notebook 04: Multi-Family Biophysical Featurization."""
    cells = [
        md_cell(
            r"""# Multi-Family Biophysical Featurization: Beyond Pairwise Contacts

[![Open in JupyterLab](https://img.shields.io/badge/Launch-JupyterLab-orange.svg)](https://jupyter.org)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Most molecular dynamics analysis packages limit featurization to pairwise contact maps. MolScope provides a rich library of **12+ biophysical feature families**, allowing researchers to track secondary structure transitions, rotamer flips, solvent exposure, and electrostatic networks.

In this notebook, we explore the feature plugins available in MolScope:
1. **Cα Distances & Interaction Graphs**
2. **Secondary Structure Tracking via DSSP**
3. **Backbone & Sidechain Dihedrals ($\phi, \psi, \chi_1$)**
4. **Solvent Accessible Surface Area (SASA)**
5. **Salt Bridges & Electrostatic Networks**
6. **Aromatic $\pi$-$\pi$ Stacking Geometries**"""
        ),
        md_cell("## 1. Feature Family Overview"),
        code_cell(
            """import pandas as pd
from molscope.registries import get_feature_registry

registry = get_feature_registry()
print(f"Registered Feature Plugins in MolScope: {len(registry.list_all())}")
for name in registry.list_all():
    print(f"  - {name}")"""
        ),
        md_cell(
            """## 2. Feature Configurations and Mathematical Definitions

| Feature Family | Plugin Name | Mathematical Representation | Scientific Use Case |
| :--- | :--- | :--- | :--- |
| **Cα Distances** | `protein.ca_distances_contacts` | $d_{ij} = \\|\\mathbf{r}_i - \\mathbf{r}_j\\|$ and $\\Theta(d_{\\text{cut}} - d_{ij})$ | Domain movements, hinge flexing |
| **Secondary Structure** | `protein.secondary_structure_dssp` | $\\alpha$-helix, $\\beta$-sheet, coil fractions | Unfolding, secondary structure transitions |
| **Backbone Dihedrals** | `protein.backbone_torsions` | $\\sin(\\phi), \\cos(\\phi), \\sin(\\psi), \\cos(\\psi)$ | Ramachandran regional shifts |
| **Sidechain Rotamers** | `protein.sidechain_torsions` | $\\sin(\\chi_1), \\cos(\\chi_1), \\sin(\\chi_2), \\cos(\\chi_2)$ | Hydrophobic core packing, allosteric triggers |
| **Solvent Exposure** | `solvation.sasa` | Surface area ($Å^2$) via Shrake-Rupley / FreeSASA | Cavity opening, hydrophobic burial |
| **Salt Bridges** | `protein.salt_bridges` | $d(\\text{Arg/Lys}^+, \\text{Asp/Glu}^-) \\le 4.0$ Å | Electrostatic latching, pH stability |
| **Aromatic Stacking** | `protein.aromatic_stacking` | Centroid distance $d \\le 5.5$ Å, normal angle $\\theta$ | $\\pi$-network stabilization |"""
        ),
        md_cell("## 3. Configuring and Extracting Multi-Family Panels"),
        code_cell(
            """from pathlib import Path
from molscope.api import Study

# Locate 2RJY dataset
data_candidates = [
    Path("../mdxplain study/mdxplain/data/2RJY"),
    Path("data/2RJY"),
    Path("/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY"),
]
data_dir = next((p for p in data_candidates if (p / "protein.pdb").is_file()), None)

if data_dir is not None:
    study = Study(project_id="feature_showcase", out_dir="results/features")
    study.add_system(
        system_id="villin",
        topology=data_dir / "protein.pdb",
        trajectories=[data_dir / "simulation_run1.xtc"],
    )
    study.configure_imaging(unwrap=True, center_selection="protein")
    study.configure_features([
        "protein.ca_distances_contacts",
        "protein.secondary_structure_dssp",
        "protein.sidechain_torsions",
    ])
    print("Multi-family panel configured.")"""
        ),
        md_cell(
            """## 4. Feature Redundancy and Correlation Analysis

Extracting many features can introduce high collinearity. MolScope assesses pairwise Spearman rank correlations ($|r_s|$) to ensure selected features provide orthogonal biophysical information."""
        ),
        code_cell(
            """print("To explore feature correlation maps and quality metrics:")
print("  study.get_quality_metrics()")
print("  study.get_features().corr(method='spearman')")"""
        ),
    ]
    return make_notebook(cells)


def main():
    notebooks_dir = Path("notebooks")
    notebooks_dir.mkdir(parents=True, exist_ok=True)

    nb1 = build_notebook_01()
    nb2 = build_notebook_02()
    nb3 = build_notebook_03()
    nb4 = build_notebook_04()

    (notebooks_dir / "01_Quickstart_Villin_Headpiece.ipynb").write_text(
        json.dumps(nb1, indent=2), encoding="utf-8"
    )
    (notebooks_dir / "02_Simulation_to_Discovery_Pipeline.ipynb").write_text(
        json.dumps(nb2, indent=2), encoding="utf-8"
    )
    (notebooks_dir / "03_Statistical_ML_and_SAM_MFP_Rigor.ipynb").write_text(
        json.dumps(nb3, indent=2), encoding="utf-8"
    )
    (notebooks_dir / "04_Multi_Family_Biophysical_Featurization.ipynb").write_text(
        json.dumps(nb4, indent=2), encoding="utf-8"
    )

    print("All 4 Jupyter notebooks successfully generated in notebooks/ directory.")


if __name__ == "__main__":
    main()
