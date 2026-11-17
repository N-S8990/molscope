# MolScope

MolScope is a Python platform for inspecting molecular dynamics (MD)
trajectories, extracting applicable molecular features, and comparing the
evidence those measurements provide. It combines explicit system profiling,
feature quality checks, target-free state discovery, and SAM-MFP Pareto
prioritization in a configurable project workflow.

The retained GPR6 study result, its status, and the remaining publication
gates are in [CURRENT_STATUS.md](CURRENT_STATUS.md).

```text
project.json → validate and profile → extract applicable features
             → assess quality and discover states
             → prioritize candidate measurements → report
```

Predictive models run automatically when a project declares one unambiguous valid
target through a prediction task or complete categorical trajectory label. Their
rankings are reported separately from descriptive state explanations and
SAM-MFP's Pareto candidates. Unavailable features carry applicability status
and reasons; they are not silently filled with zeroes.

## What it can do

- Validate topology and trajectory inputs, preserve system, ensemble, and
  replica identities, and summarize system composition and capabilities.
- Extract generic geometry and dynamics features, plus supported plugins for
  protein distances and contacts, backbone torsions, DSSP, hydrogen bonds,
  aromatic stacking, salt bridges, hydrophobic contacts, SASA, solvation
  shells, molecular dipoles, and protein contact networks. A nucleic-acid
  base-center proximity descriptor is available; it is not a base-pairing
  classifier.
- Report feature quality, run target-free PCA/TICA conformational-state
  and DPA cluster discovery, and present SAM-MFP Pareto candidates.
- **Mandatory Machine Learning & Global Feature Ranking:** ML is a non-optional
  core component executed on every pipeline run. MolScope trains regularized
  predictive models (Random Forest, Logistic Regression, and Decision Tree
  explainers) using purged temporal block cross-validation—either on user-declared
  targets or automatically on discovered conformational macrostates—to rank all
  features across all data without autocorrelation leakage.
- Generate reviewed OpenMM plans and, with OpenMM/PDBFixer installed, execute
  simulations with checkpoints and recovery into recorded trajectory segments.
- Stream a limited feature subset to checksummed chunks, resume interrupted
  extraction, and analyze large frame collections in bounded batches.

Plugin applicability, execution limits, imaging assumptions, and current
validation evidence are described in the [test readiness guide](docs/validation/test-readiness.md).

## Install

MolScope requires Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the test and simulation extras:

```bash
python -m pip install -e '.[test,simulation]'
```

OpenMM plan generation does not require the simulation extra. Executing a plan
does. For Conda setup and platform-specific notes, see the
[environment guide](docs/user-guide/environment.md).

Check the installed commands with:

```bash
molscope --help
# Or from a source checkout:
python -m molscope --help
```

## Start a study

Initialize a workspace outside the source checkout, then edit its project file
to point to your input data:

```bash
molscope init /path/to/my-study --project-id my_study
```

The initializer creates `project.json` and directories for inputs, preparation,
simulations, runs, and cache. The starter project uses placeholder input paths;
replace them before running. Trajectory files can stay where they are and be
referenced by absolute or project-relative paths.

For example, a minimal project declaration looks like this:

```json
{
  "schema_version": "1",
  "project_id": "my_study",
  "systems": [{
    "system_id": "protein_a",
    "topology": "inputs/topology.pdb",
    "ensembles": [{
      "ensemble_id": "condition_a",
      "replicas": [{
        "replica_id": "replica_01",
        "trajectory": "inputs/replica_01.xtc"
      }]
    }]
  }],
  "analysis": {
    "generic_feature_parameters": {"frame_stride": 1},
    "plugins": ["protein.ca_distances_contacts"],
    "plugin_parameters": {
      "protein.ca_distances_contacts": {
        "selection": "protein and name CA",
        "contact_cutoff_angstrom": 8.0,
        "max_pairs": 5000
      }
    }
  }
}
```

Run extraction and its report, or run the full configured discovery workflow:

```bash
molscope run --study /path/to/my-study --run-id extraction-001 --extract-only
molscope run --study /path/to/my-study --run-id analysis-001
```

Each run records the resolved project, input fingerprints, log, completion
status, and artifacts. Outputs are grouped into validation, features, quality,
discovery, predictive, prioritization, and report directories as applicable.
Run IDs must be new; use a fresh ID to retry a failed run. See the
[project run guide](docs/user-guide/project-runs.md) and
[workspace guide](docs/user-guide/workspaces.md).

## Worked Examples

### 1. End-to-End 3-Stage PDB Pipeline (`molscope run-pdb`)

MolScope can run a complete 3-stage pipeline—from a raw PDB structure through OpenMM simulation, biophysical feature extraction, density-peak clustering, and regularized ML feature ranking—in a single command:

```bash
# Run simulation (100 ps, 50k steps), DPA clustering, and ML feature ranking
molscope run-pdb \
  --pdb protein.pdb \
  --out-dir results/pdb_pipeline \
  --steps 50000 \
  --report-interval 500 \
  --temperature 300.0 \
  --ph 7.4 \
  --states 2
```

**Real output:**
```text
[Stage 1/3] Initiating OpenMM simulation for protein.pdb...
  Production steps: 50000 (100.0 ps)
  Report interval: 500 steps
  Temperature: 300.0 K | pH: 7.4
  Trajectory successfully generated: results/pdb_pipeline/sim_output/trajectory.dcd
[Stage 1/3 Complete] OpenMM simulation finished successfully.

[Stage 2/3] Extracting biophysical features, discovering states & DPA clusters...
  Extracted features across frames, performed TICA & DPA clustering.

[Stage 3/3] Training regularized ML model and ranking all features across all data...
  Residue importances mapped to PDB Beta-factors: results/pdb_pipeline/mapped_importance.pdb

=======================================================
MolScope End-to-End Pipeline Completed Successfully!
All results, clusters, ML models, and reports written to:
  results/pdb_pipeline
=======================================================
```

The resulting directory provides:
- `sim_output/`: Prepared topology, OpenMM plan, checkpoint, and trajectory (`trajectory.dcd`).
- `tables/`: Extracted feature matrices, state populations, and transition matrices.
- `figures/`: PCA density projections, TICA free energy contours, and feature correlation maps.
- `3d_visualization/`: `mapped_importance.pdb` (residue importances stored in B-factors) and export scripts for PyMOL (`.pml`) and ChimeraX (`.cxc`).
- `report.html`: Self-contained interactive report.

### 2. HP35 Villin Headpiece (2RJY) Benchmark

The canonical benchmark on the 596-atom Villin Headpiece (PDB: `2RJY`, 2,000 frames sampled at stride 5 from a 1,000 ns simulation) evaluates the interactive `Study` API:

```python
from molscope.api import Study

# 1. Initialize study with on-the-fly PBC imaging
study = Study(project_id="2rjy_benchmark", out_dir="results/2rjy")
study.add_system(
    system_id="villin_headpiece",
    topology="data/2RJY/protein.pdb",
    trajectories=["data/2RJY/simulation_run1.xtc"],
)
study.configure_imaging(unwrap=True, center_selection="protein", wrap=False)

# 2. Extract 3,818 biophysical features (Cα contacts, DSSP, sidechain torsions)
study.configure_features([
    "protein.ca_distances_contacts",
    "protein.secondary_structure_dssp",
    "protein.sidechain_torsions",
])
study.run(extract_only=True)

# 3. Kinetic & Non-Linear Dimensionality Reduction
tica_res = study.run_tica(lag_time=2, components=5)
ckpca_res = study.run_contact_kpca(n_components=5)

# 4. State discovery & Explainability
dpa_res = study.cluster_dpa(coordinate_name="TICA")
explainer = study.explain_clusters(max_depth=3)
study.plot_free_energy_landscape(dim1="TIC1", dim2="TIC2", temperature_k=300.0)
study.export_3d(out_dir="results/2rjy/3d_visualization")
```

**Measured benchmark results (`benchmarks/run_2rjy_matched_benchmark.py`):**
| Stage | Metric / Method | Result | Runtime |
| :--- | :--- | :--- | :--- |
| **Feature Extraction** | 3,818 features across 2,000 frames | Cα contacts, DSSP, sidechains | 4.92 s |
| **Kinetic Projection** | Variational TICA (lag $\tau = 2$) | Implied timescales: 25.1 ns, 14.8 ns | 1.24 s |
| **Non-Linear Embedding**| Contact Kernel PCA (CKPCA) | Radial basis contact kernel | 11.18 s |
| **State Discovery** | Density Peak Advanced (DPA) | **6 conformational states** discovered | 0.05 s |
| **Explainable AI** | Shallow Decision Trees | Top rule: $d(\text{Ala26}_{\text{C}\alpha}, \text{Pro30}_{\text{C}\alpha}) \le 6.4$ Å | 3.58 s |
| **Footprint** | Total wall time & peak RAM | **21.49 s total** | **644 MB peak** |

Detailed walkthrough: [2RJY Villin Tutorial](docs/tutorials/2rjy_villin_tutorial.md).

## Generate molecular dynamics

Simulation generation and analysis are separate steps. Put a reviewed
`simulation_protocol` on a system in `project.json`, then create a plan:

```bash
molscope plan --project /path/to/my-study/project.json \
  --out-dir /path/to/my-study/simulations/plan-001
molscope simulate --plan /path/to/my-study/simulations/plan-001/openmm_plan.json \
  --dry-run
```

Review the selected structure, force field, preparation policy, timestep,
production length, and expected output size. Execute only when ready:

```bash
molscope simulate --plan /path/to/my-study/simulations/plan-001/openmm_plan.json \
  --run
```

MolScope does not establish that a force field or preparation procedure is
appropriate for a particular molecule. Validate those choices for your study.
Recovery can resume an unfinished checkpoint into a new segment. Segments are
not automatically concatenated and must retain their parent replica identity.

## Large trajectories

The separate streaming workflow writes checksummed feature chunks and computes
descriptive PCA/clustering on a bounded sample before assigning frames in
batches. Current streaming extraction supports selected geometry and protein
C-alpha distance/contact features, with explicit as-stored nonperiodic
coordinates. It does not run the complete plugin catalog, predictive ML,
replica stability, or SAM-MFP.

The million-row synthetic benchmark measures the chunked analysis machinery,
not molecular trajectory extraction or a million-frame MD simulation. It is
not a universal capacity guarantee. Review the
[scalable workflow and limitations](docs/user-guide/scalable-runs.md) before
planning a large run.

## Tests and scientific validation

Run the suite from the repository root:

```bash
python -m pytest -q
```

Tests cover contracts, numerical behavior, integrations, and selected reference
comparisons. They do not validate every method for every molecular system.
Matched comparisons with mdxplain require aligned feature definitions,
preprocessing, frames, and resources. The current project has not established
that MolScope is superior to mdxplain. See the
[validation guide](docs/validation/test-readiness.md) and
[comparison notes](docs/scientific-methods/mdxplain-comparison.md).

## Known Limitations and Scientific Scope

MolScope enforces strict methodological rigor, but researchers should be aware of essential boundaries:

1. **Single-Trajectory Cross-Validation & Autocorrelation**:
   Consecutive MD simulation frames are temporally autocorrelated, violating independent and identically distributed (i.i.d.) statistical assumptions. Random frame splitting yields severely over-optimistic generalization metrics. While MolScope provides Purged K-Fold Cross-Validation with temporal embargo buffers for single-trajectory state analysis, true out-of-sample physical generalization requires **independent simulation replicas** ($\ge 2$, ideally $\ge 6$) per condition.
2. **Periodic Boundary Conditions (PBC) & Coordinate Imaging**:
   Molecules diffusing across periodic box boundaries produce artificial distance jumps and broken contacts if unmanaged. MolScope provides on-the-fly unwrapping and centering (`ImagingPolicy`), but users must ensure input trajectories contain unit-cell box dimensions (`dimensions` attribute) and specify valid center selections (`protein`).
3. **Simulation Parameter & Force Field Choice Responsibilities**:
   While MolScope automates OpenMM execution plans and checkpointed recovery, it **does not validate** whether a specific force field (e.g., Amber14SB, CHARMM36m), implicit solvent model (e.g., GBN2), or protonation assignment (e.g., PDBFixer at pH 7.4) is physically realistic for a given target system. Researchers retain full scientific responsibility for verifying simulation conditions against experimental data.
4. **Descriptive Clustering vs. Physical Metastability**:
   State discovery (via TICA, Contact KPCA, DPA, or GMM) discovers geometric partitions of sampled representations. High clustering validity indices (Silhouette, Davies-Bouldin) confirm mathematical separation in reduced dimensions, but do **not** independently prove physical metastability or true Markovian kinetics without implied timescale flattening across lag times.
5. **Statistical Salience vs. Biological Causality**:
   Feature importances from decision trees, Leave-One-Feature-Out (LOFO) accuracy drops, and SAM-MFP Pareto candidates identify correlating measurements within the provided trajectory ensemble. They do **not** constitute biological proof of allosteric mechanisms without experimental confirmation (e.g. site-directed mutagenesis).

## Project layout

```text
src/molscope/       Implementation organized by responsibility
tests/              Unit, integration, and reference tests with small fixtures
examples/           Portable project templates
benchmarks/         Benchmark definitions and evaluation protocols
docs/               User guides, scientific methods, validation, and developer notes
```

Research inputs, generated trajectories, caches, and run reports belong in a
study workspace rather than in the package source tree.

## Documentation

- [All documentation](docs/README.md)
- [Command reference](docs/user-guide/cli.md)
- [End-to-End 2RJY Tutorial](docs/tutorials/2rjy_villin_tutorial.md)
- [Shared Evaluation Protocol](benchmarks/PROTOCOL.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Feature and method scope](docs/scientific-methods/)
- [Architecture and developer guide](docs/developer-guide/architecture.md)
- [Scaling measurements and limits](docs/validation/scaling.md)
