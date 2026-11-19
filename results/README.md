# MolScope Results Directory

This directory serves as the default output workspace for MolScope interactive studies, tutorials, benchmarks, and production pipeline runs.

---

## Directory Organization

| Directory | Generating Workflow | Description & Key Artifacts |
| :--- | :--- | :--- |
| `2rjy_interpretability/` | `examples/visualize_2rjy_interpretability.ipynb` | Consolidated interactive HTML report (`report.html`), state FES landscapes, and 3D PyMOL (`.pml`) / ChimeraX (`.cxc`) visualization scripts. |
| `2rjy_matched_benchmark/` | `benchmarks/run_2rjy_matched_benchmark.ipynb` | Runtime and memory profiling benchmarks comparing MolScope extraction against standard MD analysis toolkits. |
| `2rjy_tutorial/` | `docs/tutorials/2rjy_villin_tutorial.ipynb` | Canonical HP35 Villin Headpiece tutorial run outputs: 2D Free Energy Surfaces (`free_energy/`), decision trees, and 3D B-factor annotations. |
| `notebook_features/` | `notebooks/04_Multi_Family_Biophysical_Featurization.ipynb` | Multi-family biophysical extraction results (Cα distances, DSSP secondary structure, circular sidechain dihedrals, salt bridges, SASA). |
| `quickstart_2rjy/` | `notebooks/01_Quickstart_Villin_Headpiece.ipynb` | End-to-end quickstart study run: TICA kinetic coordinates, DPA macrostates, and conformational rule extractions. |
| `sam_mfp_study/` | `benchmarks/sam_mfp/run_benchmarks.ipynb` | SAM-MFP Pareto-optimal feature selection benchmarks, objective ablation summaries, and Variance Fallacy demonstrations. |
| `simulation_demo/` | `notebooks/02_Simulation_to_Discovery_Pipeline.ipynb` | Declarative OpenMM simulation plans (`openmm_plan.json`), execution provenance logs, and dry-run manifests. |
| `streaming_demo/` | `notebooks/06_Large_Scale_Streaming_Analysis.ipynb` | Out-of-core streaming feature store, chunk manifests (`manifest.json`), and incremental PCA trajectory coordinates. |
| `workspace_demo/` | `notebooks/08_Configurable_Projects_and_Study_Workspaces.ipynb` | Structured multi-run study workspaces, validated `project.json` specifications, and audited run provenance. |

---

## Output Retention & Version Control Policy

- **Intermediate Trajectories & Parquet Stores**: Large binary trajectories (`*.xtc`, `*.dcd`, `*.trr`) and cached Parquet feature stores are excluded from version control via `.gitignore`.
- **Lightweight Metrics & Reports**: Interactive HTML dashboards, high-resolution figures, and summary CSV tables can be exported to documentation or retained for publication supplementary materials.
- **Reproducibility**: Any deleted result directory can be deterministically re-generated at any time by executing its corresponding Jupyter notebook.
