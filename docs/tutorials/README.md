# MolScope Tutorials

This directory contains interactive tutorials demonstrating MolScope workflows on real-world biophysical benchmark systems.

## Available Tutorials

| Tutorial Notebook | System | Description |
| :--- | :--- | :--- |
| **[`2rjy_villin_tutorial.ipynb`](2rjy_villin_tutorial.ipynb)** | **HP35 Villin Headpiece (2RJY)** | Complete conformational discovery workflow using the high-level `Study` API: on-the-fly PBC imaging, Cα contact networks, DSSP secondary structure, sidechain rotamers, TICA, Contact KPCA, DPA state discovery, 2D Free Energy Surfaces, transparent Boolean rules, and 3D PyMOL / ChimeraX visualization exports. |

## Running Tutorials Locally

Launch the tutorial notebook directly in JupyterLab:

```bash
jupyter lab docs/tutorials/2rjy_villin_tutorial.ipynb
```

For the comprehensive suite of 8 topic-focused notebooks (covering OpenMM simulation, SAM-MFP Pareto optimization, multi-family featurization, physical correctness checks, out-of-core streaming, leakage-free ML validation, and study workspaces), see the top-level **[`notebooks/`](../../notebooks/)** directory.
