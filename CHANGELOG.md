# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2026-11-17

### Added
- **Interactive Study API**: Introduced `molscope.Study` for fluent trajectory analysis, filtering, and visualization.
- **TICA Kinetics**: Trajectory-isolated time-lagged independent component analysis with implied timescales (`molscope.analysis.discovery.tica`).
- **Contact Kernel PCA (CKPCA)**: Non-linear contact similarity representation (`molscope.analysis.discovery.contact_kpca`).
- **Density Peak Advanced (DPA) Clustering**: Topographic cluster discovery from representation coordinates (`molscope.analysis.discovery.density_peaks`).
- **Free Energy Surfaces (FES)**: 2D potential of mean force computation with automated local basin identification, publication static contours, and interactive Plotly HTML (`molscope.reporting.free_energy`).
- **Decision Tree State Explainer**: Interpretable shallow boolean if/then rule extraction, Gini feature importance, and tree visualization (`molscope.analysis.discovery.decision_tree`).
- **Protein-Ligand Interaction Fingerprints (PLIF)**: Domain plugin measuring hydrophobic contacts, hydrogen bonds, $\pi$-$\pi$ aromatic stacking, and salt bridges (`molscope.features.ligand.plif`).
- **Automated 3D Visualization**: B-factor scored PDB generation and automated PyMOL (`.pml`) and ChimeraX (`.cxc`) script rendering (`molscope.reporting.pymol`).
- **On-the-fly PBC Coordinate Transformations**: Unwrapping, centering, and wrapping powered by MDAnalysis transformations (`molscope.io.transformations`).
- **Comprehensive HTML Dashboard**: Single self-contained interactive report (`report.html`) summarizing metrics, plots, and tables.
- **Project Schema Versioning**: JSON Schema declaration (`schemas/project.schema.json`) and schema validation tests.

### Changed
- Standardized directory layout into clean subdirectories (`tables/`, `figures/`, `3d_visualization/`, `audit/`).
- Removed legacy flat CLI wrappers directly under `src/molscope/*.py` in favor of modular subpackages.
- Updated requirements and dependencies for Python 3.10 through 3.13.

## [0.1.0] - 2026-11-06

### Added
- Initial public release of MolScope.
- Core pipeline architecture: `MDContactMLPipeline`.
- SAM-MFP Pareto multi-feature prioritization.
- Capability-gated domain feature plugins (C-alpha contacts, backbone torsions, DSSP, SASA).
- System profiling and topology validation.
- Grouped cross-validation predictive modeling.
