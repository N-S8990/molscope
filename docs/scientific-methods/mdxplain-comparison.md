# mdxplain comparison and research boundary

This is a design comparison, not a claim that MolScope supersedes
mdxplain. It was prepared from the local mdxplain checkout and the published
mdxplain article cited below.

## What mdxplain already provides

mdxplain presents a unified `PipelineManager` for trajectory loading,
featurization, feature selection, dimensionality reduction, clustering,
decision-tree explanation, plots, and archival/reuse. Its local checkout has
dedicated modules for trajectory handling, features, feature selection,
decomposition, clustering, comparison, feature importance, plots, and 3D
visualization. Supported built-in feature families include contacts, distances,
coordinates, DSSP, SASA, and torsions; its API also exposes several reduction
and clustering choices. The published paper describes large-trajectory memory
handling, metadata-based selection, visualization, and ML feature selection.

Reference: Salomon M, Pankonin M, Butler MB, et al. *mdxplain: Scalable
molecular dynamics analysis with machine learning-based feature selection and
modular workflows*. Journal of Molecular Biology. 2026;438(18):169809.
doi:10.1016/j.jmb.2026.169809.

## Honest capability comparison

| Area | mdxplain | MolScope now | MolScope required next |
| --- | --- | --- | --- |
| User flow | Mature unified manager and tutorials | One configured project runner plus lower-level CLIs | Consolidate more analysis choices behind registries |
| Feature breadth | Contacts, distances, coordinates, DSSP, SASA, torsions and related tooling | Generic geometry/dynamics plus initial interaction/solvation/network plugins | Broaden and validate molecular-class feature families |
| ML | Decomposition, clustering, decision-tree feature selection/explanation | Target-free PCA salience, clustering, SAM-MFP, grouped supervised workflow | More validated representations/models and comparative baselines |
| Scale/reuse | Dask/Zarr-oriented architecture, caching/archive support | Sessions/cache plus a scoped chunked CSV workflow with bounded-sample state discovery | Broader streaming plugin parity and matched real-trajectory scaling validation |
| Visualization | Broad plots plus PyMOL/NGLView support | Markdown/JSON/HTML reports, descriptive plots, and extraction heatmaps | Broader interactive state/trajectory plots and 3D/export views |
| Molecular scope | Primarily biomolecular MD analysis API | Architecture records general components, but only some initial plugins are validated | Cross-system validation before claiming broad support |
| Core research question | Explain trajectory patterns and features that distinguish clusters | Prioritize applicable, reliable, nonredundant, reproducible, cost-aware measurements | Demonstrate this methodology against baselines and mdxplain |

## Defensible research direction

The potential contribution is **not** automatic feature extraction, clustering,
or ML-based feature selection. mdxplain already demonstrates those areas.
MolScope must test whether explicit system applicability, measurement
quality, replica reproducibility, uncertainty, redundancy/complementarity, and
cost-aware Pareto prioritization improve the choice of molecular analyses. That
is the provisional SAM-MFP hypothesis, not an established novelty claim.

## Consequence for implementation

MolScope should adopt mdxplain's strengths in clean public workflow,
documentation, scalable storage, visual outputs, and analysis breadth where
scientifically justified. It should not duplicate mdxplain's decision-tree
workflow and call that novelty. Any publication comparison must run both tools
on matched data and report failure cases as well as successes.
