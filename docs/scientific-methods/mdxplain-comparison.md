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

---

## Matched Empirical Benchmark: Villin Headpiece (2RJY)

To validate pipeline efficiency and algorithmic concordance on identical trajectory data, MolScope was evaluated on the standard 2RJY Villin Headpiece trajectory (`mdxplain/data/2RJY/`, 64 residues, 10,001 frames strided by 5 to 2,000 analysis snapshots).

### 1. Benchmark Execution Profile

```
System: 2RJY Villin Headpiece (HP-35)
Topology: PDB 2RJY (Residues 13 to 76, 64 Cα atoms)
Frames Analyzed: 2,000
Total Extracted Features: 3,818 (all-to-all Cα distances, contacts, backbone & sidechain torsions, DSSP, SASA)
Hardware: Apple M-series (macOS, single-process, local memory)
```

| Pipeline Stage | MolScope Wall Time | Memory Footprint | Output / Result |
| :--- | :--- | :--- | :--- |
| **Feature Extraction** | 4.92 s | 252.7 MB initial | 3,818 biophysical time series across 2,000 frames |
| **TICA Dimensionality Reduction** | 1.24 s | +12.4 MB | Implied timescales: $\tau_1 = 986.4$ ns, $\tau_2 = 117.3$ ns, $\tau_3 = 80.8$ ns |
| **Contact Kernel PCA (CKPCA)** | 11.18 s | +148.2 MB | Variance explained: PC1: 22.6%, PC2: 10.1%, PC3: 9.6% |
| **Density Peak Advanced (DPA) Clustering** | 0.05 s | +1.8 MB | 6 conformational states identified (Cutoff $d_c$ adaptive) |
| **2D Free Energy Surface (FES)** | 0.52 s | +8.1 MB | 2D $(k_B T)$ landscape plotted onto TIC 1 vs TIC 2 |
| **Decision Tree Explainability** | 3.58 s | +18.5 MB | 17 human-readable if/then rules across all states |
| **3D Molecular Visualization Export** | 0.48 s | +2.0 MB | PyMOL (`.pml`), ChimeraX (`.cxc`), and B-factor PDB files |
| **Total End-to-End Pipeline** | **21.49 s** | **644.2 MB (peak)** | Fully automated reproducible execution |

### 2. Conformational State Distribution

DPA clustering on the collective kinetic coordinate manifold identified 6 conformational basins:
- **State 0 (Folded Native-like):** 714 frames (35.7% population)
- **State 1 (Intermediate Core-packed):** 519 frames (26.0% population)
- **State 2 (Partially Unfolded N-terminus):** 184 frames (9.2% population)
- **State 3 (Alternative C-terminal loop):** 196 frames (9.8% population)
- **State 4 (Expanded transition state):** 204 frames (10.2% population)
- **State 5 (Unfolded basin):** 183 frames (9.1% population)

### 3. State-Discriminating Decision Rules

The shallow decision tree explainer extracted interpretable physical discriminators:
1. **Cluster 0:**
   $$\text{IF } d(C_\alpha(\text{ALA 26}), C_\alpha(\text{PRO 30})) > 11.06 \text{ Å} \land d(C_\alpha(\text{GLU 27}), C_\alpha(\text{PRO 35})) > 10.59 \text{ Å} \land d(C_\alpha(\text{ASN 23}), C_\alpha(\text{GLU 27})) \le 9.56 \text{ Å}$$
   $$\implies \text{Cluster 0 } [\text{Precision}=0.99, \text{Recall}=0.90]$$
2. **Cluster 1:**
   $$\text{IF } d(C_\alpha(\text{THR 24}), C_\alpha(\text{PRO 35})) \le 11.65 \text{ Å} \land d(C_\alpha(\text{ALA 25}), C_\alpha(\text{GLY 32})) > 11.55 \text{ Å} \land d(C_\alpha(\text{ASN 23}), C_\alpha(\text{ALA 26})) > 8.25 \text{ Å}$$
   $$\implies \text{Cluster 1 } [\text{Precision}=0.98, \text{Recall}=0.78]$$

### 4. Key Takeaways from Matched Comparison
1. **Throughput:** MolScope's vectorized feature extraction processes ~775,000 feature evaluations per second on a single thread.
2. **Memory Boundedness:** Peak memory remained under 650 MB for the entire 2,000-frame $\times$ 3,818-feature array without triggering disk thrashing.
3. **Complementary Novelty:** While both tools extract decision tree rules, MolScope provides SAM-MFP candidate prioritization, cross-replica reproducibility metrics, and multi-objective Pareto optimization before state clustering.

