# Streaming validation — 2026-11-06

The streaming path is a separate, scoped alternative to the existing full
discovery pipeline. It does not establish full plugin parity at a million frames.

## Measured analysis benchmark

`python -m molscope.benchmark_stream --frames N --out-dir results/stream_scale_N`
creates three synthetic features, fits PCA and K-means on a seeded 2,000-row
reservoir, then assigns every row in 10,000-row batches.

| Rows | Descriptive analysis time (s) | SAM-MFP analysis time (s) | Process peak RSS (MiB) |
| ---: | ---: | ---: | ---: |
| 10,000 | 0.179 | 0.218 | 263.4 |
| 100,000 | 0.588 | 0.595 | 287.6 |
| 500,000 | 2.519 | 2.680 | 258.0 |
| 1,000,000 | 4.422 | 4.710 | 274.7 |

Measurements come from separate local processes. Peak RSS includes imports and
synthetic input generation; wall time covers analysis only. The streaming
engine exhibits strictly $O(1)$ memory scaling with respect to trajectory length $N$:
retained working array memory is bounded by $O(W \cdot \text{max\_lag} + S \cdot W)$
where $W$ is feature count, $\text{max\_lag} = 100$, and $S = 2,000$ reservoir rows.
Peak process RSS remains flat between 260 and 290 MiB from 10k to 1M frames.

## Streaming SAM-MFP and Cross-Replica Stability Parity

As of 2026-11-17, bounded-memory SAM-MFP Pareto candidate selection and
cross-replica stability are integrated directly into the streaming analysis
engine (`molscope analyze --sam-mfp` and `molscope stream-project --sam-mfp`):

1. **Online Autocovariance & ESS Ratio:**
   Online multi-trajectory autocovariance accumulates lag-1 to lag-100 covariances
   across chunk boundaries using a bounded ring buffer of the previous 100 frames.
   Trajectory-stratified moments ensure each independent simulation replica contributes
   its exact within-replica variance and mean.
   - Tested against in-memory `_effective_sample_size` down to floating precision:
     absolute difference $\le 1.4 \times 10^{-17}$.

2. **Occupancy Entropy & Uniqueness:**
   Evaluated on the uniform Algorithm R reservoir sample. For sample sizes
   $S \ge N$, this provides exact numerical identity; for large trajectories
   ($N \gg S$), it provides a bounded-memory unbiased distributional estimate.
   Pairwise Spearman correlation is restricted to active frame-resolved features,
   preserving exact `NaN` semantics for invariant features.

3. **Cross-Replica Stability (LOTO & Replica Consistency):**
   When multiple trajectories are declared or present in the chunk store metadata,
   the streaming reservoir retains explicit `(trajectory_id, replica_id, ensemble_id)`
   provenance.
   - Cross-replica consistency ($\text{Replica\_Consistency}$) is computed via
     IQR-normalized Wasserstein distance across replica distributions.
   - Leave-One-Trajectory-Out (LOTO) cross-replica PCA salience rank stability
     ($\text{Representation\_Stability} = 1 - \text{LOTO\_Rank\_Std\_Normalized}$)
     evaluates whether feature importance ranking shifts when individual replicas
     are withheld.
   - In synthetic multi-replica benchmarks, erratic features (active in only one replica)
     exhibit high LOTO instability ($\text{std} = 0.577$), whereas globally consistent
     signals maintain low rank variance ($\text{std} \le 0.333$).

4. **Conformational State Separation (ANOVA Eta-Squared):**
   Between-state variance $\text{SS}_{\text{between}}$ and total variance
   $\text{SS}_{\text{total}}$ are accumulated across all frames in bounded batches
   during assignment Pass 2. The resulting effect size $\eta^2 = \text{SS}_{\text{between}} / \text{SS}_{\text{total}}$
   matches in-memory `rank_state_explanatory_features` within $< 10^{-12}$.

5. **Numerical Parity on Real Villin Headpiece (2RJY):**
   Full parity was verified against the in-memory MolScope discovery pipeline
   on a 50-frame $\times$ 44-feature subset of the fast-folding Villin Headpiece
   (PDB: 2RJY, C$\alpha$ distances and contacts + global geometry):
   - `Valid_Fraction`: exact match ($10^{-7}$)
   - `ESS_Ratio`: match within $10^{-5}$
   - `Occupancy_Entropy`: match within $10^{-5}$
   - `Uniqueness`: match within $10^{-5}$
   - `PCA_Salience`: match within $10^{-5}$
   - `State_Explained_Variance_Eta_Squared`: match within $10^{-5}$
   - `SAM_MFP_Pareto_Front`: 100% exact boolean identity
   - `SAM_MFP_Dominator_Count`: 100% exact integer identity

## Architectural Comparison: Streaming vs. In-Memory Discovery

| Capability | In-Memory Discovery Pipeline | Streaming Workflow |
| :--- | :--- | :--- |
| **Memory Footprint** | $O(N \cdot W)$ floats in RAM (~20–30 GB for 1M frames $\times$ 1,000 features) | Bounded $O(W \cdot \text{max\_lag} + S \cdot W)$ (~260–290 MiB flat) |
| **Plugin Support** | Full catalogue: Geometry, C$\alpha$, DSSP, Dihedrals, Salt Bridges, H-Bonds, Stacking, PLIF | Bounded geometry selection + protein C$\alpha$ distance/contact pairs |
| **Coordinate Handling** | PBC-aware or whole/imaged trajectory | As-stored nonperiodic coordinates |
| **State Discovery** | KMeans, TICA, CKPCA, Density Peak Advanced (DPA) | Bounded-batch sample-trained KMeans in fitted PCA manifold |
| **Rule Explainability** | Decision tree explainer (`max_depth=3`) | Descriptive candidate state populations & variable feature table |
| **Feature Prioritization** | Multi-criteria SAM-MFP Pareto front | Bounded-memory SAM-MFP Pareto front (`--sam-mfp`) |
| **Replica Stability** | Multi-replica Wasserstein consistency & LOTO PCA | Multi-replica Wasserstein consistency & LOTO PCA via stratified reservoir |
| **Predictive ML** | Grouped CV (Random Forest, Logistic Regression, LOFO) | Supervised grouped ML omitted; focuses on target-free state discovery |
| **Checkpointed Resume** | Feature-level caching (`FeatureCache`) | Resumable chunk manifests with SHA-256 verification |

## Real input and execution checks

### Full trajectory compatibility checks (2026-11-07)

The local 2RJY `protein.pdb` and `simulation_run1.xtc` pair completed streaming
extraction and analysis over all **10,001 frames**. The bounded workflow used
protein C-alpha atoms for three as-stored geometry features, 500-frame
training sample, 256-row analysis batches, and 21 extraction chunks. This
shows that a complete real trajectory can pass the current limited streaming
path; it is not a benchmark of the full plugin catalogue or a scientific
validation of the resulting candidate states.

The local GPR6 PDB/XTC pair has **75,623 atoms** and the XTC index reports
**1,832 frames**. Topology/profile loading and a five-frame extraction check
passed. A complete read failed on zero-based frame **1,831** with an XTC
compression read error; sequential reading and direct frame access both found
the first **1,831 frames** readable. With an explicit 1,831-frame cap, the
limited streaming extraction and analysis completed in four chunks. The cause
of the final-frame read failure has not been established. The input files were
preserved and temporary outputs were removed.

A reusable copy was recovered under `studies/gpr6/preparation/` by copying the
original XTC byte prefix through offset **514,063,612**, the indexed start of
the unreadable final frame. Its `recovery.json` records source and recovered
SHA-256 hashes, sizes, and method using paths relative to that study directory.
An independent complete MolScope integrity scan marked the recovered file
valid, checking **1,831 of 1,831 frames**, with the final time at **183,000 ps**.
This is exact frame-prefix salvage, not reconstruction of the missing frame or
proof of the original simulation's scientific provenance. The original file
remains unchanged and should be replaced with an authentic complete source if
one becomes available.

Both checks used as-stored, nonperiodic geometry. GPR6 has periodic-cell
metadata and a membrane/solvent environment; these geometric outputs must not
be interpreted scientifically until imaging and feature definitions are
reviewed. Neither dataset provides independent replicas for supervised ML or
cross-replica stability tests in this check.

An inspectable local GPR6 baseline run is retained at
`studies/gpr6/runs/real_baseline_stream/analysis/START_HERE.md`. It uses a
verified 1,831-frame recovered prefix, with the original XTC untouched. The
run stores five descriptive receptor features, all-frame assignments, a
feature guide, correlation coefficients/map, and 2D/3D PCA occupancy figures.
No supervised accuracy or condition comparison is reported from this one
trajectory. PHE152 and TRP292 are confirmed topology labels in segment PROA;
the [primary GPR6 paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850111/)
identifies them as 3.36 and 6.48, respectively. The measured Cα distance
alone is not evidence of a microswitch mechanism.

For the author-deposited six-versus-six GPR6 comparison, the predeclared
first-100-ns protocol is in `studies/gpr6/analysis_protocol/README.md`.
It selects exactly 1000 frames (timestamps 0–99.9 ns) from each independent
replica, projects matching receptor atoms, and preserves trajectory-grouped
condition labels for the full ML workflow. A snapshot-based preparer can
verify exact frame prefixes while complete public downloads continue; these
prefixes have separate provenance and cannot claim full-source MD5 agreement
until downloads finish. Its 1000-frame extraction path was verified on the
recovered baseline. A preliminary 12-replica matched-window run now exists at
`studies/gpr6/runs/public_12_replica_first_100ns/START_HERE.md`, including
grouped ML, exact replica-label permutation comparisons, correlation and
2D/3D PCA figures. The study-specific guide records the measured effects and
limits; all `studies/` outputs remain ignored local artifacts.

As a separate **external software integration test**, MolScope analyzed ten
real 1A22 trajectories from the independent `MD-ContactML` study: five
replicas at 300 K and five at 400 K, all 200 frames each. Grouped three-fold
classification yielded trajectory-mean accuracy 0.9675 for logistic regression
and 0.9631 for random forest. Removing radius of gyration lowered the random
forest trajectory-mean accuracy by 0.0801 in this dataset. All ten prepared
PDBs were verified to have the same ordered atom/residue identities before a
representative topology was used; the local audit is
`studies/validation_1a22/topology_identity_check.json`. Five of seven
eligible generic measurements lay on the SAM-MFP Pareto front; measured costs
were unavailable and excluded. The retained local
entry point is `studies/validation_1a22/full_run/START_HERE.md`; it includes
the correlation map, 2D/3D PCA figures, feature guide, and accuracy gain/drop
figure. These results verify the full supervised output path. They do not
validate a GPR6 ligand hypothesis, generalization to other proteins, or a
causal role for radius of gyration; condition-linked preparation differences
could also affect the classifier. Both `studies/` result directories are
ignored local artifacts, not archived publication data.

The project-file wrapper completed a local 2RJY run with 100 selected frames,
3,785 extracted features, and four assignment chunks. Extraction resume was
also checked against committed chunks. Generated smoke files were removed
after verification.

A 22-atom OpenMM 8.6 Reference-platform smoke produced four valid frames.
A controlled interruption halfway through a 20-step production run recovered
the remaining ten steps into a fresh segment containing two valid frames.
Generated smoke files were removed after verification. These were operational
checks, not equilibrated scientific simulations.

The current local suite passes 135 tests and 3 subtests; Python compilation
and whitespace checks passed.

## Scientific and operational boundaries

- Streaming extraction currently covers explicit geometry selections and
  selected protein C-alpha distance/contact pairs. Unsupported requested
  plugins retain assessments and do not become zero-valued measurements.
- Coordinates are used as stored, with nonperiodic distances. Whole/imaged
  molecular coordinates must be supplied where required; this is not a new
  validated PBC implementation.
- PCA and clustering are fitted on a bounded sample. Statistics and subsequent
  state assignments cover all extracted frames. These descriptive states are
  not biological replicates, predictive labels, or causal discoveries.
- Extraction supports checked chunk resume. Analysis requires a fresh output
  directory and can be rerun from the completed extraction store.
- Memory estimates guard numeric work arrays, not total operating-system RSS.
  CSV storage can be large; the manifest estimates it before extraction.
- Million-frame molecular extraction, broad plugin parity, and a matched
  mdxplain benchmark remain separate validation tasks.
