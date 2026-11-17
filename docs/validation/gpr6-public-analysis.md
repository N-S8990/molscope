# GPR6 published-trajectory reanalysis: current evidence

The retained result is the
[63-feature first-window report](../../studies/gpr6/runs/public_12_replica_expanded_first_100ns/START_HERE.md)
and its matched
[second-window report](../../studies/gpr6/runs/public_12_replica_expanded_second_100ns/START_HERE.md).
Earlier pilot artifacts were intentionally removed during project cleanup.
The input and source-audit procedure is in
`studies/gpr6/analysis_protocol/README.md`; the primary sources are the
[no-added-inverse-agonist](https://zenodo.org/records/14060756) and
[3h-bound](https://zenodo.org/records/14060604) Zenodo records. This is a
MolScope analysis of the authors' simulations, **not new production MD** and
not a CVN424 comparison.

## Reproducible analysis window

Six independent replicas per condition contribute exactly frames 0–999 at
100 ps spacing (0–99.9 ns); each frame has the same ordered 4,273 receptor
atoms. Every source prefix was scanned independently, checked for timestamp,
finite coordinates, and adjacent receptor Cα continuity. Four prefixes were
assembled from separately audited HTTP byte ranges; the source full-file MD5
checks are pending. One previously resumed no-added-inverse-agonist replica 00
file had a byte mismatch against Zenodo and was excluded from analysis in
favor of a separately verified prefix. The result therefore remains
**preliminary** until all complete source files match author checksums.
For the other eight live-derived prefixes, 24 independent 1 MB Zenodo ranges
(beginning, middle, and end of each prefix) returned exact HTTP 206 ranges and
matched the saved bytes. Full source provenance and checksums are retained in
[the source-provenance record](gpr6-source-provenance.md).

## Main observed results

The three-fold, trajectory-grouped condition classifier achieved mean test
accuracy 0.7303 with logistic regression and 0.7097 with random forest. Each
fold held out two entire replicas per condition. In the random forest,
removing receptor RMSD to each replica's first frame lowered trajectory-mean
accuracy by 0.0362; removing receptor radius of gyration lowered it by 0.0266.
These are predictive associations, not causal residue importance.

An exploratory comparison of the **12 replica means** used all 924 possible
six-versus-six label assignments for two-sided exact permutation p values,
followed by Benjamini–Hochberg adjustment across all 39 measured features.
The PHE152–TRP292 Cα distance averaged 9.435 Å without added inverse agonist
and 10.363 Å with 3h (3h minus no-added difference +0.928 Å, p=0.00649,
q=0.0362). Nine features had q<0.05; they are strongly correlated distances,
so this count is not nine independent molecular mechanisms. Phe152 and
Trp292 match the primary paper's 3.36 and 6.48 numbering; this Cα distance
alone does not validate a microswitch mechanism. See the
[residue map](../scientific-methods/gpr6-residue-map.md). Seventeen of the 18
declared 8 Å contact indicators (among 39 features) were constant across all
12 replicas and provide no condition
information.

The local report includes a correlation map, 2D and 3D PCA occupancy plots,
all 39 feature definitions and condition comparisons, trajectory-level fold
assignments, accuracy and feature-ablation figures, candidate-state tables,
and machine-readable provenance. PCA occupancy is in learned feature space,
not physical three-dimensional receptor density.

## Publication gates still open

1. Complete all 12 source downloads and verify their author MD5 hashes;
   replace the corrupt resumed file with the clean download.
2. Extend beyond the two matched 100 ns windows, assess time convergence,
   and reserve independent validation trajectories or a distinct system.
3. Verify imaging across all replicas and test mapped pocket, TM6, DRY,
   NPxxY, and sodium observables with reviewed selections.
4. Prepare a complete parameterized receptor–membrane–ligand system before
   claiming any new MD, and independently validate any mechanistic or
   therapeutic claim.

## Expanded panel and temporal repeat

The 63-feature panel contains three global receptor measures, 25 curated
Cα distances, 32 sine/cosine components of 16 sidechain torsions, and three
DSSP measures. All 63 columns vary and are finite across 12,000 frames in
each window. Residue identities and motif numbers were checked against the
[primary-paper residue map](../scientific-methods/gpr6-residue-map.md).
The panel was designed after inspecting the pilot and the paper, so its
first-window advantage is **exploratory model selection**, not an unbiased
test. A separate 18-feature heavy-atom chemistry panel and an 81-feature
combination were also evaluated with the same folds. Mean held-out
trajectory accuracies are:

| Panel | 0–99.9 ns logistic | 0–99.9 ns forest | 100–199.9 ns logistic | 100–199.9 ns forest |
| --- | ---: | ---: | ---: | ---: |
| 39-feature pilot | 0.7303 | 0.7097 | 0.8480 | 0.8010 |
| 18-feature chemistry | 0.7638 | 0.7629 | 0.8233 | 0.8990 |
| 63-feature motif | 0.8563 | 0.8643 | 0.8926 | 0.9614 |
| 81-feature combination | 0.8644 | 0.8710 | 0.9018 | 0.9678 |

The 81-feature model improves the forest by only 0.68 and 0.63 percentage
points over 63 in the two windows. The 63-feature panel is a practical
current choice because it is smaller and physically interpretable. These
windows are **consecutive segments of the same 12 replicas** with identical
trajectory-held-out folds, so they test temporal stability and do not
provide independent-replica or external validation. The
[panel comparison table and accuracy figure](../../studies/gpr6/runs/panel_window_comparison/START_HERE.md)
show all models; fold-level scores are in each run.

The exact 924-assignment, replica-mean comparisons were repeated with BH
adjustment over all 63 features in each window. Condition associations for
Phe152–Phe295 Cα distance (+1.18 and +1.56 Å; q=0.0455 and 0.0341),
Phe152 χ2 sine (+0.741 and +0.949; q=0.0455 and 0.0341), and helix fraction
(−0.0178 and −0.0215; q=0.0455 and 0.0496) have the same directions and
q<0.05 in both windows. Phe152–Trp292 Cα distance is +0.928 and +1.412 Å;
its q is 0.0584 in the first window and 0.0341 in the second. These are
correlated, exploratory condition associations, not independent mechanisms.

Predictive ablation is less stable. Removing all Phe152 sidechain torsion
components lowered forest trajectory accuracy by **4.67 percentage points**
in the first window but **0.38 points** in the second. Single-column
Phe152–Phe295 Cα ablation changed from +1.70 points to −0.28 points.
Redundant features can trade predictive credit; no one residue can be
declared the definitive classifier driver. The complete 63-row
[feature stability table](../../studies/gpr6/runs/gpr6_cross_window/gpr6_expanded_feature_stability.csv)
places each condition effect, q value, and single-feature forest accuracy
drop side by side. The companion
[40-group table](../../studies/gpr6/runs/gpr6_cross_window/gpr6_expanded_group_stability.csv)
does the same for residue-level torsion groups. Their generator is tracked as
`studies/gpr6/analysis_protocol/compare_windows.py`.

The second window used frames 1000–1999 (100.0–199.9 ns) from the 12 source
XTCs, including the clean no-added-ligand replica 00. Its source-byte
region passed 36 independently fetched 1 MB HTTP 206 range comparisons
against Zenodo. Complete author MD5 checks remain pending; see the
[source provenance](gpr6-source-provenance.md). The same receptor-only
projection means these analyses do not measure ligand occupancy, sodium
occupancy, membrane interactions, or atomistic mechanism.

The source protocol and per-replica analysis scripts are tracked while
trajectory and report artifacts remain local study outputs. The repository
suite passed **145 tests and three subtests** in the project environment on
2026-11-07 after these panel additions.
