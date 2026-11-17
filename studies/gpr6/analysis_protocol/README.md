# GPR6 author-deposited 12-replica protocol

This local protocol prepares a **published-data reanalysis** of six no-added-
inverse-agonist GPR6 replicas and six **3h-bound** replicas. The 3h condition is
the author's IAG Set 3. It is not CVN424. Provenance and author MD5 values are
in [`docs/validation/gpr6-source-provenance.md`](../../../docs/validation/gpr6-source-provenance.md).

## Fixed comparison before reading outcomes

- Use exactly the first 1000 zero-based frames, 0 through 999, from every
  replica. The deposited stride is 100 ps, yielding 100 ns of sampled frames
  with timestamps 0–99.9 ns. The preparation script verifies frames 0, 1,
  and 999 have times 0, 100, and 99900 ps before writing.
- Project only the ordered 4273 protein atoms. Both deposited PDBs have the
  same ordered protein atom names, residue names/IDs, and segments. Keep the
  no-ligand and 3h source PDB/XTC files unchanged.
- Use the three generic receptor shape/displacement features and 18 Cα
  pairs (distance and 8 Å contact) among topology residue IDs 150–153 and
  290–293, with sequence separation ≥3. The primary GPR6 paper identifies
  local Phe152 as 3.36 and Trp292 as 6.48; see the
  [verified residue map](../../../docs/scientific-methods/gpr6-residue-map.md).
  This feature
  subset was chosen after inspecting the earlier single-trajectory baseline;
  it is not a preregistered or externally validated biological panel.
- Assign one external categorical condition label to each independent
  trajectory. Fit and test logistic regression and random forest with grouped
  three-fold splits by trajectory. Ablation is per measured feature. Frames
  from one replica cannot appear in both train and test.
- Individual Cα distances and contact indicators are correlated, and many
  pairs share atoms. Removing one feature while retaining correlated partners
  can understate its predictive contribution. Ablation estimates only how the
  fitted model's held-out accuracy changes for this measured set; it is not a
  significance test or evidence that a residue causes the condition difference.
- Treat all geometry as stored and nonperiodic. This is a screening analysis
  until protein imaging/whole-molecule integrity and feature interpretation
  are validated. A condition classifier is predictive association, not a
  ligand mechanism or a new MD simulation.

The recovered no-ligand baseline was checked across all 1831 readable frames:
no adjacent Cα distance exceeded 4.123 Å, but 66 frames had a receptor Cα
jump above 10 Å from the previous frame (maximum 86.96 Å). This is consistent
with a whole receptor translated by periodic wrapping rather than a split
backbone in that one replica. The prefix preparer independently rejects any
frame with adjacent Cα distance above 5 Å; all 12 condition prefixes must
pass the same check. The baseline summary is in `baseline_continuity.json`.

## Source layout and execution

The source files are:

```text
studies/gpr6/datasets/no_ligand/topology.pdb
studies/gpr6/datasets/no_ligand/replica_00.xtc ... replica_05.xtc
studies/gpr6/datasets/iag/topology.pdb
studies/gpr6/datasets/iag/replica_00.xtc ... replica_05.xtc
```

For an initial matched-window analysis while the complete Zenodo downloads
continue, run from the repository root:

```bash
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/prepare_windows.py --from-growing-downloads
```

This waits for enough bytes from each source, takes a fixed-size snapshot of
the growing file, and reads **only that snapshot** with MDAnalysis. It copies
the exact XTC byte prefix through the start of frame 1000, then independently
checks all 1000 frames of that prefix. The discarded snapshot never becomes
an analysis input. `source_prefixes/provenance.json` records byte counts and
prefix MD5 hashes. These are **not full author MD5 verifications**; results
must be labeled preliminary until the complete downloads match the author's
checksums.

After **all 12 downloads stop**, the default command verifies their complete
author MD5 hashes, then regenerates the same fixed-window derived inputs:

```bash
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/prepare_windows.py
PYTHONPATH=src /Users/nirav/ds-env/bin/python -m molscope run \
  --project studies/gpr6/analysis_protocol/protein_first_100ns_1000_frames/project.json \
  --out-dir studies/gpr6/runs/public_12_replica_first_100ns \
  --components 3 --states 2 --supervised-folds 3
```

The default mode rejects any missing or incomplete source against the
author MD5s before creating derived files. It writes a provenance manifest,
a common protein PDB, 12 protein-only XTCs, and a MolScope project manifest.
The run should produce `START_HERE.md`, the correlation map, 2D/3D PCA plots,
feature guide, grouped-ML accuracy and ablation figure, and machine-readable
tables. Those outputs must be inspected before scientific claims are made.

Run `PYTHONPATH=src /Users/nirav/ds-env/bin/python
studies/gpr6/analysis_protocol/compare_replica_means.py` after MolScope. It
re-extracts the same 39 features, saves the complete 12,000-frame feature
matrix and one mean per independent trajectory,
computes all 924 possible assignments of six trajectories to either label,
and applies Benjamini–Hochberg correction across 39 exact two-sided tests.
It also writes a condition-effect PNG. These tests are exploratory; contact
indicators and distances sharing atoms are correlated.

The derived local XTCs and run outputs are ignored study artifacts. Archive
the project manifest, code revision, source hashes, and result tables/figures
separately for a publication. Check the source trajectory's frame time and
identity before citing an effective simulated duration.

## Expanded exploratory motif panel

The first run's 39 features included 17 contact indicators that were constant
across all 12 replicas. A second, explicitly **post-pilot exploratory** panel
uses the same 1000-frame receptor-only trajectories and has 63 frame features:
three generic shape/displacement features, 25 selected Cα distances, 32
sin/cos components of 16 side-chain χ torsions, and three aggregate DSSP
secondary-structure fractions. It uses no 8 Å contact indicators. The Cα
pairs cover the paper's Phe152–Trp292 toggle, Val156–Val237–Phe288 core,
Asp118–Ser155–Asn321 sodium-site geometry, Arg166–Thr274 opening distance,
DRY/NPxxY, and selected pocket geometry. These are geometry proxies; the
receptor-only inputs cannot measure actual ligand, sodium, water, or lipid
occupancy. The assignments follow the [primary GPR6 structural study](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850111/)
and were verified against the deposited topology's residue identities.

Create and run the second project from the repository root:

```bash
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/prepare_expanded_project.py
PYTHONPATH=src /Users/nirav/ds-env/bin/python -m molscope run \
  --project studies/gpr6/analysis_protocol/protein_first_100ns_1000_frames/project_expanded.json \
  --out-dir studies/gpr6/runs/public_12_replica_expanded_first_100ns \
  --components 3 --states 2 --supervised-folds 3
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/compare_replica_means.py \
  --project studies/gpr6/analysis_protocol/protein_first_100ns_1000_frames/project_expanded.json \
  --run studies/gpr6/runs/public_12_replica_expanded_first_100ns \
  --panel-label 'GPR6 expanded first 100 ns'
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/group_sidechain_ablation.py
```

The expanded panel was chosen after the original run's outcome was inspected.
Use its grouped scores and feature associations to form hypotheses, then test
them on independent trajectories and longer time windows. The first 39-feature
run remains intact for direct comparison.

The same 63-feature panel can be checked for temporal stability in frames
1000–1999 (100.0–199.9 ns). `prepare_second_window.py` copies a fixed 650 MB
snapshot from each growing source, verifies the first-window prefix hash,
saves and scans a byte-exact 2000-frame prefix, then projects only the second
1000 frames. The no-added-inverse-agonist replica 00 source is the separate
clean replacement. The [independent range audit](../preparation/second_window_range_audit.json)
checks 36 exact Zenodo byte ranges across the 100–200 ns source region.
This is a **same-replica temporal stability check**, not an independent
validation set; complete author MD5 verification remains pending.

```bash
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/prepare_second_window.py
PYTHONPATH=src /Users/nirav/ds-env/bin/python -m molscope run \
  --project studies/gpr6/analysis_protocol/protein_second_100ns_1000_frames/project_expanded.json \
  --out-dir studies/gpr6/runs/public_12_replica_expanded_second_100ns \
  --components 3 --states 2 --supervised-folds 3
PYTHONPATH=src /Users/nirav/ds-env/bin/python -m molscope run \
  --project studies/gpr6/analysis_protocol/protein_second_100ns_1000_frames/project_baseline.json \
  --out-dir studies/gpr6/runs/public_12_replica_baseline_second_100ns \
  --components 3 --states 2 --supervised-folds 3
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/compare_replica_means.py \
  --project studies/gpr6/analysis_protocol/protein_second_100ns_1000_frames/project_expanded.json \
  --run studies/gpr6/runs/public_12_replica_expanded_second_100ns \
  --panel-label 'GPR6 expanded second 100 ns'
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/compare_replica_means.py \
  --project studies/gpr6/analysis_protocol/protein_second_100ns_1000_frames/project_baseline.json \
  --run studies/gpr6/runs/public_12_replica_baseline_second_100ns \
  --panel-label 'GPR6 39-feature second 100 ns'
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/group_sidechain_ablation.py \
  --run studies/gpr6/runs/public_12_replica_expanded_second_100ns
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/plot_temporal_profiles.py
PYTHONPATH=src /Users/nirav/ds-env/bin/python studies/gpr6/analysis_protocol/compare_windows.py
```

## Extraction path check

On 2026-11-07, the script's `--baseline-test` mode projected the recovered
readable no-ligand prefix to a real 1000-frame, 0–99.9 ns protein-only XTC.
The resulting protein PDB/XTC loaded with 4273 atoms. MolScope's generic
extractor returned 1000 rows and three finite frame features; the Cα plugin
returned 1000 rows and 36 feature columns from 18 pairs. A separate snapshot
test produced and independently scanned an exact 1000-frame XTC byte prefix
(280,631,580 bytes) from the recovered baseline, then removed that temporary
prefix. This verifies
feasibility of the declared feature schema on one condition. It is not an
independent replica or a supervised validation result.

## Preliminary matched-window result

The 12-replica run completed locally at
[`runs/public_12_replica_first_100ns/START_HERE.md`](../runs/public_12_replica_first_100ns/START_HERE.md).
It contains 12,000 frames, 39 declared features, a correlation map, 2D and 3D
PCA occupancy plots, a full feature guide, and grouped-ML accuracy/ablation
figures and tables. Three-fold trajectory-grouped accuracy was 0.7303 for
logistic regression and 0.7097 for random forest. Removing receptor RMSD to
its first frame lowered random-forest held-out accuracy by 0.0362; removing
radius of gyration lowered it by 0.0266.

Exact trajectory-label permutation tests on six replica means per condition
gave PHE152–TRP292 Cα distance 9.435 Å versus 10.363 Å (3h minus no-added
inverse agonist +0.928 Å; p=0.00649, BH q=0.0362 across 39 features). Nine
correlated distance/RMSD features had exploratory q<0.05. The 17 features
tagged `Resolution=trajectory` in the quality table are 8 Å contacts that
never change anywhere in the data (15 always 0, two always 1); they add no
condition information. The remaining 22 features vary. Neither the q values
nor the ML accuracy changes establish a ligand mechanism.

All 12 frame-prefix files passed independent 1000-frame scans and receptor
Cα continuity checks. Four were assembled from audited Zenodo HTTP ranges;
the no-ligand replica-0 live resumed file had a byte mismatch and was never
used for analysis. Full author-file MD5 verification remains pending for all
12 XTCs. This result is preliminary, represents only the first 100 ns, and
does not claim the full published trajectories were analyzed. The receptor
atom projection excludes ligand, membrane, lipids, and solvent features;
stored nonperiodic coordinates need further imaging review before molecular
mechanism claims. Publication archiving and independent experimental
validation remain separate work.
