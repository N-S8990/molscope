# MolScope current status

Last updated: 2026-11-07

## Current GPR6 result

The retained result is the published-trajectory reanalysis of GPR6 with no
added inverse agonist versus 3h-bound receptor. It uses six independent
replicas per condition and two consecutive 100 ns windows (0–99.9 ns and
100–199.9 ns), with trajectory-grouped three-fold validation.

The current practical analysis panel has **63 receptor features**:

- 3 global receptor shape/motion measures;
- 25 paper-mapped C-alpha distances;
- 32 sin/cos components for 16 selected side-chain torsions;
- 3 secondary-structure fractions.

The 63-feature panel is retained because it nearly matches the larger
81-feature panel while using 18 fewer measurements. Mean held-out
trajectory accuracy was 0.856/0.864 (logistic regression/random forest) in
the first window and 0.893/0.961 in the second. The consecutive windows use
the same replicas, so they test temporal stability rather than independent
validation.

The analysis includes complete per-frame feature tables, per-replica means,
condition comparisons, correlation maps, PCA occupancy plots, grouped ML
accuracy and ablation tables, temporal profiles, and offline interactive and
static receptor heatmaps.

## Read these first

1. [Main first-window report](studies/gpr6/runs/public_12_replica_expanded_first_100ns/START_HERE.md)
2. [Second-window report](studies/gpr6/runs/public_12_replica_expanded_second_100ns/START_HERE.md)
3. [Feature-panel comparison](studies/gpr6/runs/panel_window_comparison/START_HERE.md)
4. [Interactive receptor feature map](studies/gpr6/runs/public_12_replica_expanded_first_100ns/gpr6_receptor_predictive_map.html)
5. [Scientific interpretation and limits](docs/validation/gpr6-public-analysis.md)
6. [Feature coverage and remaining gaps](docs/validation/gpr6-feature-coverage.md)

## What the result supports

The selected receptor measurements distinguish the two author-deposited
conditions within the analyzed trajectories. Several mapped structural
associations repeat directionally over both windows. Feature ablation credit
changes between windows because correlated measurements can substitute for
one another; no individual residue is established as a mechanism.

## What remains before publication claims

- Re-download or retain the full author trajectories and verify every full
  source MD5 checksum. The local bulk trajectory copies were removed during
  cleanup; source records and checksums remain in
  [provenance](docs/validation/gpr6-source-provenance.md).
- Validate on independent trajectories or a separately prepared system.
- Use the full imaged systems for ligand pose/contact, water and sodium
  occupancy, and membrane-lipid analyses. The retained classifier uses
  receptor-only coordinates and does not establish those effects.
- Treat condition associations and residue heatmaps as hypotheses until
  independent validation and mechanistic checks are complete.

## Reproducibility

Tracked protocol scripts live in
[`studies/gpr6/analysis_protocol`](studies/gpr6/analysis_protocol/README.md).
The source study, residue mapping, and exact selection logic are documented
in [study design](docs/scientific-methods/gpr6-study-design.md) and the
[GPR6 residue map](docs/scientific-methods/gpr6-residue-map.md).

Repository verification after the final feature work: **145 tests and 3
subtests passed**. `git diff --check` passed.
