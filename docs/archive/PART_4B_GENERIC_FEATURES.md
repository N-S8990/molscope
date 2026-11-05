> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 4B: Generic Feature Families

This step adds a conservative set of real calculators that do not assume a protein, chain name, residue numbering, ligand identity, or two predefined partners.

## Current feature families

- global geometric radius of gyration and asphericity;
- Kabsch-aligned RMSD from the first selected frame;
- center-of-geometry distances between every pair of detected non-environment components;
- minimum interatomic distances, contact-pair counts, and contact-pair fractions between detected molecular components;
- trajectory-level contact occupancy and longest consecutive contact-run fraction;
- trajectory-level global RMSF mean and maximum.
- trajectory-level mean and maximum RMSF for each selected residue.
- radius-of-gyration fluctuation and lag-one autocorrelation diagnostics.

By default, global shape, aligned displacement, and RMSF use non-environment
molecular components. Solvent and ions can be included explicitly with
`--include-environment-in-global-features` when that scope is scientifically
intended.

Frame and trajectory quantities are written to different CSV files. The new extractor does not repeat trajectory-level RMSF values for every frame.

Contact occupancy means the fraction of extracted frames with at least one atom
pair within the configured cutoff. Longest-run fraction is a descriptive
sampling statistic, not a contact lifetime or kinetic rate.

Lag-one autocorrelation depends on the extraction stride and is a sampling
descriptor, not a relaxation time or kinetic observable.

Residue RMSF is calculated after the same declared alignment as global RMSF.
It is trajectory-level; independent-replica uncertainty must be assessed from
multiple trajectories rather than by treating individual frames as replicates.

## Run

```bash
python -m molscope.extract_generic_features \
  --topology path/to/topology.pdb \
  --trajectory path/to/trajectory.dcd \
  --out-dir generic_features
```

For a short smoke test:

```bash
python -m molscope.extract_generic_features \
  --topology data/replica_1/temp_300K/sim_300K_prepared.pdb \
  --trajectory data/replica_1/temp_300K/sim_300K_traj.dcd \
  --max-frames 2 \
  --out-dir results/generic_feature_smoke_test
```

## PBC warning

`--use-pbc` is opt-in. Positive box dimensions do not prove that the simulation used valid periodic boundary conditions or that molecules are correctly imaged. Verify those conditions before enabling it.

When PBC is enabled, the generic component center distance, minimum atom
distance, and contact calculations use the trajectory cell's minimum-image
convention. The extractor rejects a PBC request when no positive trajectory
cell is available; it cannot validate imaging or equilibration automatically.

## Deliberately deferred

Protein sidechain chemistry, nucleic base pairing, membrane order, ligand pharmacophore features, and polymer-specific descriptors remain separate capability-gated plugins. They are not approximated by this generic extractor.
