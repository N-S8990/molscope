> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 2: System-Aware Foundation

Part 2 adds a small `molscope` package alongside the legacy scripts. It does not replace `ml_pipeline.py` or alter simulation behavior.

## What is now available

- `SystemProfile`: reports broad topology components and capabilities without assuming chain names such as `A` or `B`.
- `TrajectoryInput` and `build_frame_manifest`: define explicit frame, trajectory, replica, and ensemble identity metadata.
- `AnalyzerSpec` and `AnalyzerCatalog`: state the capabilities an analysis family requires before any values are calculated.
- `FeatureAssessment`: returns `applicable` or `not_applicable` with a reason and missing capabilities. This prevents inapplicable analysis from becoming a numerical zero.
- `FeatureProvenance`: stores the analyzer version, parameters, sources, and measured runtime for future feature values.

## Try it

Profile only a topology:

```bash
python -m molscope.inspect_system --topology 1A22.pdb
```

Profile a topology and trajectory:

```bash
python -m molscope.inspect_system \
  --topology data/replica_1/temp_300K/sim_300K_prepared.pdb \
  --trajectory data/replica_1/temp_300K/sim_300K_traj.dcd
```

Write the report to JSON:

```bash
python -m molscope.inspect_system --topology 1A22.pdb --output system_profile.json
```

## Interpretation rules

- `applicable` means the declared topology requirements are present. It does not imply a calculation is already implemented or that sampling is adequate.
- `not_applicable` is a valid scientific result. For example, protein torsions should be unavailable for a polymer without protein backbone metadata.
- `periodic_cell` means a trajectory carries box metadata. It does **not** prove that PBC was physically used or that molecules are correctly imaged. PBC-sensitive analyses must validate that separately.
- Broad component labels intentionally err on the conservative side: unknown organic residues are classified as `small_molecule_or_polymer`, not guessed to be a particular chemistry.

## Deliberately deferred

This part does not yet calculate new molecular features, rank features, train ML models, or modify the legacy extractor. Those belong to later parts after the metadata and applicability contract is stable.
