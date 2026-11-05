> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 4A: Generic Project Specification

The legacy CLI names its inputs `class0_top`, `class1_top`, `class0_traj`, and `class1_traj`. That layout is retained only as a historical compatibility path. New work should use a JSON project specification.

The generic project format supports any number of systems, ensembles, and replicas. It uses neutral scientific metadata; no field is intrinsically an ML label.

## Start from the template

```bash
cp examples/generic_project.example.json my_project.json
```

Edit `my_project.json` to reference actual files. Relative paths are resolved relative to the JSON file, not the terminal's current directory.

## Validate a project

```bash
python -m molscope.project_spec --project my_project.json
```

## Build frame identity metadata

```bash
python -m molscope.project_spec \
  --project my_project.json \
  --write-manifest frame_manifest.csv
```

The manifest preserves `system_id`, `ensemble_id`, `replica_id`, `trajectory_id`, frame index, time, and source paths. This becomes the shared identity layer for feature extraction, quality assessment, and later ML validation.

## Simulation protocols

`simulation_protocol` is optional and supports a deliberately explicit OpenMM
production plan. It records force-field XML files, nonbonded and constraint
choices, temperature, friction, timestep, minimization/equilibration/production
steps, reporting cadence, platform, seed, pH, and heterogen handling.
MolScope does not infer that a protein force field or implicit solvent is
suitable for a ligand, membrane, nucleic acid, polymer, or arbitrary molecule.

Generate a reviewable plan; this command does **not** run molecular dynamics:

```bash
python -m molscope.generate_openmm_plan \
  --project my_project.json \
  --out-dir results/simulation_plan
```

The output directory contains `openmm_plan.json` and an executable
`run_openmm_plan.py`. Review the plan with `--dry-run`, then explicitly run its
launcher or `python -m molscope.openmm_plan --plan … --run`. `--run` is required
for execution; `--dry-run` lists and validates selected jobs without importing
OpenMM. Generated paths are rooted under the chosen output directory, so plan
generation never overwrites declared analysis inputs. Existing simulation
outputs are rejected at execution time rather than replaced.

## Analysis configuration

An optional project-level `analysis` object selects generic extractor settings
and plugin IDs reproducibly:

```json
"analysis": {
  "generic_feature_parameters": {"max_frames": 500, "frame_stride": 5},
  "plugins": ["protein.salt_bridges", "structural.sasa"]
}
```

`MDContactMLPipeline.load_project("project.json")` applies this configuration.
Unknown or inapplicable plugins still report explicit states; no unavailable
feature values are zero-filled.

## Predictive ML tasks

The target-free ML discovery workflow is always run by the configured pipeline.
`prediction_tasks` declares a target metadata key for the separate predictive
ML layer when a scientifically valid target exists. A task key must refer to a
scalar value in system, ensemble, or replica metadata. It becomes a manifest
column named `metadata_<target_metadata_key>`; the configured runner validates
that before fitting a model.

Metadata are preserved both in their unprefixed form (for a declared target)
and namespaced form (for provenance), for example `metadata_condition` and
`metadata_ensemble_condition`. Conflicting values using the same unprefixed key
across system, ensemble, and replica scopes are rejected rather than silently
choosing a label.
