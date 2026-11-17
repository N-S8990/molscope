# Running a project

Install the package in your scientific environment from the repository root:

```bash
python -m pip install --no-deps --no-build-isolation -e .
python -m molscope --help
```

The environment must already supply the scientific dependencies when using
`--no-deps`. See [environment setup](environment.md).

For a managed study, initialize a fresh workspace, edit its `project.json`, and
run `python -m molscope run --study /path/to/study --run-id run-001`. See
[workspace organization](workspaces.md) for the directory and provenance contract.

Existing module commands remain supported:

```bash
python -m molscope.project_spec --project project.json
python -m molscope.run_project --project project.json --out-dir /path/to/study/runs/run-001
```

The full workflow validates inputs, profiles systems, extracts applicable
features, computes quality and target-free discovery evidence, applies SAM-MFP,
and writes reports. Predictive ML runs automatically when there is one declared
prediction task, or one complete categorical trajectory label and no task.
With multiple possible targets, select a declared task using `--prediction-task`.
MolScope does not invent experimental labels from trajectory coordinates.

Declare an externally assigned condition in `project.json` with a stable label
ID and one value for each complete trajectory ID:

```json
"labels": [{
  "label_id": "condition",
  "label_type": "categorical",
  "level": "trajectory",
  "source": "external_annotation",
  "assignments": {
    "system_a:control:replica_01": "control",
    "system_a:treated:replica_01": "treated"
  }
}]
```

The example shows the schema, not enough independent replicas for predictive
validation. With the default three folds, each class needs at least three
independent trajectories. Frame labels and labels derived from MolScope's own
clusters cannot be used as supervised targets. A task may name a label with
`target_label_id`; the older `target_metadata_key` form remains supported.
Predictive results report trajectory-grouped model metrics and the change in
accuracy when a feature group is removed. These are predictive associations,
not causal explanations.

For one trajectory, `--extract-only` generates extraction outputs and its
report. Full discovery needs independent trajectories for stability checks;
do not duplicate inputs or relabel restart segments as replicas.

For large inputs, use the [streaming workflow](scalable-runs.md). Its supported
features and analysis stages differ from the full workflow. Do not assume
switching execution modes preserves all scientific outputs.

Full-run outputs include quality, PCA, state, and SAM-MFP tables; supervised
tables are added when a valid target is available. Reports and figures are
descriptive evidence. A configured analysis needs a consistent feature schema
across its trajectories; analyze incompatible systems separately.

Open `START_HERE.md` in the run results first. It links the report, the complete
`feature_guide.csv`, relevant numerical tables, and every figure produced by
that run. The correlation map shows at most 30 features in PCA-salience order;
the full feature list remains in CSV. The 2D and 3D occupancy figures use PCA
coordinates, not physical atom positions. When supervised labels are available,
the accuracy figure compares the grouped random-forest accuracy with all
features against accuracy after each feature group is removed. These changes
are predictive associations, not molecular or statistical proof of significance.
