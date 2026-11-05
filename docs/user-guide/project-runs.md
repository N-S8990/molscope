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
and writes reports. Optional predictive ML requires a declared task and an
explicit `--prediction-task`.

For one trajectory, `--extract-only` generates extraction outputs and its
report. Full discovery needs independent trajectories for stability checks;
do not duplicate inputs or relabel restart segments as replicas.

For large inputs, use the [streaming workflow](scalable-runs.md). Its supported
features and analysis stages differ from the full workflow. Do not assume
switching execution modes preserves all scientific outputs.

Details of the existing configured runner and labels are retained in the
[runner note](../archive/PART_24_PROJECT_RUNNER.md) and
[label/ranking note](../archive/PART_26_LABELS_AND_RANKING.md).
