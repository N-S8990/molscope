# Run against your own trajectory

Initialize a separate study workspace and edit `project.json` to reference a
topology and its matching trajectory:

```bash
python -m molscope init /path/to/study --project-id my_study
```

For a first smoke test, configure a small frame cap or stride, then run:

```bash
python -m molscope run --study /path/to/study --run-id extraction-001 --extract-only
```

This validates and extracts supported features, records applicability evidence,
and creates an extraction report. Check the run provenance, feature table,
missingness, and report before scaling up.

Full representation-stability analysis requires at least two actual independent
trajectories. Do not split or duplicate one trajectory and call the pieces
independent. Generated smoke datasets and reports were removed; compact
molecular fixtures needed by tests remain in `tests/fixtures/`.
