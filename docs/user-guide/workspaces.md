# Organizing a study

Keep source code and scientific data separate. A study workspace contains its
declaration, immutable inputs, preparation records, simulation segments, and runs:

```bash
python -m molscope init /path/to/my-study --project-id my_study
```

Initialization requires a new directory. Edit its project declaration and supply
or reference real inputs before running an analysis.

```text
my-study/
├── project.json
├── inputs/
├── preparation/
├── simulations/
│   └── replica-001/
├── runs/
│   └── run-001/
└── cache/
```

Use a new output directory for each analysis run. Keep configuration, source
fingerprints, applicability assessments, and completion markers with the results.
Never infer success merely because an output directory exists.

The managed batch command creates a named run with configuration and provenance:

```bash
python -m molscope run --study /path/to/my-study --run-id run-001
```

Use `--extract-only` for extraction without full discovery. The managed runner
groups artifacts into stage directories and snapshots the resolved project and
input fingerprints. Existing `--project`/`--out-dir` commands retain their native
artifact layouts for compatibility.

Managed outputs include `resolved_project.json`, `provenance.json`,
`artifacts.json`, and `logs/run.log`. Produced artifacts appear in `features/`,
`validation/`, `quality/`, `discovery/`, `predictive/`, `prioritization/`, or
`report/` as appropriate. Only stages with artifacts receive directories.
The hidden `.native/` directory contains compatibility links to these artifacts;
use it with an existing session reader that expects the original flat layout.
Report-local links preserve relative table and image references. Failed runs
retain their log and failure status; choose a new run ID for a retry.

In this checkout, `studies/` is an ignored local convenience location. For
ongoing research, initialize a workspace outside the source repository.
Historical `results/` outputs remain in place to preserve embedded absolute
paths. Do not relocate checkpoint bundles without checking their provenance.

Inputs can remain in their original storage location and be referenced from
`project.json`; do not copy large trajectories just to match a directory tree.
Retain preprocessing policies and original input identities. Simulation restart
segments belong to their original replica and must not be counted as independent
replicas.

The full and streaming workflows currently have different artifact schemas.
Preserve their native files and provenance when archiving existing runs.
A common study directory does not imply a merged or interchangeable feature store.
Only requested stages should produce outputs; caches are reproducible and disposable.

Portable examples should use relative paths. Machine-specific configurations and
large trajectory files belong in local study workspaces and should be excluded
from source control.
