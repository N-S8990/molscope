# Command reference

Use `python -m molscope --help` for the unified command list and
`python -m molscope COMMAND --help` for the exact arguments. An installed package
also exposes the `molscope` console entry point.

| Command | Responsibility |
| --- | --- |
| `init` | Initialize a separate study workspace |
| `run` | Run a project via `--project`/`--out-dir`, or a managed study via `--study`/`--run-id` |
| `validate` | Validate molecular inputs |
| `profile` | Inspect system composition and capabilities |
| `plan` | Generate a reviewable OpenMM plan |
| `simulate` | Preview or explicitly execute an OpenMM plan |
| `extract` | Extract features through the documented extraction entry point |
| `analyze` | Analyze completed streaming feature chunks |
| `stream-project` | Run the limited streaming workflow from a project |
| `report` | Build a report from analysis artifacts |
| `benchmark` | Execute the configured matched benchmark harness |

Commands have different input contracts; they are not interchangeable stages
accepting arbitrary outputs. Use their help and the relevant guide. Simulation
execution requires `--run`; plan generation and dry runs do not run dynamics.

The existing module commands remain supported, including:

```bash
python -m molscope.run_project --help
python -m molscope.streaming --help
python -m molscope.stream_analysis --help
python -m molscope.stream_project --help
python -m molscope.generate_openmm_plan --help
python -m molscope.openmm_plan --help
python -m molscope.run_benchmark --help
```

The limited streaming workflow is documented in [scalable runs](scalable-runs.md).
For labels and predictive ranking, see the
[label/ranking reference](../archive/PART_26_LABELS_AND_RANKING.md).
