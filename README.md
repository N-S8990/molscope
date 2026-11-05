# MolScope

MolScope is a Python platform for inspecting molecular dynamics (MD)
trajectories, extracting applicable molecular features, and comparing the
evidence those measurements provide. It combines explicit system profiling,
feature quality checks, target-free state discovery, and SAM-MFP Pareto
prioritization in a configurable project workflow.

```text
project.json → validate and profile → extract applicable features
             → assess quality and discover states
             → prioritize candidate measurements → report
```

Predictive models run only when a project declares a valid target. Their
rankings are reported separately from descriptive state explanations and
SAM-MFP's Pareto candidates. Unavailable features carry applicability status
and reasons; they are not silently filled with zeroes.

## What it can do

- Validate topology and trajectory inputs, preserve system, ensemble, and
  replica identities, and summarize system composition and capabilities.
- Extract generic geometry and dynamics features, plus supported plugins for
  protein distances and contacts, backbone torsions, DSSP, hydrogen bonds,
  aromatic stacking, salt bridges, hydrophobic contacts, SASA, solvation
  shells, molecular dipoles, and protein contact networks. A nucleic-acid
  base-center proximity descriptor is available; it is not a base-pairing
  classifier.
- Report feature quality and run target-free PCA and conformational-state
  discovery, then present SAM-MFP Pareto candidates.
- Optionally run grouped predictive ML and predictive feature rankings when
  researchers supply declared trajectory-level labels.
- Generate reviewed OpenMM plans and, with OpenMM/PDBFixer installed, execute
  simulations with checkpoints and recovery into recorded trajectory segments.
- Stream a limited feature subset to checksummed chunks, resume interrupted
  extraction, and analyze large frame collections in bounded batches.

Plugin applicability, execution limits, imaging assumptions, and current
validation evidence are described in the [test readiness guide](docs/validation/test-readiness.md).

## Install

MolScope requires Python 3.10 or newer. From a clone of this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

For the test and simulation extras:

```bash
python -m pip install -e '.[test,simulation]'
```

OpenMM plan generation does not require the simulation extra. Executing a plan
does. For Conda setup and platform-specific notes, see the
[environment guide](docs/user-guide/environment.md).

Check the installed commands with:

```bash
molscope --help
# Or from a source checkout:
python -m molscope --help
```

## Start a study

Initialize a workspace outside the source checkout, then edit its project file
to point to your input data:

```bash
molscope init /path/to/my-study --project-id my_study
```

The initializer creates `project.json` and directories for inputs, preparation,
simulations, runs, and cache. The starter project uses placeholder input paths;
replace them before running. Trajectory files can stay where they are and be
referenced by absolute or project-relative paths.

For example, a minimal project declaration looks like this:

```json
{
  "schema_version": "1",
  "project_id": "my_study",
  "systems": [{
    "system_id": "protein_a",
    "topology": "inputs/topology.pdb",
    "ensembles": [{
      "ensemble_id": "condition_a",
      "replicas": [{
        "replica_id": "replica_01",
        "trajectory": "inputs/replica_01.xtc"
      }]
    }]
  }],
  "analysis": {
    "generic_feature_parameters": {"frame_stride": 1},
    "plugins": ["protein.ca_distances_contacts"],
    "plugin_parameters": {
      "protein.ca_distances_contacts": {
        "selection": "protein and name CA",
        "contact_cutoff_angstrom": 8.0,
        "max_pairs": 5000
      }
    }
  }
}
```

Run extraction and its report, or run the full configured discovery workflow:

```bash
molscope run --study /path/to/my-study --run-id extraction-001 --extract-only
molscope run --study /path/to/my-study --run-id analysis-001
```

Each run records the resolved project, input fingerprints, log, completion
status, and artifacts. Outputs are grouped into validation, features, quality,
discovery, predictive, prioritization, and report directories as applicable.
Run IDs must be new; use a fresh ID to retry a failed run. See the
[project run guide](docs/user-guide/project-runs.md) and
[workspace guide](docs/user-guide/workspaces.md).

## Generate molecular dynamics

Simulation generation and analysis are separate steps. Put a reviewed
`simulation_protocol` on a system in `project.json`, then create a plan:

```bash
molscope plan --project /path/to/my-study/project.json \
  --out-dir /path/to/my-study/simulations/plan-001
molscope simulate --plan /path/to/my-study/simulations/plan-001/openmm_plan.json \
  --dry-run
```

Review the selected structure, force field, preparation policy, timestep,
production length, and expected output size. Execute only when ready:

```bash
molscope simulate --plan /path/to/my-study/simulations/plan-001/openmm_plan.json \
  --run
```

MolScope does not establish that a force field or preparation procedure is
appropriate for a particular molecule. Validate those choices for your study.
Recovery can resume an unfinished checkpoint into a new segment. Segments are
not automatically concatenated and must retain their parent replica identity.

## Large trajectories

The separate streaming workflow writes checksummed feature chunks and computes
descriptive PCA/clustering on a bounded sample before assigning frames in
batches. Current streaming extraction supports selected geometry and protein
C-alpha distance/contact features, with explicit as-stored nonperiodic
coordinates. It does not run the complete plugin catalog, predictive ML,
replica stability, or SAM-MFP.

The million-row synthetic benchmark measures the chunked analysis machinery,
not molecular trajectory extraction or a million-frame MD simulation. It is
not a universal capacity guarantee. Review the
[scalable workflow and limitations](docs/user-guide/scalable-runs.md) before
planning a large run.

## Tests and scientific validation

Run the suite from the repository root:

```bash
python -m pytest -q
```

Tests cover contracts, numerical behavior, integrations, and selected reference
comparisons. They do not validate every method for every molecular system.
Matched comparisons with mdxplain require aligned feature definitions,
preprocessing, frames, and resources. The current project has not established
that MolScope is superior to mdxplain. See the
[validation guide](docs/validation/test-readiness.md) and
[comparison notes](docs/scientific-methods/mdxplain-comparison.md).

## Project layout

```text
src/molscope/       Implementation organized by responsibility
tests/              Unit, integration, and reference tests with small fixtures
examples/           Portable project templates
benchmarks/         Benchmark definitions
docs/               User guides, scientific methods, validation, and developer notes
```

Research inputs, generated trajectories, caches, and run reports belong in a
study workspace rather than in the package source tree.

## Documentation

- [All documentation](docs/README.md)
- [Command reference](docs/user-guide/cli.md)
- [Feature and method scope](docs/scientific-methods/)
- [Architecture and contribution guide](docs/developer-guide/architecture.md)
- [Scaling measurements and limits](docs/validation/scaling.md)
