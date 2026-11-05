> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 25: Configurable OpenMM Plan Generation

Release safeguards validate protocol values, selected-job output collisions,
input existence, and output containment before optional runtime imports. The
production interval must yield at least one saved frame. Declared seeds are
applied to both velocities and the Langevin integrator.

Execution writes periodic/final `checkpoint.chk`, preparation-policy provenance,
and `output_integrity.json`. Completion requires readable, finite trajectory
coordinates and nonempty trajectory, prepared topology, log, and checkpoint.
A per-job `analysis_project.json` points to the generated inputs; it represents
one trajectory and does not invent independent replicas. Combine multiple jobs
in a study declaration before interpreting replica-based evidence.

Interrupted production can be recovered explicitly into a fresh segment:

```bash
python -m molscope.openmm_plan --plan results/openmm_plan/openmm_plan.json --job system:ensemble:replica --resume-checkpoint results/openmm_plan/simulations/system/ensemble/replica/checkpoint.chk --dry-run
python -m molscope.openmm_plan --plan results/openmm_plan/openmm_plan.json --job system:ensemble:replica --resume-checkpoint results/openmm_plan/simulations/system/ensemble/replica/checkpoint.chk --run
```

Resume requires matching job/protocol provenance, SHA-256 identities of input
and prepared topology, the same OpenMM version/platform, and an intact saved
`system.xml`. Recovery deserializes this exact force-field system instead of
rebuilding atom parameters from a rounded PDB export. The OpenMM checkpoint loader also enforces its hardware
compatibility constraints. Old checkpoints without this provenance cannot be
resumed automatically. No preparation, minimization, velocity initialization, or
equilibration is repeated. Only the original remaining production steps run;
a finished checkpoint is rejected. Outputs go into `segments/<unique-id>/`,
with their own checkpoint, provenance, integrity report, and analysis project.
The original outputs remain intact. Further interruptions can recover from the
new segment's checkpoint. Provenance records the segment's absolute step range.
Interrupted files may contain frames beyond the saved checkpoint; segments
are **not** automatically concatenated and must not be treated as independent
replicas. Inspect step ranges before merging to avoid duplicate sampling.

Dry-run previews include duration, save interval, expected full-run frame count,
and approximate uncompressed DCD size based on input atom count. Preparation
changes atom counts, and these estimates exclude logs and checkpoints. Resume
dry runs check input identities; remaining steps and runtime compatibility are
validated only when the checkpoint is loaded. Tests exercise execution wiring
with mocks; real force-field suitability, hardware execution, and restart
reproducibility still require runtime and system-specific validation.

The new platform can generate a reproducible OpenMM execution plan from a
system-level `simulation_protocol` in `project.json`:

```bash
python -m molscope.generate_openmm_plan --project my_project.json --out-dir results/openmm_plan
```

Generation is intentionally separate from execution. It does not import
OpenMM, prepare a structure, or run dynamics. The resulting JSON records every
job identity, source structure, output locations, protocol values, seed, and
metadata. Each replica receives a deterministic seed offset from the declared
base seed.

Protocols can express production length as `production_duration_ns` and saved
frame spacing as `save_interval_ps`. These convert to integer integration steps
using `timestep_femtoseconds`; use either these duration fields or the equivalent
`production_steps`/`report_interval_steps`, avoiding conflicting declarations.
Choose save intervals for the molecular process being studied: saving a million
highly correlated frames does not create a million independent observations.

Review the plan with a dry run, then authorize execution explicitly:

```bash
python -m molscope.openmm_plan --plan results/openmm_plan/openmm_plan.json --dry-run
python -m molscope.openmm_plan --plan results/openmm_plan/openmm_plan.json --run
```

The generated launcher includes the same explicit `--run` authorization. The
executor imports OpenMM and PDBFixer lazily, rejects existing generated
outputs, writes `execution_provenance.json` after a completed or failed job,
and has no implicit retries or overwrite policy. Use a reviewed,
system-specific force field and validate the simulation protocol independently
before treating the resulting trajectories as scientific evidence.

The current generator accepts PDB inputs through PDBFixer. It is a platform
execution facility, not a claim that a default force field, implicit solvent,
or preparation procedure is generally appropriate.
