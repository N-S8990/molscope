# MolScope OpenMM Simulation Pipeline: Planning, Execution, and Recovery

## 1. Architectural Philosophy: Two-Phase Simulation

MolScope separates simulation **planning** from simulation **execution**:
1. **Plan Generation (`build_openmm_plan` / `write_openmm_plan`)**:
   - Strictly deterministic, non-invasive, and fast.
   - Evaluates system topology, calculates file hashes, validates simulation parameters, and generates an `openmm_plan.json` plan.
   - **Never imports heavy optional runtime libraries** (OpenMM, PDBFixer) during plan generation.
   - Computes expected trajectory frames and approximate uncompressed DCD byte sizes so researchers can review storage commitments before spending compute resources.
2. **Explicit Execution Authorization (`execute_openmm_plan` / `molscope-openmm-plan --run`)**:
   - Refuses to execute without explicit user confirmation (`--run` flag).
   - Can be previewed anytime with `--dry-run`, checking file accessibility and blocking overwrite collisions without writing a single byte.

---

## 2. Checkpoint Recovery and Replica Preservation

Long molecular dynamics simulations frequently terminate prematurely due to hardware faults, cluster wall-time limits, or transient node failures.

MolScope provides **immutable segment-based checkpoint recovery**:
- When a job is interrupted, its partially written outputs and `checkpoint.chk` (accompanied by `execution_provenance.json`) form a verifiable recovery anchor.
- Resuming via `--resume-checkpoint /path/to/checkpoint.chk`:
  1. Validates that the checkpoint belongs to the declared plan and verifies SHA-256 integrity of the input topology, protocol, and serialized system XML.
  2. Allocates a new immutable segment under `segments/<uuid>/` rather than mutating or truncating existing files.
  3. Seamlessly loads OpenMM state velocities, positions, and box vectors via `simulation.loadCheckpoint()`.
  4. Preserves the exact parent replica identity (`pep_sys:ens_0:rep_0`), recording provenance linking the segment back to its originating checkpoint.

---

## 3. Scientific Responsibilities: Force Field and System Preparation

> [!IMPORTANT]
> **Force-field and chemistry choices are the researcher's responsibility.**
> MolScope automates reproducible pipeline execution, but does not substitute for chemical domain knowledge.

Researchers must ensure:
1. **Force Field Selection**: Standard default parameters use `amber14-all.xml` with implicit solvent `implicit/gbn2.xml` for peptide prototypes. Explicit solvent simulations requires explicit water models (`tip3p.xml`, `tip4pew.xml`, etc.) and appropriately sized periodic boxes.
2. **Protonation States and pH**: PDBFixer automatically adds missing hydrogens at declared pH (default 7.0). Non-standard ionization (e.g., protonated histidines, catalytic dyad pKa shifts) should be reviewed prior to simulation.
3. **Periodic Boundary Conditions (PBC)**: If simulating with cutoffs (`CutoffNonPeriodic` or `PME`), ensure adequate padding buffer ($\ge 1.0\text{ nm}$) to prevent periodic self-interactions.
