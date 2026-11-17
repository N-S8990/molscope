# Physical Correctness & Input Verification Guide

MolScope treats physical correctness and input integrity as foundational prerequisites. Downstream featurization, representation learning, and machine learning models are only as valid as the underlying coordinates and topologies ingested into the pipeline.

This guide details MolScope's physical correctness architecture:
1. **Periodic Boundary Condition (PBC) & Coordinate Wrapping Checks**
2. **Recommended Trajectory Imaging Fixes**
3. **Transparent Feature Applicability Propagation (Zero-Imputation Policy)**
4. **CLI & Python API Workflows**

---

## 1. Periodic Boundary Conditions (PBC) & Coordinate Wrapping

### The Physical Problem
Molecular dynamics engines (such as GROMACS, OpenMM, AMBER, NAMD, and CHARMM) simulate systems in periodic unit cells to eliminate boundary vacuum effects. During integration, coordinates are frequently "wrapped" into the primary unit cell (`[0, L]`) independently for each atom or solvent molecule.

When coordinates are wrapped without making macromolecular solutes whole:
- **Covalent bonds break across unit cell faces**: One bonded atom sits at $x \approx 0.5$ Å while its partner wraps to $x \approx L - 0.5$ Å.
- **Catastrophic measurement distortion**: A $1.4$ Å covalent bond appears as a $50$ Å or $100$ Å separation in raw Cartesian coordinates.
- **False dynamical transitions**: Protein domains or ligand complexes appear to explode or teleport across the simulation box, completely corrupting distance matrices, contact networks, radius of gyration ($R_g$), root-mean-square deviation (RMSD), and secondary structure calculations.

```
       Wrapped (Broken Molecule)                     Imaged (Whole Molecule)
+-----------------------------------+     +-----------------------------------+
|  (C)                              |     |                                   |
|   | (broken bond across box edge) |     |            (CA)-(C)-(N)           |
|                                   |     |              |                  |
|                               (CA)|     |                                   |
+-----------------------------------+     +-----------------------------------+
      Raw distance: ~48.5 Å                        Raw distance: ~1.5 Å
      PBC distance: ~1.5 Å                         PBC distance: ~1.5 Å
```

### Detection Architecture
MolScope performs automated, vectorized checks during `inspect_md_inputs(...)` and `profile_system(...)`:

1. **Explicit Covalent Bond Verification**:
   When topology connectivity is available (e.g. from PDB `CONECT`, PRMTOP, PSF, or TPR files), MolScope evaluates every covalent bond:
   $$\Delta \mathbf{r}_{\text{raw}} = \mathbf{r}_i - \mathbf{r}_j, \quad d_{\text{raw}} = \|\Delta \mathbf{r}_{\text{raw}}\|$$
   $$d_{\text{pbc}} = \text{calc\_bonds}(\mathbf{r}_i, \mathbf{r}_j, \text{box}=\mathbf{L})$$
   A wrapped bond is flagged when $d_{\text{raw}} > 3.0\text{ Å}$ and $d_{\text{pbc}} < 2.5\text{ Å}$.
   An unphysically stretched/broken bond is flagged when $d_{\text{raw}} > 3.0\text{ Å}$ and $d_{\text{pbc}} \ge 2.5\text{ Å}$.

2. **Protein Backbone Virtual Bond Continuity**:
   When explicit bonds are not defined in the source format, MolScope inspects sequential protein $C\alpha$ atoms within the same chain. Because peptide $C\alpha_i - C\alpha_{i+1}$ distances are strictly bounded to $2.9\text{--}4.0\text{ Å}$, a raw separation $d_{\text{raw}} > 5.0\text{ Å}$ with $d_{\text{pbc}} \le 4.5\text{ Å}$ uniquely identifies a backbone broken across periodic boundaries.

3. **Intra-Residue Coordinate Span**:
   For multi-atom residues, MolScope monitors bounding-box coordinate spans ($\max(\mathbf{r}) - \min(\mathbf{r})$). If the Cartesian span exceeds half the smallest unit-cell dimension ($\frac{1}{2}\min(L_x, L_y, L_z)$), the residue has been severed across periodic boundaries.

---

## 2. Recommended Imaging Fixes

When MolScope detects un-imaged or wrapped coordinates, it emits loud, actionable warnings detailing exact fix recipes:

### Option A: MolScope On-The-Fly Transformations (Python)
MolScope provides native, auditable MDAnalysis transformations via `ImagingPolicy`:

```python
from molscope.io.transformations import apply_imaging_transformations, ImagingPolicy
import MDAnalysis as mda

u = mda.Universe("topology.pdb", "trajectory.xtc")

# Apply unwrapping, solute centering, and solvent wrapping
policy = ImagingPolicy(
    unwrap=True,
    center_selection="protein",
    center_to="box",
    wrap=True,
    wrap_compound="fragments",
)
applied_transforms = apply_imaging_transformations(u, policy)
print(f"Applied on-the-fly transformations: {applied_transforms}")
```

### Option B: GROMACS (`gmx trjconv`)
For GROMACS users, preprocessing trajectories prior to analysis guarantees clean, whole coordinates:

```bash
# Step 1: Make molecules whole across periodic boundaries
gmx trjconv -s topol.tpr -f traj.xtc -o traj_whole.xtc -pbc whole

# Step 2: Center the macromolecule and compactly wrap solvent
gmx trjconv -s topol.tpr -f traj_whole.xtc -o traj_imaged.xtc -center -pbc mol -ur compact
```

### Option C: MDTraj / `mdconvert`
Using MDTraj CLI or Python API:

```bash
# CLI:
mdconvert traj.xtc -o traj_imaged.xtc -t topol.pdb
```

```python
# Python API:
import mdtraj as md
traj = md.load("traj.xtc", top="topol.pdb")
traj.image_molecules(inplace=True)
traj.save_xtc("traj_imaged.xtc")
```

---

## 3. Transparent Feature Propagation (No Silent Imputation)

### Scientific Rationale
In machine learning and statistics, missing data is often imputed using column means, medians, or zeros. **In physical chemistry, silent imputation is catastrophic**:
- Setting a missing distance or contact feature to `0.0` asserts that atoms are overlapping or forming intense physical interactions when no measurement existed.
- Filling mean values across heterogeneous conditions obscures real conformational transitions and creates synthetic data points that violate physical laws.

MolScope enforces a strict **Zero-Imputation Policy**:
> Missing, non-finite, or inapplicable features must propagate transparently with explicit statuses and human-readable audit reasons throughout all matrices, quality reports, and final rankings.

### Propagation Across Pipeline Stages

| Pipeline Stage | Output Artifact | Propagation Mechanism |
| :--- | :--- | :--- |
| **Applicability Contract** | `FeatureAssessment` | Flags status (`APPLICABLE`, `NOT_APPLICABLE`, `MISSING_INPUT`, `UNSUPPORTED`) before values are computed. |
| **Feature Extraction** | `extraction_evidence.json` | Records all assessments and missing capabilities per trajectory replica. |
| **Quality Assessment** | `feature_quality_metrics.csv` | Explicit `Eligibility_Status` (`eligible`, `unavailable`, `insufficient_valid_frames`, `trajectory_level`, `missing_diagnostics`) and `Ineligibility_Reason`. |
| **State Explanatory Ranking** | `state_explanatory_feature_ranking.csv` | Inapplicable features receive `Rank = NaN`, `State_Explained_Variance_Eta_Squared = NaN`, `Ranking_Status = "inapplicable"`, and explicit `Status_Reason`. |
| **SAM-MFP Prioritization** | `pareto_feature_candidates.csv` | Explicit `SAM_MFP_Status` (`pareto_candidate`, `dominated`, `ineligible`) and `SAM_MFP_Reason` explaining exact exclusion criteria. |
| **Predictive ML LOFO** | `predictive_feature_ranking.csv` | Non-finite features excluded from training without imputation; appended with `Ranking_Status = "inapplicable"`. |

---

## 4. Workflows & Command-Line Interface

### Preflight Verification via CLI
Audit topology and trajectory compatibility before initiating heavy analysis runs:

```bash
# Standard read-only inspection (emits loud warnings for wrapped coordinates):
molscope-validate-inputs \
  --topology system/topology.pdb \
  --trajectory system/trajectory.xtc

# Strict enforcement (exits with code 2 if coordinates are wrapped or broken):
molscope-validate-inputs \
  --topology system/topology.pdb \
  --trajectory system/trajectory.xtc \
  --require-imaged \
  --output inspection_report.json
```

### Python API Integration
```python
from molscope.io.validation import inspect_md_inputs, InputIntegrityStatus
from molscope.profiling.system import profile_system

# 1. Trajectory Integrity Scan
report = inspect_md_inputs(
    topology="topology.pdb",
    trajectory="trajectory.xtc",
    frame_stride=10,
    check_pbc_wrapping=True,
    require_imaged=False, # Set True for fatal errors
)

if report.status == InputIntegrityStatus.VALID:
    print("Coordinates are finite, compatible, and properly imaged.")
elif report.status == InputIntegrityStatus.VALID_WITH_WARNINGS:
    print("Warnings detected:")
    for warning in report.warnings:
        print(f"  - {warning}")
else:
    print("Integrity check failed:")
    for error in report.errors:
        print(f"  - {error}")

# 2. System Profile with PBC Inspection
profile = profile_system("topology.pdb", "trajectory.xtc")
if profile.wrapped_coordinates_detected:
    print("Warning: Trajectory has wrapped coordinates requiring imaging transformations.")
```
