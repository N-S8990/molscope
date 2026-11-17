# GPR6 inverse-agonism study design

## Research question

Which reproducible GPR6 conformational measurements distinguish receptor
simulations with no added inverse agonist from simulations with inverse agonist
3h bound?
The comparison tests structural associations, not drug efficacy or causation.

## Primary conditions

| Condition | Target label | Preparation |
| --- | --- | --- |
| Same receptor and membrane, no added inverse agonist | `no_added_inverse_agonist` | Preserve and record pocket occupancy; do not call this a proven ligand-free state. |
| Same receptor and membrane, 3h bound | `3h_bound` | Use the author's published 3h-bound trajectories as the matched comparison; parameterize ligand if generating new runs. |

Use the same receptor sequence/construct, protonation policy, lipid composition,
water/ion model, temperature, pressure control, and simulation settings in both
conditions. The published pseudo-apo GPR6 structure (PDB 8TF5) contains an
unassigned lipid-like pocket density; absence of added ligand is not evidence
that the pocket is empty. The 3h complex is PDB 8T1V. The GPR6–Gs complex
(8TYW) is an active-state structural reference, not a matched receptor-only
simulation condition.

## Replica and analysis plan

The primary analysis uses the authors' six deposited production replicas per
condition, subject to checksum and trajectory validation. If generating new
simulations, prepare at least five independent production replicas per condition
with distinct initial velocities and recorded seeds. First run a shorter pilot
to check system stability and choose production length from convergence of the
predeclared receptor observables; do not count equilibration as production.
For supervised MolScope analysis, assign one condition label to each complete
replica. Keep all frames of a replica in one train/test fold. Reserve independent
replicas or an external system for final assessment. Do not infer labels from
MolScope's discovered clusters.

Predeclare pocket contacts, intracellular TM6 displacement/opening, and
Phe152³·³⁶, Trp292⁶·⁴⁸, DRY, NPxxY, and sodium-pocket observables. Record
atom selections, alignment, units, imaging policy, and expected direction
before comparing conditions. Report distribution and replica variability,
not only ML accuracy. After the primary comparison, a newly prepared
CVN424-bound condition (PDB 8T1W pose reference) can test whether findings
generalize to another inverse agonist.

## Current input and preparation gate

The local GPR6 PDB/XTC is an existing membrane trajectory, not a production
simulation system definition. It contains 75,623 atoms and DPPC, DOPC,
cholesterol-like CHL1, water, and ions, but no accompanying force-field topology
or parameter files. No CVN424 or 3h residue is present in the local no-ligand
PDB; a verified author-deposited 3h coordinate PDB is available separately. OpenMM currently exposes
Reference, CPU, and OpenCL platforms locally; no production throughput has
been established. The original local no-ligand XTC indexes 1,832 frames, but frame 1,831 (zero-based)
raises a compression read error. The first 1,831 frames are readable. Preserve
the source file; obtain the original trajectory or a verified repaired copy
before using it as complete study evidence.

Production runs require a reviewed receptor–membrane–ligand topology with
compatible parameters for every component, a preparation record, equilibration
checks, and a resource budget. The PDB and XTC alone cannot supply those.
MolScope's current streaming measurements use as-stored nonperiodic geometry;
receptor imaging and scientifically defined features must be addressed before
interpreting this membrane system.

## Sources

- Barekatain et al., *Science Signaling* 2024, DOI: 10.1126/scisignal.ado8741.
- PDB 8TF5 (pseudo-apo), 8T1W (CVN424), 8T1V (3h), 8TYW (GPR6–Gs).
- [Public replica provenance](../validation/gpr6-source-provenance.md).
- [Verified GPR6 residue map and candidate atom selections](gpr6-residue-map.md).
