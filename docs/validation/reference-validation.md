# Reference-validation fixtures

`tests/test_reference_validation.py` holds matched-policy, cross-system
reference checks for selected capability-gated plugins. The fixtures cover a
protein–ligand/charged-protein system, a solvated protein, and a nucleic-acid
base-center system.

| Measurement | Reference calculation | Fixture | Validation scope |
| --- | --- | --- | --- |
| Protein–ligand atom-pair contacts | MDAnalysis `distance_array` | protein–ligand | 4.5 Å all-atom pairs, no PBC |
| Protein salt-bridge atom pairs | MDAnalysis `distance_array` | protein–ligand | declared ASP/GLU–LYS/ARG atom sets, 4.0 Å, no PBC |
| Water-shell occupancy | MDAnalysis `distance_array` | solvated protein | unique water oxygens within 3.5 Å, no PBC |
| Base-center proximity | direct center geometry | nucleic bases | recognized-atom arithmetic centers, 7.0 Å, no PBC |
| SASA wrapper | direct FreeSASA calculation | protein–ligand | identical atom scope, probe radius, and point count |

These are wrapper and convention checks, not a broad claim that the methods
are scientifically validated for every force field, topology, protonation
state, or system class. They intentionally do not validate PBC imaging,
chemical interaction identity, free energies, base pairing, or stacking.
Those remain explicit limitations in the scientific registry. Plugins not
listed above remain implementation-tested but not reference-validated.
