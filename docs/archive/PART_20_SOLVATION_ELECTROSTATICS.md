> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 20 — Solvation and electrostatics

`solvation.water_shells` uses topology-identified solvent oxygen atoms to
count waters within 3.5 Å of each molecular component. For multi-component
systems it also counts water oxygens shared between component shells as an
operational water-bridge measure.

`electrostatics.molecular_dipole` calculates dipole magnitude for a neutral
non-solvent/non-ion scope with finite supplied partial charges. It is not
applicable to non-neutral scopes because an absolute dipole is then
origin-dependent. Neither plugin currently applies PBC correction.
