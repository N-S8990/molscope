> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 16 — Protein salt-bridge geometry

`protein.salt_bridges` uses standard ASP/GLU oxygen atoms and LYS/ARG nitrogen
atoms. It reports the count of acidic-basic atom pairs within 4.0 Å, the
closest in-cutoff distance, and occupancy of any detected bridge.

This is an operational geometric descriptor, not a direct electrostatic energy
or a universal salt-bridge assignment. It excludes histidine because its charge
depends strongly on protonation and environment, uses no PBC correction in this
initial version, and is not applicable if the needed atom sets are absent.
