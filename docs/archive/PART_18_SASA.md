> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 18 — Solvent-accessible surface area

`structural.sasa` uses the optional FreeSASA backend with a 1.4 Å probe radius
and 100-point Shrake-Rupley sampling. It reports total SASA and per-residue
SASA for non-solvent/non-ion atoms when those components are detected.

The FreeSASA version, probe radius, atom scope, and sampling settings are
recorded in provenance. SASA depends on atomic radii, structure preparation,
and scope. It is not a solvation free energy. PBC-aware SASA is not implemented
in this initial plugin; trajectories must be appropriately imaged first.
