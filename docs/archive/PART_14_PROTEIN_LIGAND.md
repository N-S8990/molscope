> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 14 — Protein–ligand contact plugin

`protein_ligand.contacts` applies only when the topology contains protein and
exactly one detected `small_molecule_or_polymer` component. It emits total
protein–ligand contact-pair counts, closest in-cutoff contact distance,
per-residue contact fractions, and per-residue occupancy.

The plugin uses a declared 4.5 Å all-atom geometric cutoff without PBC. It does
not identify chemical interaction types, binding free energy, or causality.
Systems with multiple ligand-like components are explicitly not applicable
until the user can declare the intended ligand scope.
