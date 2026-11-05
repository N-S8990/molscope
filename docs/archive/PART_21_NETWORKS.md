> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 21 — Dynamic protein contact networks

`networks.protein_contact_network` builds an unweighted graph for each frame:
protein residues are nodes and nonlocal heavy-atom contacts within 4.5 Å are
edges. It reports graph density, mean degree, exact betweenness and greedy
community count for systems of up to 400 residues, plus trajectory-level edge
persistence.

For larger systems, expensive exact metrics are explicitly marked
`skipped_by_cost_policy`; graph construction and persistence remain available.
The current implementation does not use PBC and contact-network topology is
not evidence of allosteric causality.
