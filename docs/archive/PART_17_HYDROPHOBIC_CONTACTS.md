> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 17 — Protein hydrophobic-contact geometry

`protein.hydrophobic_contacts` counts nonlocal carbon-carbon contacts among
ALA, VAL, ILE, LEU, MET, PHE, TRP, TYR, and PRO side chains within 4.5 Å.
Same-residue contacts and same-chain pairs separated by fewer than three
residues are excluded. It also reports the closest retained contact and the
occupancy of any retained contact.

This is a cutoff-based geometric proxy for packing. It is not a hydrophobic
free-energy estimate, excludes PBC in the initial implementation, and depends
on the residue and sequence-separation policy.
