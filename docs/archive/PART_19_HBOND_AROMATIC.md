> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 19 — Hydrogen-bond and aromatic geometry

`protein.hydrogen_bonds` requires explicit hydrogens and donor-H bond topology.
It counts supported protein donor/acceptor geometries with a 3.5 Å donor-acceptor
cutoff and a 150° donor-H-acceptor angle threshold. It is not applicable when
those prerequisites are absent.

`protein.aromatic_stacking` identifies PHE/TYR ring pairs with 3.5–7.0 Å ring
center separation and plane angle no greater than 30°. It is a conservative
face-to-face geometric proxy, not a complete pi-interaction classifier.
