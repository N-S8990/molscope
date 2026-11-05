# Feature library and validation scope

The first MolScope library deliberately covers a limited, transparent set
of feature families rather than claiming universal analysis support.
Implementation and reference coverage differ by method; consult the scientific
registry and [reference-validation evidence](../validation/reference-validation.md)
before treating a family as validated for a new system.

| Family | Current implemented features | Key limits |
| --- | --- | --- |
| Structural | geometric radius of gyration, asphericity, aligned RMSD, global RMSF | scope and reference dependent |
| Geometric | component center/minimum distances, contact count/fraction | PBC only when explicitly enabled and validated |
| Interactions | generic contacts, protein-ligand residue contacts, salt-bridge geometry, hydrophobic packing, directional H-bonds, PHE/TYR aromatic stacking | cutoff/policy based; not energies or causal evidence |
| Solvation/electrostatics | SASA, water-shell occupancy, neutral-scope molecular dipole | optional backend/partial charges required; not free energies; initial plugins are not PBC-aware |
| Networks | protein residue-contact graph metrics and persistence | contact topology is not evidence of allostery or causality |
| Nucleic acid | recognized base-center distances and proximity occupancy | geometric proximity is not base-pairing or stacking classification |
| Dynamics | RMSF, Rg fluctuation, lag-one autocorrelation | sampling descriptors, not kinetic observables |
| Conformational | PCA representation, candidate clusters, transition counts | ML candidates, not validated metastable states |

Hydrogen bonds require suitable donor/acceptor and hydrogen/topology input;
SASA requires its optional validated backend. Neither is approximated when
prerequisites are absent. Membrane/lipid order, detailed nucleic base-pairing
and stacking, broader ligand chemistry, and polymer descriptors remain future
capability-gated extensions.
