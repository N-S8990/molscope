# GPR6 public trajectory provenance

## Matched real-data comparison

The primary immediately testable comparison is **GPR6 with no added inverse
agonist versus GPR6 with inverse agonist 3h**. The trajectory author deposited
six replicas of each condition in [Zenodo no-ligand record](https://zenodo.org/records/14060756)
(DOI 10.5281/zenodo.14060756) and [Zenodo IAG record](https://zenodo.org/records/14060604)
(DOI 10.5281/zenodo.14060604). Both records cite Barekatain et al., *Science
Signaling* 2024, DOI 10.1126/scisignal.ado8741, and state a strided sampling
interval of 0.1 ns per frame. These are **published source trajectories**, not
new MolScope simulations.

The paper's protocol reports GROMACS 2020.3 with CHARMM36, wild-type GPR6
restoration and ICL3 remodeling, a DPPC:DOPC:cholesterol molar ratio of
0.55:0.15:0.30, neutral 3h parameterized through CHARMM-GUI's CGenFF
interface, 20 ns equilibration, 310 K, 1 atm, and six production trajectories
up to 1.2 μs per set. Those settings describe the authors' runs; the public
Zenodo records contain PDB/XTC, not the complete force-field input package.

The IAG PDB contains one 56-atom residue named `IAG`, with elemental formula
C22H26F2N6. That matches the deposited 3h ligand [XY8](https://www.rcsb.org/ligand/XY8)
in [GPR6–3h structure 8T1V](https://www.rcsb.org/structure/8T1V), and differs
from CVN424 (X7T, C24H29F2N5O3). The primary paper identifies its GPR6–3h
simulation set. Do **not** label the public IAG trajectories as CVN424-bound.
CVN424 is a separate future simulation condition requiring new preparation.

The no-ligand and IAG topology PDBs share the same 275-residue protein atom-name
sequence and the same counts of 111 DPPC, 30 DOPC, and 60 CHL1 molecules. The
IAG PDB has the 56-atom ligand and six fewer waters, explaining its 38-atom
increase (75,661 versus 75,623 atoms). This supports matched condition analysis
and initial receptor Cα coordinates align over all 275 residues with about
0.10 Å RMSD. It does not prove identical force-field parameters or
equilibration. The
authors report wild-type sequence restoration and ICL3 remodeling for their MD
systems; the experimental crystal constructs contain fusion/mutations and must
not be substituted directly.

The local receptor residue identities match the primary paper's numbering for
the Phe152³·³⁶/Trp292⁶·⁴⁸ activation region, DRY, NPxxY, and sodium-cavity
residues. The [residue map](../scientific-methods/gpr6-residue-map.md) lists
verified positions and candidate atom selections. This numbering correspondence
does not by itself validate a mechanistic interpretation of trajectory features.

## Author-declared MD5 checksums

| Replica | No-ligand XTC MD5 | 3h-bound XTC MD5 |
| --- | --- | --- |
| 0 | `1b7333a821f224e0831713ae046be26c` | `40adfd0b1fed9f3a36d8a7051c70f38c` |
| 1 | `fc34ba5e1114d16215822be6e1516933` | `aa1e2c834c7489a84e39176cd092e4ba` |
| 2 | `04cb508685fa22942e405f67e81af1ed` | `2b687ec4b6952101f7326d3d419438af` |
| 3 | `818f71ef45f3c0df602b6ad285055b3e` | `073c55a12276d5c16358c8873f8b6e27` |
| 4 | `e543d0a932bf73b778d1b8780e593f54` | `dc0293bfb46523bbc0cd7133c474e174` |
| 5 | `33c3c0a288a95b3a09ea2cd6117647f2` | `8d259a6e964bdd5405f0fc843261e9e9` |

The no-ligand PDB MD5 is `2911c38cd4355fb670d2ee6b3466d678`; the
3h-bound PDB MD5 is `a24e25de2ae8404c7accc248c0e9e6fe`. The locally
downloaded 3h PDB matches the latter. The previously local no-ligand PDB
matches the former, but its local replica-1 XTC is only 514,187,264 bytes and
fails at the final indexed frame; verify each replacement against the author
checksum before treating it as complete.

## Analysis limits

The authors state that periodic-boundary coordinates can be restored with VMD.
MolScope's current streaming geometry uses stored, nonperiodic coordinates, so
raw membrane distances, contacts, and density maps need imaging before scientific
interpretation. The 12 trajectories give independent replicas for grouped ML,
provided split identity is preserved; frames within one replica are correlated
and must never be split across train and test. Comparing 3h against no-added-IAG
is an analysis of existing published simulations, not independent evidence that
MolScope can reproduce or improve their force-field protocol.

For new production MD, obtain or regenerate complete CHARMM36/GROMACS topology
and parameter files, including ligand 3h or CVN424 parameters. The published PDB
and XTC files alone cannot restart a chemically defined MD system. The
[OpenMM user guide](https://docs.openmm.org/latest/userguide/application/02_running_sims.html)
advises loading the CHARMM-GUI PSF for such nonstandard PDBs because terminal
patches and lipid bonds may not be recoverable from PDB coordinates.
