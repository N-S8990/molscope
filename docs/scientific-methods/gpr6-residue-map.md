# GPR6 residue map for interpretable trajectory features

The [primary GPR6 structure and MD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850111/)
uses the human GPR6 residue numbers below. The two deposited MD topology PDBs
contain the same ordered 275-residue receptor protein, numbered 72–346. I
checked the no-added-inverse-agonist topology directly: every residue and
atom named below is present with the expected amino-acid identity. Thus the
**residue-number map is supported** for these deposited trajectories. A change
in a mapped coordinate does not by itself establish activation, ligand
mechanism, or a stable molecular state.

| Region | Paper-identified residues | Concrete receptor-only measurements |
| --- | --- | --- |
| Orthosteric pocket | His128²·⁶⁰, Leu148³·³², Phe152³·³⁶, Val219 in ECL2, Trp292⁶·⁴⁸, Phe295⁶·⁵¹, Leu315⁷·³⁹ | Internal pocket heavy-atom distances and aromatic ring centroid/orientation measures. Ligand contacts require the full 3h topology and are descriptive only for that condition. |
| Phe/Trp activation region | Phe152³·³⁶, Phe234⁵·⁴⁷, Trp292⁶·⁴⁸; Val156³·⁴⁰, Val237⁵·⁵⁰, Phe288⁶·⁴⁴, Phe238⁵·⁵¹ | Phe152/Phe234/Trp292 χ1/χ2 torsions and aromatic ring relationships; CA or side-chain distances for the Val156–Val237–Phe288 trio. These probe conformations discussed in the paper. |
| Cytoplasmic opening | Arg166³·⁵⁰ and Thr274⁶·³⁰ | Arg166–Thr274 distance, with atom pair stated (for example CA–CA). The paper used their distance as an MSM coordinate; a CA proxy must not be described as the same atom-level definition unless confirmed. |
| DRY and partner contacts | Asp165³·⁴⁹–Arg166³·⁵⁰–Tyr167³·⁵¹, Tyr176 in ICL2, Gly277⁶·³³ | Asp165 OD1/OD2–Tyr176 OH distance; Arg166 NH1/NH2/NE–Gly277 O distance; DRY side-chain torsions. These are geometric proxies for paper-described polar interactions, not hydrogen-bond assignments without angular checks. |
| NPxxY and intracellular network | Asn325⁷·⁴⁹–Pro326⁷·⁵⁰–Ile327–Ile328–Tyr329⁷·⁵³; Arg166³·⁵⁰ | NPxxY backbone RMSD to an aligned inactive 8T1V reference, with reference alignment documented; Arg166 CZ to Tyr329 aromatic-ring centroid distance/orientation. The paper discusses an Arg166–Tyr329 π-cation contact in the Gs-bound state. |
| Sodium cavity | Asp118²·⁵⁰, Ser155³·³⁹, Asn321⁷·⁴⁵ | Internal side-chain geometry among Asp118 OD1/OD2, Ser155 OG, and Asn321 OD1/ND2. Sodium occupancy and water networks require the original full-system XTC with ions/water and periodic imaging. The structure's putative sodium density could also be water, as the paper states. |
| ECL2 constraint | Cys209, Cys216 | SG–SG distance as a geometric continuity check for the paper-described disulfide; confirm bond topology before calling it a bonded-state measurement. |

All listed atom names were found in the deposited PDB, including side-chain
atoms and hydrogens. Histidine 128 is named `HSE` (histidine protonation
variant) in the MD topology. The receptor-only projected XTC contains these
protein atoms but excludes 3h, membrane lipids, water, and ions. Features
involving those excluded species must use the original full-system files with
an explicit periodic-imaging policy and must not be silently computed from
the receptor-only projection.

The first 39-feature run chose 18 Cα pairs among residues 150–153 and 290–293
after a one-trajectory baseline inspection. Phe152 and Trp292 **are** mapped
to the paper's 3.36 and 6.48 positions; the observed pair distance is still an
exploratory proxy. Additional feature families should be fixed before reading
their condition effects and compared on held-out trajectories or later
matched windows.

## Structure views of feature evidence

Run `python -m molscope.reporting.molecular_map --topology RECEPTOR_PROTEIN_PDB
--run-dir ANALYSIS_RUN` to create receptor-coordinate maps from the grouped
predictive and exploratory condition tables. `MOLECULAR_MAPS.md` in the run
links the interactive HTML, full-receptor cartoon PNG, mapped-region close-up,
coordinate-trace PNG, and exact feature-to-residue mapping CSV. The HTML and
cartoons use the deposited receptor PDB coordinates in ångströms; they are not
PCA plots or averaged MD structures. The displayed geometry is one deposited
frame, not a dynamic representation of every trajectory frame.

For a pair feature, both named residues receive the feature's score and a link
joins their Cα positions in the interactive view. A single-residue torsion
maps only to its named residue. Global geometry and DSSP fractions have no
honest residue location and are left off the structure. If multiple features
map to one residue, the color uses the signed score with the largest absolute
magnitude; the mapping CSV retains every contributing feature. A residue-name
mismatch stops that feature from being mapped rather than guessing identity.
The named heavy-atom chemistry panel uses `gpr6_chemical_definitions.json` for
its residue assignments; the renderer checks both atom-group names against the
PDB residue identities. Its lines connect the receptor Cα positions of the two
defined residues for orientation, even when the measured geometry itself uses
side-chain atoms, ring centroids, or ring-plane normals. Such a line is not the
measured heavy-atom vector or a chemical bond.

Predictive color is the signed change in grouped trajectory-mean test accuracy
when the feature is removed (baseline minus without-feature). This is model
dependence, not statistical or causal significance. The condition view uses the
sign of the replica-mean condition difference and the magnitude of the
exploratory BH-adjusted q value; it does not encode effect size, and q values
are tied to the tested feature family. Gray/white receptor regions have no
mapped local evidence, not a measured zero effect. Red and blue do not imply
activation or inhibition.
