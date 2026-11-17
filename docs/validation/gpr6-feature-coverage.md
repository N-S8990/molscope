# GPR6 feature coverage and open measurements

The current **63-feature classifier panel** measures receptor shape, 25
paper-mapped Cα distances, side-chain χ geometry, and secondary-structure
fractions. A separate 18-feature receptor heavy-atom panel covers selected
DRY, sodium-cavity, NPxxY, aromatic, and pocket geometries. These are useful
measurements, **not an exhaustive inventory** of GPR6 behavior. The
[primary structure and MD paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC11850111/)
discusses the Phe152 switch, 3h binding, a putative lipid in the pseudo-apo
pocket, intracellular opening, and metastable states; the public
[no-added-IAG](https://zenodo.org/records/14060756) and
[3h-bound](https://zenodo.org/records/14060604) trajectories provide atoms
for more targeted analyses.

The deposited no-added-IAG PDB has 4,273 receptor atoms, 16,086 TIP3 waters,
39 `SOD` ions, 43 `CLA` ions, 111 DPPC, 30 DOPC, and 60 CHL1 molecules. The
3h PDB has the same receptor/lipid/ion counts, 16,080 TIP3 waters, and one
56-atom `IAG` residue. Thus the *source* XTCs contain ligand, solvent, ions,
and membrane coordinates, but the current classifier's receptor-only
projection excludes them. The source PDBs do not provide the complete bond,
partial-charge, force-field, and ligand-parameter topology.

| Observable | Status and input | Gate before scientific use | Priority |
| --- | --- | --- | --- |
| Phe152/Trp292, TM3/TM5/TM6, DRY, NPxxY internal geometry | Cα, χ, DSSP in the 63 panel; selected side-chain/ring geometries in the separate 18 panel. Both conditions have the necessary receptor atoms. | Keep exact atom selections and residue mapping; assess longer-window and independent-replica stability. The 18-feature chemistry panel does not meaningfully improve the 63-feature classifier. | Already measured; refine interpretation |
| Intracellular TM6 opening and NPxxY reference state | Arg166–Thr274 Cα proxy is measured; the paper also uses Arg166–Thr274 distance and NPxxY RMSD relative to inactive structure in its state analysis. | Confirm the paper's atom-level definition, align the receptor to a documented 8T1V reference, then define NPxxY backbone RMSD. Do not call the Cα proxy the paper's exact reaction coordinate. | High |
| 3h ligand pose and residue contacts | All 56 ligand atoms and receptor atoms exist in the bound full-system PDB/XTC. No ligand exists in the no-added condition. MolScope's contact plugin can count broad contacts but currently uses nonperiodic geometry. | Image ligand and receptor together across PBC; validate 3h atom identity; calculate residue-specific occupancy, pose RMSD, and contact persistence. Keep as **bound-only descriptive evidence**, not a condition-classifier feature whose mere presence reveals the label. | High |
| Pocket lipid entry and cholesterol contacts | DPPC, DOPC, and CHL1 are present in both full-system trajectories; none is present in the receptor-only projection. The no-added-IAG condition must not be called an empty pocket. | Restore whole-molecule imaging and membrane leaflets; define a pocket/entry region and per-lipid residence, not only a 4.5 Å frame count. Lipid identity in the experimental pseudo-apo structure is unresolved. | High |
| Sodium cavity and pocket hydration | `SOD` ions and TIP3 waters exist in both full systems. The 63/18 panels measure receptor geometry near Asp118/Ser155/Asn321, **not ion or water occupancy**. The generic water-shell plugin is whole-component and nonperiodic. | Image ions/waters with receptor, identify cavity selection, measure occupancy and residence against controls; avoid equating a structural density with a sodium ion. | High |
| DRY/NPxxY hydrogen bonds and polar networks | Selected donor/acceptor heavy-atom distances are measured in the 18 panel. MolScope's hydrogen-bond plugin requires donor–H bonds and currently cannot validate them from these PDB-only topologies. | Supply/reconstruct and verify bond connectivity and protonation; add angle criteria, then assess interaction occupancy. Heavy-atom proximity alone is not a hydrogen bond. | High |
| Pocket accessibility, cavity volume, and SASA | The receptor atoms support a defined SASA calculation; current 63 panel has no pocket-volume or pocket-specific SASA measure. | Define protein/ligand/membrane scope and validated cavity algorithm; receptor-only SASA cannot say whether the bound pocket is solvent exposed in the full system. | Medium |
| Kinetics and metastability | MolScope reports exploratory clusters/transitions. The present comparison covers 0–199.9 ns per replica; the author's deposited trajectories extend substantially longer. | Analyze full checksum-verified trajectories, lag-time convergence, implied timescales, and state-model uncertainty before physical metastability or transition-rate claims. | High |
| Energies, electrostatics, and new production MD | PDB/XTC coordinates alone do not contain the required force-field bonds, charges, 3h parameters, or restart state. | Obtain or rebuild and validate the complete CHARMM36/CGenFF topology and simulation protocol. Do not infer binding free energy or launch a supposedly matched production MD from coordinates alone. | Required for such claims |
| CVN424 response | No CVN424 molecules or trajectories are in these public 3h/no-added-IAG records. | Prepare and validate a separate CVN424-bound system and independent replicas. | Separate study |

## What this means for the classifier

The 63-feature panel is a **current, measured choice**, not proof that no
other feature matters. It nearly matches the 81-feature model's grouped
accuracy in two consecutive 100 ns windows while using fewer columns; those
windows reuse the same twelve replicas and are not independent validation.
Adding all possible atom pairs would be redundant and increase model-selection
bias. The next scientific gains are likely to come from valid full-system
measurements, longer trajectories, and independent testing rather than a
larger arbitrary receptor-distance matrix.

The [study summary](gpr6-public-analysis.md) and
[paper-mapped residue selections](../scientific-methods/gpr6-residue-map.md)
contain current results. The [source provenance](gpr6-source-provenance.md)
records the full-file checksum gate.
