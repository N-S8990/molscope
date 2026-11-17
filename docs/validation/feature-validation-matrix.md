# MolScope Feature Validation Matrix

This document provides the authoritative biophysical and mathematical reference validation matrix for MolScope's feature extraction plugins and generic feature analyzers. It details the reference method or library, exact numerical tolerances, panel of test systems, implementation boundaries, and validation status (`VALIDATED` vs `EXPERIMENTAL`).

---

## 1. Validation Criteria and Status Definitions

To maintain strict scientific credibility, MolScope categorizes every feature extraction module under one of two statuses:

- **`VALIDATED`**:
  - The calculated observable possesses a rigorous, unambiguous mathematical or physical formulation.
  - The extracted values have been benchmarked directly against an established, trusted reference standard (e.g., MDAnalysis, FreeSASA, or exact analytical Euclidean/spherical geometry).
  - Continuous automated regression tests pass within tight, declared numerical tolerances (e.g., $\le 10^{-5}$ Å for distance, $\le 0.001^\circ$ for dihedrals, $100\%$ discrete category concordance).
  - Boundary conditions, sequence separation filters, and applicability guards have been explicitly tested.

- **`EXPERIMENTAL`**:
  - The module implements an operational proxy, coarse geometric heuristic, or unweighted network metric rather than a direct physical observable or free-energy state.
  - While code execution and edge cases are unit-tested and applicability-guarded, broad multi-ensemble or multi-forcefield scientific transferability is not yet formally claimed.
  - Caveats and operational definitions are explicitly documented to prevent scientific misinterpretation.

---

## 2. Comprehensive Feature Validation Matrix

| Plugin Identifier / Feature Name | Category | Reference Method & Library | Numerical Tolerance | Test Systems Panel | Status | Validation Test Module |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`protein.ca_distances_contacts`**<br>`ProteinCADistanceContactPlugin` | Protein Geometry | MDAnalysis `calc_bonds`, `distance_array`, Euclidean norm | $10^{-5}\text{ \AA}$ (dist)<br>Exact binary (0/1) | `1A22`<br>Synthetic PBC box | **VALIDATED** | [`test_reference_ca_distances.py`](../../tests/reference/test_reference_ca_distances.py) |
| **`protein.backbone_torsions`**<br>`ProteinBackboneTorsionPlugin` | Protein Conformation | MDAnalysis `Dihedral` analysis (`MDAnalysis.analysis.dihedrals`) | $10^{-6}$ (sin/cos)<br>$0.001^\circ$ (angle) | `1A22` | **VALIDATED** | [`test_reference_torsions.py`](../../tests/reference/test_reference_torsions.py) |
| **`protein.sidechain_torsions`**<br>`ProteinSidechainTorsionPlugin` | Side-chain Conformation | MDAnalysis `calc_dihedrals` / atom quartets | $10^{-6}$ (sin/cos)<br>$0.001^\circ$ (angle) | `1A22` | **VALIDATED** | [`test_reference_torsions.py`](../../tests/reference/test_reference_torsions.py) |
| **`protein.secondary_structure_dssp`**<br>`ProteinDSSPPlugin` | Secondary Structure | MDAnalysis native `DSSP` analysis module | $10^{-6}$ (fractions)<br>$100\%$ code match | `1A22`<br>`nucleic_base_centers` | **VALIDATED** | [`test_reference_dssp.py`](../../tests/reference/test_reference_dssp.py) |
| **`structural.sasa`**<br>`SASAPlugin` | Surface & Exposure | FreeSASA (Lee-Richards / Shrake-Rupley) & analytical spheres | $10^{-4}\text{ \AA}^2$ (sphere)<br>$10^{-5}\text{ \AA}^2$ (FreeSASA) | Analytical spheres<br>`protein_ligand`<br>`solvated_protein`<br>`1A22` | **VALIDATED** | [`test_reference_sasa.py`](../../tests/reference/test_reference_sasa.py) |
| **`ligand.plif`**<br>`ProteinLigandInteractionFingerprintPlugin` | Protein-Ligand Interactions | MDAnalysis `capped_distance`, vector planarity & angle filters | $10^{-5}\text{ \AA}$<br>Exact integer counts | `protein_ligand` | **VALIDATED** | [`test_reference_hbonds_plif.py`](../../tests/reference/test_reference_hbonds_plif.py) |
| **`protein_ligand.contacts`**<br>`ProteinLigandContactPlugin` | Protein-Ligand Proximity | MDAnalysis `distance_array` / `capped_distance` | $10^{-5}\text{ \AA}$<br>$10^{-6}$ (fraction) | `protein_ligand` | **VALIDATED** | [`test_reference_hbonds_plif.py`](../../tests/reference/test_reference_hbonds_plif.py) |
| **`protein.hydrogen_bonds`**<br>`ProteinHydrogenBondPlugin` | Non-covalent Interactions | Explicit vector D-H...A geometry, norm, and cosine threshold | $10^{-4}\text{ \AA}$<br>Exact integer counts | Synthetic D-H...A panel | **VALIDATED** | [`test_reference_hbonds_plif.py`](../../tests/reference/test_reference_hbonds_plif.py) |
| **`protein.salt_bridges`**<br>`ProteinSaltBridgePlugin` | Electrostatic Geometry | MDAnalysis `distance_array` between ASP/GLU and LYS/ARG | $10^{-5}\text{ \AA}$<br>Exact integer counts | `protein_ligand` | **VALIDATED** | [`test_reference_hbonds_plif.py`](../../tests/reference/test_reference_hbonds_plif.py) |
| **`protein.hydrophobic_contacts`**<br>`ProteinHydrophobicContactPlugin` | Nonpolar Packing | MDAnalysis `capped_distance` on aliphatic/aromatic carbons | $10^{-5}\text{ \AA}$<br>Exact integer counts | `1A22`<br>`protein_ligand` | **VALIDATED** | [`test_reference_hbonds_plif.py`](../../tests/reference/test_reference_hbonds_plif.py) |
| **`solvation.water_shells`**<br>`SolvationShellPlugin` | Solvation Structure | MDAnalysis `distance_array` on unique water oxygens | $10^{-5}\text{ \AA}$<br>Exact integer counts | `solvated_protein` | **VALIDATED** | [`test_feature_reference.py`](../../tests/reference/test_feature_reference.py) |
| **`nucleic_acid.base_proximities`**<br>`NucleicAcidBaseProximityPlugin` | Nucleic Geometry | Direct arithmetic center Euclidean distance | $10^{-5}\text{ \AA}$<br>Exact integer counts | `nucleic_base_centers` | **VALIDATED** | [`test_feature_reference.py`](../../tests/reference/test_feature_reference.py) |
| **`protein.residue_heavy_distances_contacts`**<br>`ProteinResidueHeavyDistancePlugin` | Coarse-grain Proximity | MDAnalysis `distance_array` min heavy-atom pair distance | $10^{-5}\text{ \AA}$<br>Exact binary (0/1) | `1A22` | **VALIDATED** | [`test_expanded_protein_features.py`](../../tests/unit/test_expanded_protein_features.py) |
| **`coordinates.selected_atoms`**<br>`SelectedCoordinatePlugin` | Raw Coordinates | MDAnalysis Cartesian coordinates as stored | Exact ($0.0\text{ \AA}$) | Arbitrary PDB/XTC | **VALIDATED** | [`test_feature_plugins.py`](../../tests/integration/test_feature_plugins.py) |
| **`geometry.global_radius_of_gyration`**<br>`GenericFeatureExtractor` | Global Compactness | Root-mean-square distance from center of geometry | $10^{-6}\text{ \AA}$ | `solvated_protein`<br>`1A22` | **VALIDATED** | [`test_generic_extraction.py`](../../tests/integration/test_generic_extraction.py) |
| **`dynamics.kabsch_rmsd_to_first_frame`**<br>`GenericFeatureExtractor` | Rigid-body Dynamics | Kabsch SVD alignment & RMSD formula | $10^{-6}\text{ \AA}$ | `solvated_protein`<br>`1A22` | **VALIDATED** | [`test_generic_extraction.py`](../../tests/integration/test_generic_extraction.py) |
| **`geometry.component_contact_fraction`**<br>`GenericFeatureExtractor` | Inter-molecular Contact | Cross-component all-atom distance array threshold | $10^{-6}$ (fraction) | Multi-component PDBs | **VALIDATED** | [`test_generic_extraction.py`](../../tests/integration/test_generic_extraction.py) |
| **`dynamics.global_rmsf_mean`**<br>`GenericFeatureExtractor` | Fluctuation Magnitude | RMSF around trajectory mean after alignment | $10^{-5}\text{ \AA}$ | Trajectory benchmarks | **VALIDATED** | [`test_generic_extraction.py`](../../tests/integration/test_generic_extraction.py) |
| **`protein.aromatic_stacking`**<br>`ProteinAromaticStackingPlugin` | Aromatic Geometry | PHE/TYR ring plane normal angles & center distance | $\le 30.0^\circ$<br>$3.5 \le d \le 7.0\text{ \AA}$ | `1A22`<br>GPR6 chemical panel | **EXPERIMENTAL** | [`test_feature_plugins.py`](../../tests/integration/test_feature_plugins.py) |
| **`networks.protein_contact_network`**<br>`ProteinContactNetworkPlugin` | Residue Network Topology | NetworkX unweighted graph metrics (betweenness, degree) | Graph isomorphism<br>Exact integer counts | `1A22` | **EXPERIMENTAL** | [`test_feature_plugins.py`](../../tests/integration/test_feature_plugins.py) |
| **`electrostatics.molecular_dipole`**<br>`MolecularDipolePlugin` | Electrostatics | Partial-charge weighted center-of-charge dipole moment | $10^{-5}\text{ e}\cdot\text{\AA}$ | Neutral test systems | **EXPERIMENTAL** | [`test_feature_plugins.py`](../../tests/integration/test_feature_plugins.py) |

---

## 3. Detailed Plugin Profiles and Validation Outcomes

### 3.1 C-alpha Distances and Contacts (`protein.ca_distances_contacts`)
- **Plugin Class**: `ProteinCADistanceContactPlugin`
- **Reference Standard**: MDAnalysis `calc_bonds`, `distance_array`
- **Tolerance**: $10^{-5}$ Å for distance; exact binary match for contact indicators.
- **Test Systems**:
  - `1A22.pdb`: Human growth hormone complex with receptor (372 Cα atoms across chains A and B).
  - Synthetic periodic boundary box with dimensions $80 \times 80 \times 80\text{ \AA}^3$.
- **Validation Outcome**:
  - Distance values match direct Euclidean norms ($\|\mathbf{r}_i - \mathbf{r}_j\|_2$) with mean discrepancy $< 10^{-6}$ Å.
  - Minimum sequence separation ($\ge 3$ residues within the same chain) strictly enforced.
  - When `use_pbc=True`, minimum-image distance across cell boundaries matches MDAnalysis `calc_bonds(..., box=box)` exactly.
  - Declared residue-pair targeting (`residue_pairs=[[i, j], ...]`) accurately maps designated pairs without allocating dense $N \times N$ matrices.

### 3.2 Backbone Dihedrals (`protein.backbone_torsions`)
- **Plugin Class**: `ProteinBackboneTorsionPlugin`
- **Reference Standard**: `MDAnalysis.analysis.dihedrals.Dihedral`
- **Tolerance**: $10^{-6}$ for trigonometric components $(\sin \theta, \cos \theta)$; $\Delta \theta < 0.001^\circ$ for reconstructed angles.
- **Test Systems**: `1A22.pdb`.
- **Validation Outcome**:
  - Per-residue $\phi$ ($C_{i-1}-N_i-C\alpha_i-C_i$), $\psi$ ($N_i-C\alpha_i-C_i-N_{i+1}$), and $\omega$ ($C\alpha_i-C_i-N_{i+1}-C\alpha_{i+1}$) dihedrals compared against MDAnalysis `Dihedral` across 366 evaluated residues.
  - Reconstructed angles $\operatorname{arctan2}(\sin \theta, \cos \theta)$ match MDAnalysis values within $0.0001^\circ$.
  - Trigonometric identity $\sin^2 \theta + \cos^2 \theta = 1.0$ verified for all output pairs.
  - Terminal residues lacking complete quartets gracefully omitted without producing `NaN` or unhandled exceptions.

### 3.3 Secondary Structure Assignment (`protein.secondary_structure_dssp`)
- **Plugin Class**: `ProteinDSSPPlugin`
- **Reference Standard**: MDAnalysis native `DSSP` wrapper (`MDAnalysis.analysis.dssp.DSSP`)
- **Tolerance**: $10^{-6}$ for aggregate fractions; $100\%$ concordant assignment for per-residue binary indicators.
- **Test Systems**: `1A22.pdb`, `nucleic_base_centers.pdb`.
- **Validation Outcome**:
  - Aggregate helix, sheet, and coil fractions match MDAnalysis reference calculation to 6 decimal places.
  - Partition of unity: $f_{\text{helix}} + f_{\text{sheet}} + f_{\text{coil}} \equiv 1.000000$.
  - Per-residue binary indicator evaluation across all 372 residues demonstrates $100\%$ identity with canonical DSSP code dictionary (`H` $\to$ helix, `E` $\to$ sheet, `-` $\to$ coil).
  - Graceful refusal (`ApplicabilityStatus.NOT_APPLICABLE`) when executed on non-protein structures.

### 3.4 Solvent-Accessible Surface Area (`structural.sasa`)
- **Plugin Class**: `SASAPlugin`
- **Reference Standard**: Analytical spherical geometry and direct FreeSASA (Lee-Richards / Shrake-Rupley, probe radius 1.4 Å, 100 points).
- **Tolerance**: $10^{-4}\text{ \AA}^2$ for analytical sphere benchmarks; $10^{-5}\text{ \AA}^2$ for FreeSASA multi-atom comparisons.
- **Test Systems**:
  - Synthetic single Carbon atom at origin ($R = 1.88$ Å).
  - Synthetic Carbon dimer with infinite separation ($d = 25$ Å).
  - Synthetic Carbon dimer with covalent overlap ($d = 1.54$ Å).
  - `protein_ligand_contacts.pdb` and `solvated_protein.pdb`.
- **Validation Outcome**:
  - Single isolated atom SASA matches analytical formula $4\pi(R + r_{\text{probe}})^2 = 135.19404\text{ \AA}^2$ with $0.0\text{ \AA}^2$ discrepancy.
  - Non-overlapping dimer exhibits exact linear additivity: $A_{\text{dimer}} = 2 \times A_{\text{monomer}}$.
  - Covalent dimer demonstrates expected steric shielding ($A_{\text{dimer}} < 2 \times A_{\text{monomer}}$), matching FreeSASA integration.
  - Sum of residue-level SASA contributions strictly equals total system SASA.
  - Water and ions are excluded according to the declared scope policy.

### 3.5 Protein-Ligand Interaction Fingerprints (`ligand.plif`)
- **Plugin Class**: `ProteinLigandInteractionFingerprintPlugin`
- **Reference Standard**: MDAnalysis `capped_distance`, topological nonpolar carbon selection, planar ring detection.
- **Tolerance**: $10^{-5}$ Å for minimum contact distances; exact integer counts for interaction types.
- **Test Systems**: `protein_ligand_contacts.pdb`.
- **Validation Outcome**:
  - Nonpolar carbon-carbon contacts ($d \le 4.5$ Å) match MDAnalysis selection after excluding heteroatom-adjacent polar carbons.
  - Donor-acceptor hydrogen-bonding contacts ($d \le 3.5$ Å) match bidirectional distance queries.
  - Aromatic ring centroid proximity ($d \le 5.0$ Å) matches SVD ring plane calculations.
  - Total interaction count matches sum of individual interaction categories.

### 3.6 Directional Hydrogen Bonds (`protein.hydrogen_bonds`)
- **Plugin Class**: `ProteinHydrogenBondPlugin`
- **Reference Standard**: Direct 3D vector geometry (donor-acceptor distance $\le 3.5$ Å and $\angle(\text{D-H}\cdots\text{A}) \ge 150^\circ$).
- **Tolerance**: $10^{-4}$ Å; exact integer counts.
- **Test Systems**: Synthetic geometric hydrogen-bonding panel with known positive and negative controls.
- **Validation Outcome**:
  - Near-linear configuration ($d = 2.8$ Å, $\theta = 170^\circ$) accurately recognized.
  - Acute/bent configuration ($d = 2.8$ Å, $\theta = 90^\circ$) correctly rejected by angular criterion.
  - Distant configuration ($d = 4.2$ Å, $\theta = 175^\circ$) correctly rejected by distance cutoff.
  - Explicit hydrogen requirement validated: systems missing explicit hydrogens return `NOT_APPLICABLE` rather than silent zero values.

---

## 4. Experimental Features and Scientific Scope

The following features are designated **EXPERIMENTAL**:

1. **`protein.aromatic_stacking`**:
   - *Scope*: Identifies PHE and TYR rings with centroid separation between 3.5 and 7.0 Å and plane normal angle $\le 30^\circ$.
   - *Limitation*: Operational geometric proxy only. It does not calculate $\pi-\pi$ dispersion energy, does not model electrostatic quadrupole repulsion, and excludes TRP and HIS.

2. **`networks.protein_contact_network`**:
   - *Scope*: Constructs an unweighted graph from heavy-atom residue contacts ($d \le 4.5$ Å, sequence separation $\ge 3$) and computes graph metrics (betweenness centrality, degree, modularity communities).
   - *Limitation*: Graph centrality does not prove allosteric communication pathways or thermodynamic coupling.

3. **`electrostatics.molecular_dipole`**:
   - *Scope*: Calculates $\sum q_i \mathbf{r}_i$ for neutral solute scopes.
   - *Limitation*: Highly sensitive to net neutrality and forcefield partial charge conventions; undefined / origin-dependent for charged solutes.

---

## 5. Automated Verification Summary

All reference validation tests run continuously in CI and can be verified locally via:

```bash
/Users/nirav/ds-env/bin/python -m pytest tests/reference/ -v
```

### Test Suite Execution Output
```text
tests/reference/test_feature_reference.py::ReferenceValidationTests::test_nucleic_base_centers_match_direct_geometry_reference PASSED
tests/reference/test_feature_reference.py::ReferenceValidationTests::test_protein_ligand_and_salt_bridge_match_mdanalysis_distance_reference PASSED
tests/reference/test_feature_reference.py::ReferenceValidationTests::test_sasa_wrapper_matches_direct_freesasa_reference PASSED
tests/reference/test_feature_reference.py::ReferenceValidationTests::test_solvation_shell_matches_mdanalysis_distance_reference PASSED
tests/reference/test_reference_ca_distances.py::ReferenceCADistanceValidationTests::test_ca_contacts_match_cutoff_threshold PASSED
tests/reference/test_reference_ca_distances.py::ReferenceCADistanceValidationTests::test_ca_explicit_declared_pairs_match_reference PASSED
tests/reference/test_reference_ca_distances.py::ReferenceCADistanceValidationTests::test_ca_minimum_sequence_separation_enforced PASSED
tests/reference/test_reference_ca_distances.py::ReferenceCADistanceValidationTests::test_ca_pairwise_distances_match_mdanalysis_euclidean PASSED
tests/reference/test_reference_ca_distances.py::ReferenceCADistanceValidationTests::test_ca_pbc_minimum_image_distance_reference PASSED
tests/reference/test_reference_dssp.py::ReferenceDSSPValidationTests::test_dssp_aggregate_fractions_match_mdanalysis_reference PASSED
tests/reference/test_reference_dssp.py::ReferenceDSSPValidationTests::test_dssp_canonical_structure_composition PASSED
tests/reference/test_reference_dssp.py::ReferenceDSSPValidationTests::test_dssp_non_protein_system_gracefully_handled PASSED
tests/reference/test_reference_dssp.py::ReferenceDSSPValidationTests::test_dssp_per_residue_assignments_match_reference_dictionary PASSED
tests/reference/test_reference_hbonds_plif.py::ReferenceHBondsPLIFValidationTests::test_plif_interactions_match_direct_mdanalysis_geometry PASSED
tests/reference/test_reference_hbonds_plif.py::ReferenceHBondsPLIFValidationTests::test_protein_hydrogen_bond_directional_criteria PASSED
tests/reference/test_reference_hbonds_plif.py::ReferenceHBondsPLIFValidationTests::test_protein_ligand_contacts_match_mdanalysis_distance_array PASSED
tests/reference/test_reference_hbonds_plif.py::ReferenceHBondsPLIFValidationTests::test_protein_salt_bridge_and_hydrophobic_contact_geometry PASSED
tests/reference/test_reference_sasa.py::ReferenceSASAValidationTests::test_sasa_isolated_sphere_matches_analytical_geometry PASSED
tests/reference/test_reference_sasa.py::ReferenceSASAValidationTests::test_sasa_non_overlapping_spheres_exact_additivity PASSED
tests/reference/test_reference_sasa.py::ReferenceSASAValidationTests::test_sasa_protein_ligand_complex_vs_direct_freesasa PASSED
tests/reference/test_reference_sasa.py::ReferenceSASAValidationTests::test_sasa_solvation_scope_policy PASSED
tests/reference/test_reference_sasa.py::ReferenceSASAValidationTests::test_sasa_steric_shielding_overlapping_atoms PASSED
tests/reference/test_reference_torsions.py::ReferenceTorsionValidationTests::test_backbone_torsion_trigonometric_normalization PASSED
tests/reference/test_reference_torsions.py::ReferenceTorsionValidationTests::test_backbone_torsions_match_mdanalysis_dihedral_analysis PASSED
tests/reference/test_reference_torsions.py::ReferenceTorsionValidationTests::test_sidechain_chi_torsions_vs_mdanalysis_dihedrals PASSED
tests/reference/test_reference_torsions.py::ReferenceTorsionValidationTests::test_terminal_residues_omitted_without_errors PASSED
tests/reference/test_reference_torsions.py::ReferenceTorsionValidationTests::test_trans_peptide_omega_conformation PASSED

======================= 27 passed, 36 warnings in 3.57s ========================
```
