# End-to-End Tutorial: HP35 Villin Headpiece (2RJY) Analysis with MolScope

This tutorial demonstrates how to use MolScope's high-level `Study` API to perform a complete molecular dynamics analysis workflow on the fast-folding HP35 Villin Headpiece (PDB: `2RJY`), moving from raw trajectory frames to non-linear conformational manifolds, free energy basins, explainable decision tree rules, and 3D molecular depictions.

---

## 1. System Overview and Prerequisites

The Villin Headpiece submodule (HP35, residues 13–76) is a canonical benchmark system for protein folding and conformational dynamics.
- **Topology**: PDB structure `protein.pdb`
- **Trajectory**: 1,000 ns simulation trajectory `simulation_run1.xtc` (10,001 frames)

Ensure your environment has MolScope installed:
```bash
pip install -e ".[dssp,simulation,parquet]"
```

---

## 2. Interactive Study Workflow

### Step 2.1: Initializing the Study and Adding the System
```python
from pathlib import Path
from molscope.api import Study

# Initialize a stateful Study
study = Study(
    project_id="2rjy_villin_headpiece",
    out_dir="/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark",
)

# Add the 2RJY system with trajectory files
study.add_system(
    system_id="villin_headpiece",
    topology="/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY/protein.pdb",
    trajectories=["/Users/nirav/Docs/mdxplain study/mdxplain/data/2RJY/simulation_run1.xtc"],
)
```

### Step 2.2: Configuring Periodic Boundary Imaging and Feature Plugins
Configure PBC centering and feature extraction plugins (Cα pairwise contacts, secondary structure fractions via MDAnalysis DSSP, and key sidechain rotamers):
```python
# Unfold and center on the protein to eliminate boundary jump artifacts
study.configure_imaging(unwrap=True, center_selection="protein", wrap=False)

# Configure domain plugins
study.configure_features([
    "protein.ca_distances_contacts",
    "protein.secondary_structure_dssp",
    "protein.sidechain_torsions",
])

# Set parameters (e.g., Cα distance cutoff 8.0 Å, stride 5 across 10,001 frames = 2,000 frames)
study.set_plugin_parameters("protein.ca_distances_contacts", {
    "selection": "protein and name CA",
    "contact_cutoff_angstrom": 8.0,
    "max_pairs": 3000,
})
study.set_plugin_parameters("protein.sidechain_torsions", {
    "residue_ids": [17, 20, 23, 27, 30, 35, 40, 50],
})

# Run extraction
study.run(extract_only=True)
features = study.get_features()
print(f"Extracted {features.shape[1]} features across {features.shape[0]} frames.")
```

---

## 3. Conformational Dimensionality Reduction

### Step 3.1: Time-lagged Independent Component Analysis (TICA)
Isolate slow kinetic transition modes using variational TICA:
```python
# Fit TICA with lag_time = 2 frames (1.0 ns at stride 5)
tica_res = study.run_tica(lag_time=2, components=5)
print("Slowest implied timescales (ps):", tica_res.implied_timescales[:3])
```

### Step 3.2: Contact Kernel PCA (CKPCA)
Capture non-linear conformational manifolds by projecting heavy-atom residue contact kernels into RKHS embeddings:
```python
ckpca_res = study.run_contact_kpca(n_components=5)
print("Contact KPCA explained variance ratio:", ckpca_res.explained_variance_ratio[:3])
```

---

## 4. State Discovery via Density Peak Advanced (DPA)

Clustering on the low-dimensional TICA reaction coordinates discovers metastable conformational free energy basins:
```python
# Cluster with Density Peak Advanced
dpa_res = study.cluster_dpa(coordinate_name="TICA", z_score=2.0)
print(f"Discovered {dpa_res.n_clusters} conformational basins:")
for state_id, pop in dpa_res.state_populations.items():
    print(f"  State {state_id}: {pop} frames ({pop / len(features) * 100:.1f}%)")
```

---

## 5. Free Energy Surface (FES) Visualization

Compute the potential of mean force $\Delta G(x, y) = -k_B T \ln P(x, y)$ on the slowest kinetic coordinates (TIC1 vs TIC2) and export interactive Plotly contours:
```python
fes = study.plot_free_energy_landscape(
    dim1="TIC1",
    dim2="TIC2",
    temperature_kelvin=300.0,
    energy_units="kcal/mol",
    out_path="/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/fes_tic1_tic2.png",
    interactive_html="/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/fes_tic1_tic2.html",
)
print(f"Detected {len(fes.basins)} free energy minima.")
```

---

## 6. Explainable AI: Decision Tree Rule Extraction

Translate abstract clusters into human-readable biophysical criteria:
```python
# Extract boolean decision rules explaining each cluster
explainer = study.explain_clusters(max_depth=4, top_k_features=8)

for rule in explainer.rules:
    print(f"\n--- State {rule.cluster_id} ({rule.cluster_name}) ---")
    print(f"Support: {rule.support_fraction * 100:.1f}% | Purity: {rule.purity * 100:.1f}%")
    print("Rule:", rule.simplified_rule_text())
```
*Example extracted rule:*
$$\text{IF } [\text{dist}(C_\alpha 35, C_\alpha 50) \le 6.8\text{ \AA}] \text{ AND } [\chi_1(\text{PHE 47}) \ge 65.2^\circ] \implies \text{Folded Native State (Purity 98.4\%)}$$

---

## 7. 3D Structural Export for PyMOL and ChimeraX

Project feature scores and state-discriminating importance directly onto residue B-factors in a PDB file:
```python
scripts = study.export_3d(out_dir="/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/3d_visualization")
print(f"Generated PyMOL script: {scripts['pymol_script']}")
print(f"Generated ChimeraX script: {scripts['chimerax_script']}")
```

Run in PyMOL:
```bash
pymol /Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/3d_visualization/view_scores.pml
```
