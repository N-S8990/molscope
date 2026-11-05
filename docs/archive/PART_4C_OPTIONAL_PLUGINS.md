> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 4C: Optional Domain Plugins

The generic extractor intentionally does not guess protein, nucleic-acid, lipid, ligand, or polymer chemistry. Domain-specific calculations are added as plugins with declared requirements.

## Capability-gated plugins

`protein.backbone_torsions` computes sin/cos encodings of phi, psi, and omega only when both capabilities are present:

- `protein_backbone`
- `bonds`

For unsupported systems it writes an explicit `not_applicable` assessment, empty feature tables, and no fabricated zeros.

Run it with:

```bash
python -m molscope.extract_plugin_features \
  --plugin protein.backbone_torsions \
  --topology path/to/topology.pdb \
  --trajectory path/to/trajectory.dcd \
  --out-dir protein_torsions
```

The output includes feature values, an applicability assessment, and timing/parameter provenance.

The current platform registers the following implemented plugin families with
their applicability requirements, provenance, limitations, and validation
status in `molscope.registries.DEFAULT_FEATURE_REGISTRY`:

- protein backbone torsions;
- nonlocal Cα distance/contact maps and aggregate MDAnalysis-DSSP secondary-
  structure fractions for proteins with supported backbone geometry;
- protein-ligand contacts, salt bridges, hydrophobic contacts, directional
  hydrogen bonds, and PHE/TYR aromatic stacking;
- all-atom SASA (optional FreeSASA backend);
- water-shell occupancy and neutral-scope molecular dipole;
- dynamic protein residue-contact networks.

Matched-policy reference fixtures now cover the protein-ligand contact,
salt-bridge, water-shell, base-center proximity, and SASA wrappers; see
`docs/REFERENCE_VALIDATION.md`. This is deliberately narrower than universal
scientific validation: PBC imaging, chemical interaction identity, and
cross-force-field transferability are not established by those fixtures.
Other plugins remain implementation-tested but not reference-validated.
The Cα and DSSP plugins are non-PBC initial implementations. Their declared
cutoffs, sequence-separation, and backend conventions must be matched during
comparison; secondary-structure fractions do not establish a folding or
functional mechanism.
Nucleic-acid base-pair/stacking, membrane/lipid order and thickness, broader
ligand chemistry, and polymer-specific descriptors remain future
capability-gated families.

Adding a plugin does not make it globally applicable.
