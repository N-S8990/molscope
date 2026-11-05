# Test-readiness handoff

The reorganized platform passes **129 tests and 3 subtests** in
`/Users/nirav/ds-env/bin/python` (baseline: 117 tests). Compilation, all eleven
unified command help checks, wheel construction, and installation/import checks
outside the source checkout passed. The real managed 2RJY extraction run
completed, and its report-local image/table references resolve.
This is readiness for controlled testing of implemented paths, not a claim of
complete molecular coverage or demonstrated superiority to mdxplain.

The organized real report is at
`studies/2rjy/runs/organized_validation/report/extraction_report.html`.
New managed-run tests cover input snapshots, output grouping, duplicate-run
rejection, path traversal rejection, failure records, and import compatibility.
The source-snapshot diff check is clean. Text whitespace/conflict checks passed;
fixed-width PDB data retains its original padding. Git repository initialization
remains blocked by filesystem permissions, so there is no Git commit for this
migration.

## Implemented for testing

- Full trajectory integrity scanning with checksums precedes extraction.
- `extraction_evidence.json` retains system profiles, applicability reasons,
  parameters, measured extractor runtime and trajectory summaries. Sessions and
  caches preserve this evidence; old cache entries without it are recomputed.
- `analysis.plugin_parameters` configures enabled plugins and contributes to
  cache identity. CA maps default to a 5000-pair limit and avoid dense NxN
  distance arrays. Backbone torsion selection is configurable.
- DSSP computes only requested frames and optionally exports per-residue
  one-hot assignments and occupancies with residue mapping.
- OpenMM execution validates plans before importing optional dependencies,
  seeds the integrator, writes checkpoints and preparation provenance, validates
  generated trajectories, and writes a per-job analysis project.
- System inspection uses the scientific registry as its authoritative
  catalogue. ProposalCatalog entries remain labeled proposals; AnalyzerCatalog is its
  compatibility alias. Implemented-feature views derive from FeatureRegistry.
- Packaging declares core dependencies and optional test/DSSP/simulation/
  Parquet extras. A GitHub Actions test workflow is provided but has not been
  executed remotely. Git initialization is blocked by the current filesystem permissions; the
  pre-refactor source snapshot is retained locally.

## Configure features

```json
{
  "analysis": {
    "plugins": ["protein.ca_distances_contacts", "protein.secondary_structure_dssp", "protein.backbone_torsions"],
    "plugin_parameters": {
      "protein.ca_distances_contacts": {
        "selection": "protein and name CA and resid 1:30",
        "contact_cutoff_angstrom": 8.0,
        "max_pairs": 5000
      },
      "protein.secondary_structure_dssp": {"per_residue": true},
      "protein.backbone_torsions": {"selection": "protein", "torsions": ["phi", "psi"]}
    }
  }
}
```

## Commands

```bash
/Users/nirav/ds-env/bin/python -m molscope.check_environment
/Users/nirav/ds-env/bin/python -m pytest -q
/Users/nirav/ds-env/bin/python -m compileall -q src/molscope
/Users/nirav/ds-env/bin/python -m molscope.run_project --project project.json --out-dir results/test
```

The discovery runner needs independent trajectories for representation
stability. Do not label duplicated trajectories or arbitrary segments as
independent replicas. Predictive classification additionally requires enough
independent trajectories per class for the requested folds.

## Real-data smoke check

Three frames from the local mdxplain `data/2RJY` topology/trajectory were tested:
selected CA distances/contacts produced a 3 x 33 table (including identity
columns), and residue DSSP produced 3 x 198. Backbone torsions explicitly
reported not applicable because the supplied topology lacks bond information.
This exercises extraction only; it does not establish benchmark performance.

## Remaining release and scientific work

- Actual matched mdxplain execution is blocked: `mdtraj` is absent and the
  isolated `.benchmark-env` dependency installation failed because PyPI DNS
  resolution is unavailable in this session. Real 2RJY input data exists.
- Current mdxplain distances use closest-heavy residue distance; MolScope CA
  distance is a different observable. A matched adapter must align atom/pair
  definitions before comparing values or performance.
- OpenMM execution and recovery passed short real smoke checks in addition to
  automated execution tests. No long production simulation or general force-field
  validation was performed; see [scaling evidence](scaling.md).
- Validated PBC imaging, broad membrane/polymer/ligand/nucleic chemistry,
  full scientific-plugin streaming parity, permutation/tree explanation, and interactive
  structure visualization remain additional development tracks.
- Reference fixtures currently provide limited wrapper/convention checks;
  independent multi-system scientific validation remains necessary.
- Extractor timings are retained, but allocating family runtime to individual
  features requires a declared cost model before use in SAM-MFP.

Keep these boundaries visible during user testing; unavailable analyses retain
explicit states and are not replaced with numeric zeroes.
