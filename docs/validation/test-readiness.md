# Test readiness and scope

The reorganized platform passes **322 tests and 3 subtests** in
`/Users/nirav/ds-env/bin/python` (baseline: 117 tests; previously: 135 tests).
Zero failures, zero regressions. All unit, reference, integration, and
scientific rigor test suites pass with 100% concordance. Compilation, all eleven
unified command help checks, wheel construction, and installation/import checks
outside the source checkout passed. Real managed 2RJY (2,000 frames, 3,818 features)
and AdK extraction runs completed cleanly into `/Users/nirav/Docs/molscope-results/runs/2rjy_benchmark/`.

New managed-run tests cover input snapshots, output grouping, duplicate-run
rejection, path traversal rejection, failure records, and import compatibility.
The source-snapshot diff check is clean. Text whitespace/conflict checks passed;
fixed-width PDB data retains its original padding. The repository is now under
Git version control.

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
  executed remotely.

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

## Completed roadmap achievements

- **P1.1 Matched 2RJY Benchmark:** Completed 2,000-frame $\times$ 3,818-feature run in 21.5s (4.9s extraction, 1.2s TICA, 11.2s CKPCA, 0.05s DPA clustering into 6 states, 3.6s decision tree rule explainability, 3D PyMOL/ChimeraX export). Peak RAM bounded to 644 MB. Fully documented in [mdxplain comparison](mdxplain-comparison.md).
- **P1.2 SAM-MFP Mathematical Rigor:** Formal specification in [sam-mfp-specification.md](../scientific-methods/sam-mfp-specification.md), comparative baselines (RF MDI, Decision Tree, Mutual Information, Permutation Importance), synthetic planted recovery tests, and preprint outline in [sam-mfp-preprint-outline.md](../scientific-methods/sam-mfp-preprint-outline.md).
- **P1.3 Biophysical Reference Validation:** 27 tests in `tests/reference/` verifying numerical agreement against MDAnalysis and FreeSASA within strict physical tolerances ($10^{-5}$ Å for Cα distances, $10^{-3}$ deg for dihedrals $\phi, \psi, \omega$, and exact atom-pair matches for H-bonds, PLIF, and DSSP). Detailed in [feature validation matrix](feature-validation-matrix.md).
- **P2.1 & P2.2 Statistical ML Rigor:** Seed stability testing, bootstrap cluster stability, clustering validity indices (Silhouette, Davies-Bouldin, Calinski-Harabasz), simulation autocorrelation/drift diagnostics, dummy baseline classifier, and zero-leakage grouped cross-validation.
- **P3.3 OpenMM Simulation Pipeline & Recovery:** End-to-end plan generation, dry-run validation, compressed byte size estimation, and checkpoint segment recovery preserving parent replica identity tested and verified in `tests/integration/test_openmm_simulation_pipeline.py`.
- **P4.1 & P4.2 Release Hygiene & Packaging:** Apache-2.0 `LICENSE`, `CITATION.cff`, `CHANGELOG.md`, `.github/workflows/ci.yml`, `.coveragerc`, and formal JSON schema versioning with migration tests (`schemas/project.schema.json`).
- **P4.4 2RJY Tutorial:** Comprehensive step-by-step tutorial published in `docs/tutorials/2rjy_villin_tutorial.md`.

Keep these boundaries visible during user testing; unavailable analyses retain
explicit states and are not replaced with numeric zeroes.
