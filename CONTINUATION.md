# MolScope continuation handoff

## Project location and boundary

- New platform root: `/Users/nirav/Docs/molscope`
- Python package: `src/molscope/` (checkout bootstrap: `molscope/`)
- Legacy project retained separately: `/Users/nirav/Docs/MD-ContactML`

Work on new studies belongs in MolScope. Do not move, rewrite, or use the
legacy 1A22 temperature-classification workflow as the architecture for new
platform development.

## Intended workflow

```text
project.json
  → input validation and system profiling
  → applicable generic/plugin feature extraction
  → quality diagnostics and target-free representation/state discovery
  → SAM-MFP Pareto prioritization
  → Markdown/JSON/PNG report
  ↘ grouped predictive ML only for a declared valid target
```

Main entry point:

```bash
python -m molscope run --project my_project.json --out-dir results/my_study
```

Read first:

1. `README.md`
2. `docs/developer-guide/project-map.md`
3. `docs/scientific-methods/mdxplain-comparison.md`
4. `docs/user-guide/project-runs.md`
5. `docs/developer-guide/implementation-status.md`
6. `docs/developer-guide/dead-code-audit.md`

## Current implementation state

- Capability-gated generic and domain feature extraction.
- Scientific registry records implemented plugins, applicability requirements,
  limitations, and implementation-test status.
- Feature families include protein geometry/dynamics/interactions, SASA,
  solvation shells, neutral-scope dipole, protein contact networks, and a
  nucleic-acid base-center proximity descriptor. Do not describe the latter as
  a base-pairing or stacking classifier.
- Content-addressed extraction cache (source SHA-256 + configuration + plugin
  versions) and Parquet storage with CSV fallback.
- Target-free PCA representation discovery, state discovery, feature-quality
  diagnostics, and SAM-MFP Pareto candidates.
- Markdown/JSON reports and descriptive PNGs for Pareto trade-offs, state
  populations, feature-quality trade-offs, and representation salience.
- Configured OpenMM plans can be generated and reviewed without automatically
  running MD. Execution is explicit and requires local OpenMM/PDBFixer.
- Matched benchmark harness supports MolScope, mdxplain availability checks,
  and declared baselines without importing mdxplain.

## Validation baseline

The latest completed full suite is **129 passing tests**, with 3 passing subtests
(domain-package migration, compatibility, catalogue separation, and managed study runs).
See `docs/validation/test-readiness.md` for completed checks and outstanding scientific scope.

Use the existing scientific environment in this workspace:

```bash
cd /Users/nirav/Docs/molscope
/Users/nirav/ds-env/bin/python -m pytest -q
/Users/nirav/ds-env/bin/python -m compileall -q src/molscope
```

The system Python used in one relocation check lacked `MDAnalysis`; this is an
environment dependency issue, not a test failure. Install dependencies from
`requirements.txt` in any new environment before running molecular tests.

Useful smoke commands:

```bash
python -m molscope.run_benchmark --help
python -m molscope.generate_openmm_plan --help
python -m molscope.openmm_plan --help
```

## Important design rules

- Never zero-fill unavailable analyses; use explicit applicability states.
- An extractor measures values; it does not rank scientific importance.
- SAM-MFP yields Pareto candidates, not a universal scalar importance score.
- Frame counts are not biological replicates; preserve trajectory/replica
  identity throughout quality, state, and supervised analyses.
- Keep predictive ML separate from target-free evidence and causal claims.
- Do not add a method directly to orchestration. Define its scientific method,
  applicability rule, validation requirement, registry record, extractor, and
  tests first.
- Preserve the legacy project independently; do not reintroduce it into this
  package.

## Highest-value next work

The bounded-memory extraction/analysis implementation and measured synthetic
scaling results are documented in `docs/user-guide/scalable-runs.md` and
`docs/validation/scaling.md`. Real 2RJY smoke outputs are available at
`results/2rjy_stream_analysis/report.html`. This scoped workflow does not yet
replace the full plugin/replica/SAM-MFP pipeline at large scale.

1. Reference-validate existing feature plugins against trusted tools and add
   cross-system fixtures; implementation tests are not scientific validation.
2. Add PBC-aware variants where appropriate, only with clear imaging and
   validation policies.
3. Add scientifically scoped lipid/membrane, detailed nucleic-acid,
   broader ligand, and polymer feature plugins.
4. Run matched empirical benchmarks against mdxplain and baselines, reporting
   failures and resource use as well as successes.
5. Extend shared feature definitions across batch and streaming adapters;
   the scientific registry is authoritative and ProposalCatalog is advisory.

## Repository hygiene

The canonical implementation is in `src/molscope/`. Domain packages own code;
flat compatibility modules preserve prior imports and commands. Start a new
study with `python -m molscope init PATH --project-id ID`, then configure inputs
and use `python -m molscope run --study PATH --run-id RUN_ID`.
The reorganized real extraction report is
`studies/2rjy/runs/organized_validation/report/extraction_report.html`.
Historical results remain in place to preserve provenance. Local study files
are ignored. Git initialization was denied by filesystem permissions;
`.cleanup-archive/architecture-before` preserves the pre-move source/tests.
Wheel build and off-checkout installation/import checks passed. Source snapshot
diff and text whitespace/conflict checks passed; fixed-width PDB padding was
preserved rather than stripped.

- Keep generated `__pycache__/`, `.pytest_cache/`, result directories, and
  cache directories out of version control.
- Preserve existing research outputs and their provenance; initialize new studies
  separately. See `docs/user-guide/workspaces.md`.
- Git initialization was blocked by the current filesystem permissions; a local
  pre-refactor source snapshot is retained under `.cleanup-archive/architecture-before`.
- Before handoff, run focused tests, the full suite, `py_compile`, and
  `git diff --check` where the project is under version control.
