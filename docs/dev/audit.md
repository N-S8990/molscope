# MolScope Architecture Audit & Codebase Status

**Date:** 2026-11-17  
**Orchestrator:** Phase 0 Reconnaissance Audit  
**Target Repository:** `molscope` (`https://github.com/N-S8990/molscope`)

---

## 1. System Architecture Summary

MolScope is structured into modular scientific and operational packages designed for molecular dynamics (MD) feature extraction, conformational representation, statistical state discovery, multi-objective feature prioritization (SAM-MFP), and predictive machine learning.

```
src/molscope/
├── core/                  # Core project schemas (ProjectSpec, SystemSpec, Schema versioning)
├── io/                    # Topology & trajectory validation (MDAnalysis, MDTraj loaders, sanitization)
├── features/              # Biophysical feature extraction
│   ├── geometry.py        # Backbone/sidechain dihedrals, C-alpha distances, contacts
│   ├── coordinates.py     # Selected Cartesian coordinates
│   ├── energetics.py      # Simulation energy logging and parsing
│   ├── protein/plugins.py # DSSP secondary structure, H-bonds, salt bridges, hydrophobic contacts
│   └── ligand/plif.py     # Protein-ligand interaction fingerprints (PLIF)
├── profiling/             # System inspection, capability matrix, and selection validation
├── analysis/              # Scientific ML and statistical rigor
│   ├── discovery/         # Target-free state discovery (PCA, CKPCA, TICA, DPA, GMM, Decision Tree rules)
│   ├── prioritization/    # SAM-MFP multi-objective Pareto ranking & baseline comparators
│   ├── predictive/        # Purged temporal block grouped cross-validation & LOFO ablation
│   └── quality/           # Statistical rigor suite: seed stability, bootstrap clustering, CV leakage audit
├── simulation/            # OpenMM simulation plan generation, execution, and checkpoint recovery
├── execution/             # Orchestration pipelines (In-memory discovery, Out-of-core streaming)
├── reporting/             # Visualizations, free energy surfaces, interactive HTML reports
└── cli/                   # Unified command-line interface entry points
```

---

## 2. Module Responsibilities and Current Test Coverage

| Module | Primary Responsibility | Test Status | Test Files |
| :--- | :--- | :--- | :--- |
| `core.project` | Strict Pydantic/dataclass schema versioning, schema migration | **Fully Tested** | `tests/unit/test_schema_versioning.py` |
| `core.workspace` | Isolated run directory layout and provenance logging | **Fully Tested** | `tests/unit/test_workspace.py` |
| `io.validation` | Topology/trajectory verification, corrupt frame handling | **Fully Tested** | `tests/unit/test_io_validation.py` |
| `features.protein` | Biophysical plugins (DSSP, H-bonds, salt bridges, torsions, SASA) | **Reference Validated** against MDAnalysis & FreeSASA | `tests/reference/`, `tests/integration/test_feature_plugins.py` |
| `features.ligand` | Protein-ligand interaction fingerprints | **Fully Tested** | `tests/unit/test_ligand_plif.py`, `tests/reference/test_reference_hbonds_plif.py` |
| `analysis.prioritization` | SAM-MFP Pareto multi-objective prioritization | **Fully Tested** against baselines (MDI, MI, DT, Gini) | `tests/scientific/test_sam_mfp_rigor.py` |
| `analysis.predictive` | Leakage-aware grouped & purged temporal block CV | **Fully Tested** (zero-leakage verified) | `tests/unit/test_predictive_workflow.py`, `tests/unit/test_ml_rigor.py` |
| `analysis.quality` | Seed stability, bootstrap cluster stability, convergence diagnostics | **Fully Tested** | `tests/unit/test_ml_rigor.py` |
| `simulation.openmm` | OpenMM plan generation, execution, segment checkpoint recovery | **Fully Tested** (OpenMM integration verified) | `tests/integration/test_openmm_simulation_pipeline.py` |
| `execution.streaming` | Bounded-memory chunked streaming extraction | **Fully Tested** | `tests/integration/test_stream_extraction.py` |
| `reporting` | Interactive HTML dashboard, 2D/3D PCA density, free energy | **Fully Tested** | `tests/unit/test_html_report.py`, `tests/unit/test_free_energy.py` |

**Overall Test Suite Status:** 322 passed, 0 failed (100% pass rate).

---

## 3. Status of Agent Priorities (Phases 1 & 2)

1. **Agent A (2RJY Real-Data Benchmark vs mdxplain):**  
   - 2RJY benchmark project configuration defined in `studies/2rjy_benchmark/project.json`.
   - Matched benchmark script implemented in `benchmarks/run_2rjy_matched_benchmark.py`.
   - Results documented in `docs/scientific-methods/mdxplain-comparison.md`.
2. **Agent B (SAM-MFP Mathematical Rigor & Baselines):**  
   - Formal specification authored in `docs/scientific-methods/sam-mfp-specification.md`.
   - Baselines implemented in `src/molscope/analysis/prioritization/baselines.py` (MDI, Gini, MI, permutation).
   - Validation tests in `tests/scientific/test_sam_mfp_rigor.py`.
3. **Agent C (Streaming/Scalable Parity):**  
   - Chunked extraction and uncommitted chunk recovery tested in `test_stream_extraction.py`.
   - Need streaming parity for SAM-MFP prioritization and replica stability.
4. **Agent D (ML Rigor & Autocorrelation):**  
   - Grouped cross-validation and purged temporal block cross-validation implemented.
   - Leakage prevention verified in `test_ml_rigor.py`.
5. **Agent E (Interpretability & Structure Mapping):**  
   - Decision Tree rule extraction implemented.
   - PDB Beta-factor importance mapping script needed for direct 3D visual exploration.
6. **Agent F (Physical Correctness & PBC):**  
   - Non-periodic coordinate assumption flagged in system profiler. Explicit PBC checks needed.
7. **Agent G (Packaging & Credibility):**  
   - Apache-2.0 `LICENSE`, `CHANGELOG.md`, `.github/workflows/ci.yml` merged.
8. **Agent H (Scope Review):**  
   - OpenMM simulation pipeline confirmed functional and modular. Scope review doc needed.
