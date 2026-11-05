# Streaming validation — 2026-11-06

The streaming path is a separate, scoped alternative to the existing full
discovery pipeline. It does not establish full plugin parity at a million frames.

## Measured analysis benchmark

`python -m molscope.benchmark_stream --frames N --out-dir results/stream_scale_N`
creates three synthetic features, fits PCA and K-means on a seeded 2,000-row
reservoir, then assigns every row in 10,000-row batches.

| Rows | Analysis wall time (seconds) | Process peak RSS (MiB) |
| ---: | ---: | ---: |
| 10,000 | 0.406 | 268.7 |
| 500,000 | 2.519 | 258.0 |
| 1,000,000 | 4.422 | 274.7 |

Measurements come from separate local processes. Peak RSS includes imports and
synthetic input generation; wall time covers analysis only. These are narrow
engineering measurements, not a molecular extraction benchmark, an MD speed
benchmark, or a comparison with mdxplain. Higher feature counts, atom counts,
disk costs, and trajectory formats can change performance substantially.
Machine-readable measurements are in each run's `benchmark.json`.

## Real input and execution checks

The project-file wrapper completed a local 2RJY run with 100 selected frames,
3,785 extracted features, and four assignment chunks. Its report is at
`results/2rjy_stream_project/trajectory_00000/analysis/report.html`.
Extraction resume was also checked against committed chunks.

A 22-atom OpenMM 8.6 Reference-platform smoke produced four valid frames.
A controlled interruption halfway through a 20-step production run recovered
the remaining ten steps into a fresh segment containing two valid frames.
Evidence and provenance notes are in `results/openmm_recovery_smoke/README.md`.
These are operational smoke tests, not equilibrated scientific simulations.

Final validation: 117 tests and 3 subtests passed; Python compilation and
whitespace/conflict-marker checks passed. Git diff checks were unavailable
because this workspace is not a Git repository.

## Scientific and operational boundaries

- Streaming extraction currently covers explicit geometry selections and
  selected protein C-alpha distance/contact pairs. Unsupported requested
  plugins retain assessments and do not become zero-valued measurements.
- Coordinates are used as stored, with nonperiodic distances. Whole/imaged
  molecular coordinates must be supplied where required; this is not a new
  validated PBC implementation.
- PCA and clustering are fitted on a bounded sample. Statistics and subsequent
  state assignments cover all extracted frames. These descriptive states are
  not biological replicates, predictive labels, or causal discoveries.
- Extraction supports checked chunk resume. Analysis requires a fresh output
  directory and can be rerun from the completed extraction store.
- Memory estimates guard numeric work arrays, not total operating-system RSS.
  CSV storage can be large; the manifest estimates it before extraction.
- Million-frame molecular extraction, broad plugin parity, and a matched
  mdxplain benchmark remain separate validation tasks.
