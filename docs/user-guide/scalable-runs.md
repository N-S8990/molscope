# Chunked trajectory runs

The streaming workflow stores extracted features in checksummed CSV chunks and
analyzes those chunks in bounded batches. Frame count does not determine the
size of the retained numerical feature matrix. Feature count, the training
sample size, and batch size still determine memory requirements. Disk usage
grows with frames × features, and CSV can be large.

This is a separate, deliberately limited workflow from `run_project`. It does
not yet provide the full scientific plugin catalog, predictive ML, replica
stability, or SAM-MFP prioritization. The regular project workflow remains the
route for those capabilities and their applicability checks.

## Extract a trajectory

To preserve project identities and system profiles, use the limited project
entry point. Each trajectory receives its own feature store, identity record,
report, and independently trained state model:

```bash
python -m molscope.stream_project --project my_project.json \
  --out-dir results/stream_project --geometry-selection 'protein' \
  --sample-rows 2000 --batch-rows 512 --memory-budget-mb 512 \
  --imaging-policy as_stored_nonperiodic
```

This reads frame limits and stride from `analysis.generic_feature_parameters`
and C-alpha parameters from the existing plugin configuration. Geometry is an
explicit streaming baseline using the CLI selection (default `all`); it does
not reproduce every generic batch feature. Other requested plugins receive
explicit unsupported assessments; unsupported configured parameters are
rejected. Project labels are preserved as metadata, not used as features or
targets. State 0 in one trajectory is not necessarily state 0 in another.
The project wrapper requires a fresh output directory and does not resume;
individual extraction stores support the resume command below.

The local 2RJY smoke fixture is configured for 100 frames at stride 5. Its paths
refer to the local mdxplain data checkout; update them on another machine:

```bash
python -m molscope.stream_project \
  --project examples/local_2rjy_stream_project.json \
  --out-dir results/2rjy_stream_project --geometry-selection protein \
  --sample-rows 100 --batch-rows 25 --chunk-size 25 --states 3 \
  --memory-budget-mb 512 --imaging-policy as_stored_nonperiodic
```

Run from the repository using the project's Python environment:

```bash
python -m molscope.streaming \
  --topology input/system.pdb --trajectory input/trajectory.dcd \
  --out-dir results/stream_features \
  --chunk-size 1000 --stride 1 --max-pairs 5000 \
  --geometry-selection 'protein' \
  --memory-budget-mb 512 \
  --imaging-policy as_stored_nonperiodic
```

The initial implementation extracts geometry and C-alpha distances/contacts.
Applicability assessments are stored in `manifest.json`; unavailable features
are not replaced by zero. `--max-pairs` limits the permitted pair expansion;
exceeding the bound is an error rather than silent pair truncation. Select a
smaller atom group when needed. `--max-frames` can cap a smoke test, while
`--stride` samples every Nth original frame. Preserve full temporal resolution
when it is scientifically required.

The sole supported imaging policy is **as_stored_nonperiodic**. It uses stored
coordinates without minimum-image distances, unwrapping, or automatic repair.
Input molecules must already be made whole and consistently imaged if the
measurements require it. A periodic box in the input does not activate PBC
handling. Unprepared wrapped trajectories can yield misleading geometry and
contacts. Record any preprocessing with the study.

Extraction commits each chunk and updates its manifest atomically. To resume,
repeat the exact command with `--resume`. Source fingerprints, configuration,
and committed chunk checksums must match. Resume skips committed work; it does
not append new frames to a completed study with changed input/configuration.

## Analyze committed chunks

```bash
python -m molscope.stream_analysis \
  --source results/stream_features --out-dir results/stream_analysis \
  --batch-rows 512 --sample-rows 2000 --components 3 --states 2 \
  --seed 42 --memory-budget-mb 512
```

Extraction must be complete. All frames contribute to running feature moments
and state populations. A seeded uniform reservoir trains scaling, PCA, and
K-means; the trained model assigns every frame in a second bounded pass.
These states are descriptive candidates learned from a sample, not validated
physical states or supervised predictions. Rare states can be missed by the
sample. Constants in the training sample are excluded from PCA, even if they
vary elsewhere. Nonfinite values fail analysis; there is no implicit imputation.

Outputs include `report.html`, `state_populations.png`, `feature_summary.csv`,
`state_populations.csv`, `sample_model.npz`, and chunked `assignments/*.csv`
with frame identities, PCA coordinates, and states. `analysis_summary.json`
records configuration, hashes, counts, scope, and limitations. It is committed
last as the completion marker. This report is aggregate and does not embed
every frame in an interactive plot or generate a molecular 3D viewer.

Analysis requires a fresh output directory. Completed and partial outputs are
never overwritten. If interrupted, rerun analysis into a new directory; the
extracted feature store can be reused. There is no analysis resume checkpoint.

The analyzer's numerical workspace estimate is
`8 × feature_count × (4 × batch_rows + 8 × sample_rows)` bytes. The extraction
manifest separately records its geometry/pair array estimate and approximate
CSV storage size. These guards are estimates, not operating-system RSS caps;
Python, readers, plotting, CSV parsing, and numerical libraries add overhead.
Start with a representative small run and measure RSS and disk growth before
committing to 500,000 or 1,000,000 frames. Use fewer pairs or a smaller reservoir
for wide feature sets. A larger training sample may improve coverage but
increases PCA cost and memory.

## Reproduce an analysis scaling benchmark

```bash
python -m molscope.benchmark_stream --frames 100000 \
  --out-dir results/synthetic_100k
```

Choose a new output directory for each run. The benchmark writes
`benchmark.json` with frame count, analysis wall time, and process peak RSS.
Its three synthetic features exercise the CSV/reservoir/assignment pipeline.
It does not benchmark molecular trajectory decoding, atom/pair scaling,
scientific correctness, simulation generation, or full-plugin execution.
Even a successful million-row synthetic run establishes only that measured
scope; it does not establish universal million-frame molecular capacity.

In the local November 2026 test, the three-feature analysis processed 10,000
rows in 0.406 seconds with 268.703 MiB process peak RSS and 1,000,000 rows in
4.422 seconds with 274.703 MiB process peak RSS. These are local synthetic
measurements, including process/library overhead in peak RSS, not performance
guarantees for real trajectories or wider feature sets.
