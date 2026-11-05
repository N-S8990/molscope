> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 24: Configured Project Runner

The project runner is the main end-to-end interface for the new platform:

```bash
python -m molscope.run_project \
  --project my_project.json \
  --out-dir results/my_study
```

It reads a validated project specification, applies its generic extractor
settings and selected feature-family plugins, preserves trajectory and replica
identity, and writes the analysis outputs.

## What it runs

Every configured run includes:

1. Generic and enabled applicable feature extraction.
2. Feature validity, missingness, temporal, replica, uncertainty, and cost
   diagnostics.
3. Target-free ML representation discovery.
4. ML clustering for candidate conformational states.
5. SAM-MFP Pareto prioritization, with rank-stability information.
6. A Markdown and JSON researcher report plus two descriptive PNG figures.

It does not convert unavailable analyses into zeros. Current homogeneous
feature-matrix processing rejects incompatible schemas across molecular systems;
run heterogeneous systems independently until an explicit cross-system missing-
data policy is scientifically validated.

## Predictive ML

When a declared target is scientifically valid, request it explicitly:

```bash
python -m molscope.run_project \
  --project my_project.json \
  --out-dir results/condition_model \
  --prediction-task condition_classification
```

This adds grouped, trajectory-aware supervised validation plus a predictive
LOFO feature ranking. The core target-free analyses still run, and predictive
results remain separate from state-explanatory ranking and SAM-MFP Pareto
priority evidence. Frames are never treated as independent biological
replicates. Prefer an explicit trajectory-level label declared in `project.json`
over legacy metadata; see `PART_26_LABELS_AND_RANKING.md`.

## Outputs

The output directory contains machine-readable CSV tables for feature quality,
representation salience, state results, SAM-MFP candidates, and any supervised
results. It also contains `research_report.md`, `research_report.json`,
`sam_mfp_tradeoff.png`, `state_populations.png`, `feature_quality_tradeoff.png`,
and `representation_salience.png`. When labels and a prediction task are
configured, additional predictive-ranking CSV tables are written; a declared
label definition is retained in `label_definitions.json`. The figures are descriptive visualizations,
not evidence of mechanism or causality.

## Scope today

The runner coordinates the currently implemented and tested feature families;
it is not a claim that every feature in the long-term catalogue is implemented
or validated. External engine energy terms are only accepted through their
exact frame/time alignment path, rather than being approximated by this runner.
