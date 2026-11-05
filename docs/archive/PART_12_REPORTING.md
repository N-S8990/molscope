> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 12 — Researcher report

`python -m molscope.build_report` assembles optional pipeline outputs into
`research_report.md` and `research_report.json`. It separates feature quality,
target-free SAM-MFP candidates, ML state discovery, and supervised predictive
results, and includes explicit interpretation limits.

It also writes `sam_mfp_tradeoff.png` and `state_populations.png` when the
corresponding tables are supplied. When the separate ranking tables are
available, it writes `predictive_feature_ranking.png` and
`state_explanatory_feature_ranking.png`. The former shows grouped predictive
association rather than causality; the latter shows separation among derived
candidate states rather than state validation. All report figures are
descriptive visualizations, not evidence of causal mechanism.
