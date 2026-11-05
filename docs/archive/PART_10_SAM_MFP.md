> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 10 — SAM-MFP integration

`molscope.prioritize_features` is the first end-to-end target-free
prioritization workflow. It combines feature-quality diagnostics with PCA
representation salience and leave-one-trajectory-out stability.

Its output is a Pareto candidate set, not a single feature-importance score.
The objectives are effective sample size, replica consistency, occupancy
entropy, uniqueness, representation salience, and representation stability.
Representation stability is based on leave-one-trajectory-out rank variation,
normalized by the candidate-feature rank range.
Measured computational cost is included only when it is present for every
frame-resolved candidate; otherwise the report states that it was excluded.

This establishes the target-free core of SAM-MFP. Supervised predictive
importance will be added separately and must not be presented as scientific or
causal relevance.
