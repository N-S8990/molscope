> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 11 — ML conformational-state discovery

The initial unsupervised state-discovery stage clusters target-free PCA
coordinates using K-means. It writes frame assignments, state centers,
per-trajectory state populations, and diagnostics for restart agreement and
between-trajectory population differences.

Coordinate rows must carry the same trajectory/replica/ensemble identities as
the supplied manifest, and transitions are ordered by `frame_order` (or
`frame_index`). Population agreement is summarized within ensembles; intended
differences between ensembles are reported separately.

It also writes successive-frame state-transition counts and conditional
probabilities, separately by trajectory and in aggregate. These are not
transition rates or a Markov-state model.

It deliberately does not call clusters physical metastable states. The chosen
state count must be justified using system knowledge, sampling, and these
diagnostics; silhouette is not proof. Frames are assignments, while independent
trajectories remain the unit for reproducibility reporting.

Use `--state-counts 2,3,4,5` to write a state-count sensitivity table. It is a
screening analysis, not automatic state-count selection. Initialization ARI
measures sensitivity to K-means initialization; it is not independent-trajectory
cluster stability.
