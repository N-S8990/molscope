> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 15 — Unified Python workflow API

```python
from molscope.pipeline import MDContactMLPipeline

pipeline = MDContactMLPipeline()
pipeline.load_feature_matrix(frame_features, frame_manifest)
pipeline.run(target_column="condition")  # omit only when no valid target exists
pipeline.write_results("analysis_results")
```

The facade currently accepts extracted features and their identity manifest. It
runs quality assessment, target-free ML, SAM-MFP, state discovery, optional
grouped supervised ML, report generation, and PNG figures.

For a homogeneous collection of trajectories, generic features can be loaded
directly:

```python
pipeline.load_trajectories(trajectory_inputs, GenericFeatureConfig(max_frames=500))
```

Pass `cache_dir=".molscope-cache"` to reuse a prior extraction safely. Cache
keys include SHA-256 fingerprints of topology and trajectory contents, generic
settings, and selected plugin versions; editing an input or changing settings
creates a new entry rather than reusing stale values.

The supplied `TrajectoryInput` identity and scalar metadata are preserved in
the manifest. Mixed feature schemas are rejected rather than zero-filled.

To run a validated JSON project specification, use
`pipeline.load_project("project.json", plugins=["protein.salt_bridges"])`.

Attach engine-produced energy data only through exact identity/time alignment:
`pipeline.add_external_energies(energy_table)`.
