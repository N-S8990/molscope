# Actual trajectory extraction

The local real-data configuration is `examples/local_2rjy_project.json`.
It uses the existing 2RJY topology and trajectory, analyzes 100 frames with
stride 5, and extracts generic features, CA distances/contacts, and residue DSSP.

```bash
/Users/nirav/ds-env/bin/python -m molscope.run_project \
  --project examples/local_2rjy_project.json \
  --extract-only --out-dir results/2rjy_initial_extraction
```

This command has completed successfully. Outputs include stored frame features,
the identity manifest, extraction evidence and session configuration. Inspect
`session_storage.json` for the actual table filenames (Parquet or CSV fallback).

For full discovery, supply at least two actual independent trajectories in the
project and omit `--extract-only`. Do not duplicate one trajectory to satisfy
the replica requirement. This initial extraction does not run ML or generate
the discovery report. The new Conda environment remains uninstalled because
network access failed; the verified interpreter above is available.

Generated synthetic results and disposable caches were archived under
`.cleanup-archive/` instead of deleting regression tests or molecular fixtures.
