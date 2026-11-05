> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 23 — Reproducible analysis sessions

`MDContactMLPipeline.save_session("session_dir")` writes the frame-feature
matrix, identity manifest, pipeline configuration, and—when already run—the
result tables, Markdown/JSON report, and PNG figures.

Frame tables are stored as Parquet when a compatible pandas engine is
installed, otherwise as CSV. `session_storage.json` records the representation
used, so the session is portable across the base installation. Older CSV-only
sessions remain readable.

Restore inputs and configuration using:

```python
pipeline = MDContactMLPipeline.load_session("session_dir")
pipeline.run()
```

The saved session preserves reproducible inputs and configuration; it does not
silently treat old output tables as a fresh rerun.
