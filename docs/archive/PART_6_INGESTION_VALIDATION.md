> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 6: Molecular Identity and Input Validation

Before feature discovery, validate the topology and trajectory together:

```bash
python -m molscope.validate_inputs \
  --topology system.pdb \
  --trajectory production.dcd \
  --output input_integrity.json
```

The report records source paths, file size and modification time, atom counts,
frame count, time ordering, coordinate finiteness, periodic-cell coverage, and
the scope of the coordinate scan. Use `--checksum` when a content digest is
needed for a reproducible archival run. This reads inputs only.

Use `--include-identities` to include the full source-preserving atom and
residue identity tables. Atom indices are topology-local identities; they are
not claimed to be universal chemical identifiers. A topology/trajectory pair
with matching atom count can still have a scientifically incorrect mapping, so
the report records compatibility but does not claim chemical equivalence.

`--max-frames` is useful for a fast diagnostic and is always recorded as a
partial scan. It must not be interpreted as a complete integrity check.
