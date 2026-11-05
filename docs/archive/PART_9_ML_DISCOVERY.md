> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 9: Core ML Discovery

Target-free ML is a core stage of MolScope. The initial implementation
learns a PCA representation and reports each feature's weighted absolute
loading (`PCA_Salience`) plus leave-one-trajectory-out rank stability.

```bash
python -m molscope.discover_features \
  --features frame_features.csv \
  --manifest frame_manifest.csv \
  --components 5 \
  --out-dir discovery_results
```

This reports representation salience only: it does not establish causality or
predictive utility for an external target. Later stages will add state discovery,
kinetic objectives, and supervised importance when a scientific target exists.
