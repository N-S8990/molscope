# Reproducible benchmarks

`configs/matched.example.json` declares a matched-input comparison. Copy it into
a study workspace and supply actual manifests, tool locations, protocols, and
measured metrics before running:

```bash
python -m molscope benchmark --benchmark /path/to/study/benchmark.json --out-dir /path/to/study/runs/benchmark-001
python -m molscope.benchmark_stream --frames 1000000 --out-dir /path/to/study/runs/synthetic-scale-001
```

Implementations live in `src/molscope/benchmarking/`; benchmark definitions live
here. Generated datasets and reports belong in study workspaces. The synthetic
scaling harness does not benchmark molecular extraction or scientific accuracy.
