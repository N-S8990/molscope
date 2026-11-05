> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 25: Matched-method benchmarking

`molscope.benchmarking` is a tool-neutral comparison harness. It does not
execute mdxplain or any baseline automatically: that would conceal important
differences in feature selection, preprocessing, and resource configuration.
Instead, it requires a shared trajectory identity manifest and records an
explicit status for every declared method.

Start with `examples/benchmark.example.json`, then run:

```bash
python -m molscope.run_benchmark \
  --benchmark benchmark.json \
  --out-dir results/benchmark
```

The shared manifest must have unique `system_id`, `ensemble_id`, `replica_id`,
and `trajectory_id` columns (or an explicitly declared alternative). Every
method that supplies `input_manifest` must match that identity set exactly.

Each method result CSV must contain `metric_id,value`; optional `subset` and
`notes` columns are retained. Metrics must be declared in advance in the
benchmark JSON. The harness writes method readiness, normalized metrics,
declared metric definitions, a JSON summary, and a cautious Markdown report.

## Matched protocol declaration

Declare the shared trajectory preprocessing, frame range/stride, imaging
policy, conventional feature scope, clustering settings, random seeds, and
resource-measurement policy in `shared_protocol`. Each method may also record
its own `protocol` object (runner/version/export convention). The harness
preserves these declarations and reports whether a method protocol was
supplied; it does not infer equivalence from a matching file name.

`examples/benchmark.example.json` is a ready-to-edit matched comparison
template. It deliberately leaves mdxplain results pending: run mdxplain in its
own environment, export only the predeclared metrics to `results_csv`, then
rerun the collation command. A recognizable checkout alone is never counted as
an executed benchmark.

For `kind: "mdxplain"`, a checkout path is inspected only for recognizable
project files. The harness never imports mdxplain and reports unavailable or
pending results explicitly. This keeps comparisons reproducible and avoids a
false claim that a method was evaluated when it was not.
