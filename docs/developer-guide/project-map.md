# Project map

| Location | Purpose |
| --- | --- |
| `src/molscope/` | Canonical platform implementation |
| `molscope/` | Source-checkout compatibility/bootstrap |
| `tests/unit/` | Isolated contracts and numerical behavior |
| `tests/integration/` | Configured workflows and execution wiring |
| `tests/reference/` | Reference comparisons with declared conventions |
| `tests/fixtures/` | Small portable test inputs |
| `benchmarks/` | Reproducible benchmark definitions |
| `examples/` | Portable example declarations |
| `docs/user-guide/` | Running studies |
| `docs/scientific-methods/` | Methods, assumptions, and research boundaries |
| `docs/developer-guide/` | Architecture and contribution guidance |
| `docs/validation/` | Test procedures and measured evidence |
| `docs/archive/` | Historical numbered development notes |

See [Architecture](architecture.md) for ownership and
[CLI reference](../user-guide/cli.md) for supported workflows.

Generated artifacts belong in a study workspace. Existing historical outputs
retain provenance and compatibility paths; do not edit source inputs or flatten
restart segments into independent replicas. The separately retained
`/Users/nirav/Docs/MD-ContactML` project is outside this repository's scope.
