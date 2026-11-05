> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 7: Scientific Method and Feature Registries

`molscope.registries` records what a feature measures, its scope, inputs,
requirements, limitations, implementation, citations, and validation state.
It separately records the scientific method behind that feature.

Feature availability uses explicit states: `available`, `calculated`,
`not_applicable`, `unsupported`, `missing_input`, `failed`,
`skipped_by_user`, and `skipped_by_cost_policy`. A zero-valued feature never
stands in for one of these states.

The registry covers generic features plus every currently implemented
capability-gated plugin. Its cost entries are explicitly unmeasured estimates;
measured per-analyzer runtime can be supplied separately to the prioritization
workflow.
