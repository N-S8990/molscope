> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 8: Interaction-Feature Validation

The legacy interface hydrogen-bond features are retained only as historical
baseline outputs. They are not registered as validated features in the new
workflow because the legacy angle was not the D-H-A angle and its atom-name
rules do not generalize across molecular systems.

The discovery catalog now reports hydrogen-bond analysis as `missing_input`
when atomic partial charges are absent. A future implementation must use bond
connectivity, explicit donor/acceptor chemistry or charges, declared component
scopes, D-H-A geometry, and reference tests against a trusted implementation.
