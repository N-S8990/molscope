> Historical development note. See the [documentation index](../README.md) for current workflows and validation status.

# Part 26: Declared labels and predictive-ranking inputs

MolScope accepts researcher-supplied or externally annotated labels in the
top-level `labels` array of `project.json`. A label has a stable `label_id`, a
declared source, a type, and assignments keyed by the complete trajectory ID:

```json
{
  "label_id": "experimental_condition",
  "label_type": "categorical",
  "level": "trajectory",
  "source": "external_annotation",
  "assignments": {
    "system_01:ensemble_01:replica_01": "control"
  }
}
```

Only `trajectory` labels are accepted. Frame labels are rejected because a
frame-level label can leak temporal information into a trajectory-grouped
prediction. State labels discovered by MolScope are not accepted as project
labels or predictive targets; they remain target-free hypotheses.

A predictive task can reference a declared categorical label with
`target_label_id`. All trajectories must have an assigned value before the
task can run. The legacy `target_metadata_key` path remains supported for
existing project files, but a task must declare exactly one target mechanism.
Declared labels become manifest columns named `metadata_label_<label_id>` and
retain their definition/source in project validation output.
