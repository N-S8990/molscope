# Architecture and contribution boundaries

MolScope is organized by responsibility under `src/molscope/`.
The root compatibility package and legacy module entry points preserve existing
imports and commands while implementation lives in domain packages.

| Area | Responsibility |
| --- | --- |
| core | Project configuration, contracts, identities, provenance |
| io | Input integrity, trajectory/session storage |
| profiling | System composition and capability detection |
| simulation | OpenMM preparation, execution, checkpoint recovery |
| features | Scientific registry, extraction definitions, applicability |
| analysis | Quality, target-free discovery, predictive ML, prioritization |
| execution | Batch and streaming orchestration |
| reporting | Scientific tables, reports, and visual outputs |
| cli | User commands that delegate to domain implementations |

These boundaries do not imply that every molecular family or execution mode is
implemented. Do not create empty feature packages to suggest nonexistent scope.

## Scientific flow

`project.json → validation/profile → applicable extraction → quality/discovery → SAM-MFP → report`

Predictive analysis is an optional branch requiring an explicit valid target.
It does not replace target-free evidence. Preserve trajectory/replica identity;
frames and restart segments are not independent biological replicates.

A feature implementation owns measurement definitions, units, prerequisites,
provenance, imaging policy, and validation status. Execution adapters own how
frames are read and work is scheduled. Batch and streaming adapters must agree
on the definition of any shared observable; unsupported modes must be explicit.
The current streaming subset remains smaller than the ordinary plugin pipeline.

The scientific feature registry is authoritative for implemented methods.
Historical family proposals are not evidence of implementation or applicability.

## Making changes

1. Define the scientific observable and applicability rule.
2. Add the extractor and registry entry; record supported execution modes.
3. Verify numerical behavior against suitable reference inputs/tools.
4. Add orchestration only after the method contract is established.
5. Preserve unavailable states; never fill absent measurements with zero.
6. Keep predictive importance, state explanations, and Pareto candidates separate.
7. Run focused tests, the complete suite, compilation, and diff checks.

Tests are grouped into unit, integration, and reference checks. Small portable
fixtures belong in tests; large molecular trajectories belong in study workspaces.
A directory move is not additional scientific validation.

## Compatibility

Existing `python -m molscope.<module>` commands remain available during the
migration. New documentation uses the unified CLI where the command exists.
Keep compatibility adapters thin and route to one implementation, rather than
maintaining duplicate scientific code. The separate MD-ContactML project is not
part of this package and must not be reintroduced here.
