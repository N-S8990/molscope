# Catalogue and compatibility audit

The former flat package contained public command modules as well as scientific
implementations. Low reference counts were not sufficient evidence for deletion.
The architecture migration retains old module entry points as compatibility
adapters and moves implementation into responsibility-based packages.

## Resolved structure issues

- Canonical source lives under `src/molscope/` with package discovery.
- User-facing commands are documented in the [CLI reference](../user-guide/cli.md).
- Numbered development notes live in the archive; current guides have topical names.
- The scientific registry remains authoritative for implemented methods.
- ProposalCatalog is explicitly advisory. AnalyzerCatalog remains its compatibility
  alias; implemented-feature views derive from FeatureRegistry.

Do not remove compatibility adapters without a declared deprecation policy and
tests for remaining public commands. Do not treat a proposal as an implemented
feature or infer applicability from an unvalidated family name.

## Remaining boundaries

Batch/streaming feature parity and scientific validation are separate tasks.
The streaming subset must remain explicit. PBC-aware measurements and broader
molecular-family coverage require scientific definitions, reference evidence,
and capability gates before orchestration integration.

No changes belong in the separate MD-ContactML legacy repository.
