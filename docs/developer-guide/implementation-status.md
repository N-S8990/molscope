# Implementation status

The canonical package is organized under `src/molscope/`; public legacy module
paths remain compatibility adapters. See [architecture](architecture.md) and
[project map](project-map.md). Historical development-stage status tables are no
longer active work assignments.

The configured batch workflow implements validation/profiling, applicable
extraction, quality diagnostics, target-free representation/state discovery,
SAM-MFP prioritization, optional grouped predictive ML, and reports.

The streaming workflow implements a limited feature subset with chunked storage,
checked extraction resume, bounded-sample state discovery, and all-frame
assignments. It does not provide full plugin parity, predictive ML, or SAM-MFP.

OpenMM execution supports explicit execution and checked recovery into fresh
segments. Real short execution and recovery smoke tests passed; production
protocol validation remains system-specific.

The scientific registry is authoritative for implemented features. ProposalCatalog
contains advisory family proposals; AnalyzerCatalog is a compatibility alias,
not a second implemented-feature registry.

See [test readiness](../validation/test-readiness.md),
[reference validation](../validation/reference-validation.md), and
[scaling measurements](../validation/scaling.md) for evidence and limits.

Architecture changes preserve scientific behavior. They do not establish
validated PBC imaging, broader molecular coverage, million-frame molecular
capacity, or superiority to mdxplain.
