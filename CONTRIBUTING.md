# Contributing to MolScope

Thank you for your interest in contributing to MolScope! MolScope is a scientific Python platform for inspecting molecular dynamics (MD) trajectories, extracting biophysical features, and rigorously prioritizing measurements with system-aware machine learning.

To maintain scientific integrity and code quality across contributors, please follow these guidelines.

---

## 1. Development Environment Setup

MolScope requires Python 3.10 or newer (tested on 3.10, 3.11, 3.12, and 3.13).

```bash
# Clone the repository
git clone https://github.com/your-org/molscope.git
cd molscope

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip and install editable package with development extras
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Verify your environment:
```bash
python -m pytest -q
```

---

## 2. Coding Standards and Architecture

### 2.1 Python Conventions
- **Python 3.10+ idioms**: Use modern syntax such as `from __future__ import annotations`, built-in generics (`list[str]`, `dict[str, Any]`), and union operator syntax (`int | None`).
- **Type Annotations**: Annotate public function signatures, dataclass fields, and module interfaces.
- **Explicit Imports**: Never use wildcard imports (`from module import *`). Avoid circular imports by placing abstract protocols in `molscope.core.contracts`.

### 2.2 Scientific Rigor Principles
- **No Silent Imputation or Zero-Filling**: Inapplicable features (e.g., Glycine sidechains, terminal dihedrals) must carry an explicit `ApplicabilityStatus.UNAVAILABLE` with a documented reason. Never fill unmeasured physical quantities with zeros.
- **Autocorrelation & Grouping Guardrails**:
  - For multi-trajectory datasets, cross-validation must strictly group by trajectory (`GroupKFold` or trajectory-stratified splits). **Zero frame leakage across train/test splits**.
  - For single continuous trajectories, use Purged K-fold CV with an embargo boundary to prevent temporal autocorrelation leakage.
- **Separation of Evidence**: Descriptive clustering/PCA rankings, supervised ML feature importances, and multi-objective Pareto candidates (SAM-MFP) must remain clearly partitioned in outputs and reports.

### 2.3 Modular Structure
- `molscope.core`: Base contracts, dataclasses, project specifications, and workspace managers.
- `molscope.features`: Biophysical feature extractors and capability-gated plugins.
- `molscope.analysis`: Representation discovery, TICA, clustering, and statistical rigor evaluation.
- `molscope.execution`: Orchestration pipelines (e.g. `MDContactMLPipeline`, streaming).
- `molscope.io`: Topology/trajectory validation and on-the-fly PBC coordinate transformations.
- `molscope.reporting`: Report generators, publication figures, and 3D visualization exporters.
- `molscope.cli`: Command-line dispatchers.

---

## 3. Testing Requirements

Every bug fix, optimization, and new feature must include tests.

### 3.1 Running Tests
Before submitting any changes, run the test suite locally:

```bash
# Run the fast test suite
python -m pytest -q

# Run full test suite with coverage report
python -m pytest --cov=molscope --cov-report=term-missing tests/
```

### 3.2 Test Organization
- `tests/unit/`: Fast unit tests for isolated classes, functions, and feature math.
- `tests/integration/`: End-to-end integration tests using synthetic and small molecular fixtures.
- `tests/reference/`: Ground-truth comparisons against established reference packages (e.g., MDAnalysis, DSSP).

### 3.3 Zero Regression Policy
- All existing 320+ unit, integration, and reference tests must pass.
- New features should add corresponding unit tests verifying edge cases (e.g. empty inputs, missing topologies, single vs. multi-trajectory inputs).

---

## 4. Code Formatting and Linting

Before committing, verify code formatting, syntax, and diff sanity:

```bash
# Verify byte-compilation across all modules
python -m compileall -q src/molscope tests

# Check syntax and undefined variables with flake8
flake8 src/molscope tests --count --select=E9,F63,F7,F82 --show-source --statistics

# Verify clean whitespace and diff formatting
git diff --check
```

---

## 5. File-by-File Git Commit Policy

To ensure high traceability and maintainable history:

1. **Atomic, File-by-File Staging**:
   - Stage specific files explicitly: `git add <file1> <file2>`.
   - **Do NOT use `git add .` or `git commit -a`**, which risks accidentally committing temporary files, artifacts, or untracked test outputs.
2. **Commit Message Conventions**:
   Follow conventional commit prefixes:
   - `feat:` New user-facing capability or analysis plugin.
   - `fix:` Bug fix in feature extraction, pipeline logic, or reporting.
   - `docs:` Documentation, tutorials, or docstring updates.
   - `test:` Adding or updating unit/integration tests.
   - `refactor:` Code refactoring with no functional change.
   - `ci:` Updates to CI workflows or build scripts.
3. **No Artifacts or Large Binaries in Git**:
   - Trajectory files (`.dcd`, `.xtc`), binary data (`.parquet`), run logs, figures (`.png`, `.html`), and study workspaces belong in external directories (e.g. study workspaces or results folders), never in git commits.
   - Ensure `.gitignore` covers any scratch directories or intermediate files.

---

## 6. Pull Request Process

1. Fork the repository and create a descriptive branch name (`feature/new-plugin-name` or `fix/pbc-wrapping-issue`).
2. Implement changes following coding standards and test requirements.
3. Ensure CI checks pass on all supported Python versions (3.10–3.13) and operating systems (Ubuntu, macOS).
4. Open a Pull Request detailing the purpose of the change, scientific validation evidence, and test coverage.
