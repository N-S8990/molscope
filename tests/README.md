# Test organization

- `unit/`: contracts, scientific calculations, policies, and guarded adapters.
- `integration/`: configured workflows, file/session storage, extraction,
  reporting, simulation execution wiring, and streaming orchestration.
- `reference/`: numerical comparisons with trusted tools under declared policies.
- `fixtures/`: small, portable molecular inputs shared by test groups.

Run every group with `python -m pytest -q` from the repository root. Legacy
module imports are intentionally retained in some tests to verify compatibility
with existing scripts. Mocked OpenMM execution tests do not replace real
simulation validation; reference fixtures do not establish universal scientific
validity or PBC support.
