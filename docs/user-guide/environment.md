# Dedicated testing environment

`environment.yml` defines Python 3.12, the analysis stack, DSSP-capable
MDAnalysis, Parquet storage, OpenMM/PDBFixer, pytest and FreeSASA. It is a
version-bounded specification; it is not yet a resolved lockfile. mdxplain
should use its own environment for comparisons.

From the MolScope repository root on a network-enabled terminal:

```bash
conda env create --prefix ./\.conda/molscope-test --file environment.yml
conda activate ./\.conda/molscope-test
python -m pip install --no-deps -e .
python -m molscope.check_environment --output environment_snapshot.json
python -m pytest -q
python -m molscope.smoke_test --out-dir results/conda_smoke
python -m py_compile molscope/*.py
```

Choose a new smoke output directory for each run. The smoke command generates
synthetic frame features for software validation and checks all six report
figures plus Markdown/JSON reports; it does not run molecular dynamics or
validate molecular extraction. Use `run_project` with actual independent
trajectories for scientific analysis.

After a successful install, capture the solved environment:

```bash
conda env export --no-builds > environment.resolved.yml
python -m pip freeze > environment.pip-resolved.txt
```

The initial environment creation attempt in the restricted agent session
failed due to repository DNS resolution and global cache permissions. A retry
with Conda plugins disabled also encountered repository DNS failures. The
new environment has therefore not been validated. The existing `ds-env`
remains unchanged and can run the separate smoke check.
