# Project templates and examples

- `visualize_2rjy_interpretability.ipynb`: Interactive Jupyter walkthrough of MolScope's interpretability and 3D visualization suite (2D FES landscapes, cluster timelines, Markov transition matrices, Boolean decision tree rules, and PyMOL/ChimeraX exports).
- `generic_project.example.json`: full project schema including optional labels
  and simulation configuration. Replace placeholder inputs and force fields.
- `study/project.json`: minimal portable study declaration.
- `benchmark.example.json`: compatibility path to the benchmark configuration.

Use `python -m molscope init /path/to/study --project-id my_study` to create a
fresh workspace with input, preparation, simulation, run, and cache directories.
Do not write large trajectories or generated reports into this directory.

Ignored `local_*.json` links, if present, reference machine-specific studies;
they are not distributed examples.
