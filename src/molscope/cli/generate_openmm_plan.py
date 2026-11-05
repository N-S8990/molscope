"""CLI that writes, but never executes, an OpenMM plan from project.json."""

from __future__ import annotations

import argparse

from molscope.simulation.openmm import write_openmm_plan
from molscope.core.project import load_project_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a reviewable OpenMM simulation plan from an MolScope project.")
    parser.add_argument("--project", required=True, help="Project JSON containing at least one simulation_protocol.")
    parser.add_argument("--out-dir", required=True, help="Directory for openmm_plan.json and its launcher.")
    args = parser.parse_args()
    plan, launcher = write_openmm_plan(load_project_spec(args.project), args.out_dir)
    print(f"Wrote OpenMM plan to {plan}")
    print(f"Review, then explicitly run {launcher} or python -m molscope.openmm_plan --plan {plan}")


if __name__ == "__main__":
    main()
