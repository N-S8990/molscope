"""Report installed runtime capabilities without starting a simulation."""
from __future__ import annotations

import argparse
import importlib
import json
import platform
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path


def environment_report() -> dict:
    packages = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "MDAnalysis", "networkx",
                 "matplotlib", "freesasa", "openmm", "pdbfixer", "mdtraj", "mdxplain"):
        try:
            module = importlib.import_module(name)
            try:
                release = version("scikit-learn" if name == "sklearn" else name)
            except PackageNotFoundError:
                release = getattr(module, "__version__", "unknown")
            packages[name] = {"status": "available", "version": release}
        except ImportError as error:
            packages[name] = {"status": "unavailable", "reason": str(error)}
    return {"python": platform.python_version(), "platform": platform.platform(), "packages": packages}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", help="Optional JSON environment snapshot.")
    args = parser.parse_args()
    rendered = json.dumps(environment_report(), indent=2)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
