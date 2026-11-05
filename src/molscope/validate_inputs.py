"""Compatibility entry point; implementation: molscope.cli.validate_inputs."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _module = _importlib.import_module("molscope.cli.validate_inputs")
    _module.main()
else:
    _sys.modules[__name__] = _importlib.import_module("molscope.cli.validate_inputs")
