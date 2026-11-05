"""Compatibility entry point; implementation: molscope.execution.stream_project."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _module = _importlib.import_module("molscope.execution.stream_project")
    _module.main()
else:
    _sys.modules[__name__] = _importlib.import_module("molscope.execution.stream_project")
