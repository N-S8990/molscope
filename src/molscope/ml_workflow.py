"""Compatibility entry point; implementation: molscope.analysis.predictive.workflow."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _module = _importlib.import_module("molscope.analysis.predictive.workflow")
    _module.main()
else:
    _sys.modules[__name__] = _importlib.import_module("molscope.analysis.predictive.workflow")
