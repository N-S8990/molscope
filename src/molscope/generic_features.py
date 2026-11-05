"""Compatibility entry point; implementation: molscope.features.geometry.generic."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _module = _importlib.import_module("molscope.features.geometry.generic")
    _module.main()
else:
    _sys.modules[__name__] = _importlib.import_module("molscope.features.geometry.generic")
