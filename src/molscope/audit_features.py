"""Compatibility entry point; implementation: molscope.cli.audit_features."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    _module = _importlib.import_module("molscope.cli.audit_features")
    _module.main()
else:
    _sys.modules[__name__] = _importlib.import_module("molscope.cli.audit_features")
