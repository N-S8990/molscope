"""Source-checkout bootstrap; installable implementation lives in src/molscope."""
from pathlib import Path as _Path

__path__ = [str(_Path(__file__).resolve().parent.parent / "src" / "molscope")]
_source = _Path(__path__[0]) / "__init__.py"
exec(compile(_source.read_text(), str(_source), "exec"), globals())
