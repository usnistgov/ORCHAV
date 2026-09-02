"""Load optional visualizer registrations from an external package.

The public visualizer owns extension contracts, but it does not know which
private or third-party features may be installed.  An external distribution
can provide a top-level ``visualizer_extensions`` package with a ``register``
callable.  The development repository ships that package separately; the
curated public export simply omits it.
"""

from __future__ import annotations

from importlib import import_module

_EXTENSION_PACKAGE = "visualizer_extensions"
_loaded = False


def ensure_external_extensions_loaded() -> None:
    """Run the installed visualizer extension bootstrap at most once."""
    global _loaded
    if _loaded:
        return

    try:
        package = import_module(_EXTENSION_PACKAGE)
    except ModuleNotFoundError as exc:
        if exc.name != _EXTENSION_PACKAGE:
            raise
        _loaded = True
        return

    register = getattr(package, "register", None)
    if not callable(register):
        raise TypeError(f"{_EXTENSION_PACKAGE}.register must be callable")
    register()
    _loaded = True
