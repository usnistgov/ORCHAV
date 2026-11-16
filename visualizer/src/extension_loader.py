"""Load bundled runtime features and optional external registrations.

Bundled features are part of the installed distribution. External packages can
add separate registrations without being required to activate bundled ones.
The curated release profile selects the bundled package list.
"""

from __future__ import annotations

from importlib import import_module

_BUNDLED_EXTENSION_PACKAGES = (
    "visualizer.src.sensing.runtime",
    "visualizer.src.experiments.runtime",
)
_EXTENSION_PACKAGE = "visualizer_extensions"
_loaded = False


def ensure_external_extensions_loaded() -> None:
    """Run the installed visualizer extension bootstrap at most once."""
    global _loaded
    if _loaded:
        return

    for name in (*_BUNDLED_EXTENSION_PACKAGES, _EXTENSION_PACKAGE):
        try:
            package = import_module(name)
        except ModuleNotFoundError as exc:
            if exc.name != name:
                raise
            continue
        register = getattr(package, "register", None)
        if not callable(register):
            raise TypeError(f"{name}.register must be callable")
        register()
    _loaded = True
