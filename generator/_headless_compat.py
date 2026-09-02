"""Headless compatibility layer for Sionna notebook-preview imports.

Sionna RT can import Jupyter preview dependencies while the generator is running
as a CLI, server, or batch job.  This module provides lightweight
``ipywidgets``/``pythreejs``/``IPython.display`` stubs for those headless runs so
the Sionna import path remains available with notebook UI packages treated as
optional.

The stubs give preview code a clear failure mode: importing the preview modules
succeeds, and constructing a preview widget raises an explicit
preview-disabled error.  ``ORCHAV_HEADLESS`` controls the policy, with common
notebook runtimes keeping their real widget stack by default.
"""

from __future__ import annotations

import os
import sys
import types
from collections.abc import Callable
from typing import NoReturn

_TRUTHY = {"1", "true", "yes", "on"}
_FALSEY = {"0", "false", "no", "off"}
_NOTEBOOK_MARKERS = ("JPY_PARENT_PID", "COLAB_RELEASE_TAG", "KAGGLE_KERNEL_RUN_TYPE")


def _env_flag(name: str) -> bool | None:
    """Parse an optional boolean environment flag."""
    raw = os.environ.get(name)
    if raw is None:
        return None

    normalized = raw.strip().lower()
    if normalized in _TRUTHY:
        return True
    if normalized in _FALSEY:
        return False
    return None


def _should_enable_headless_compat() -> bool:
    """Return whether headless preview stubs should be installed."""
    explicit = _env_flag("ORCHAV_HEADLESS")
    if explicit is not None:
        return explicit
    # Default to headless outside common notebook runtimes.  Notebooks should
    # keep their real widget stack so Sionna previews can work normally.
    return not any(os.environ.get(marker) for marker in _NOTEBOOK_MARKERS)


def _raise_preview_disabled(*_args: object, **_kwargs: object) -> NoReturn:
    """Fail clearly if code reaches notebook-preview functionality headlessly."""
    raise RuntimeError(
        "Sionna notebook preview support is disabled for headless ORCHAV runs. "
        "Set ORCHAV_HEADLESS=0 before importing generator if you need preview widgets."
    )


class _DynamicStubModule(types.ModuleType):
    """Module stub whose attributes all fail with the preview-disabled error."""

    def __getattr__(self, _name: str) -> Callable[..., NoReturn]:
        return _raise_preview_disabled


def enable_headless_preview_compat() -> None:
    """Install notebook-preview stubs for headless generator runs.

    Call this before importing Sionna modules in CLI, server, or batch contexts
    that treat notebook previews as optional.  The function is conservative: it
    leaves real notebook packages alone if they are already imported.
    """
    if not _should_enable_headless_compat():
        return

    # Avoid mixing real notebook packages with stubs in partially initialized sessions.
    if "ipywidgets" in sys.modules or "pythreejs" in sys.modules:
        return

    # Sionna imports module names from these packages.  The stubs satisfy those
    # imports but make any real preview use fail loudly.
    widgets_module = _DynamicStubModule("ipywidgets.widgets")

    embed_module = types.ModuleType("ipywidgets.embed")
    embed_module.embed_snippet = _raise_preview_disabled

    ipywidgets_module = types.ModuleType("ipywidgets")
    ipywidgets_module.widgets = widgets_module
    ipywidgets_module.embed = embed_module
    ipywidgets_module.__all__ = ["widgets"]

    pythreejs_module = _DynamicStubModule("pythreejs")

    existing_display = sys.modules.get("IPython.display")
    if existing_display is None:
        existing_display = types.ModuleType("IPython.display")
        existing_display.display = _raise_preview_disabled
        sys.modules.setdefault("IPython.display", existing_display)

    ipython_module = sys.modules.get("IPython")
    if ipython_module is None:
        ipython_module = types.ModuleType("IPython")
        ipython_module.get_ipython = lambda: None
        ipython_module.version_info = (0, 0)
        ipython_module.__version__ = "0"
        ipython_module.display = existing_display
        sys.modules.setdefault("IPython", ipython_module)

    sys.modules.setdefault("ipywidgets", ipywidgets_module)
    sys.modules.setdefault("ipywidgets.widgets", widgets_module)
    sys.modules.setdefault("ipywidgets.embed", embed_module)
    sys.modules.setdefault("pythreejs", pythreejs_module)
