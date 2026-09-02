"""Live actor dialogs used by the Object Management panel.

`object_panel.py` owns the tree and camera-focus workflow. This package exposes
the supported TX, RX, and target property dialog used with a live generator.
"""

from .dialogs import NodePropertiesDialog

__all__ = [
    "NodePropertiesDialog",
]
