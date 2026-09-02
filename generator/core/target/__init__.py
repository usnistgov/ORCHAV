"""Mesh-backed target configuration and scene-object management.

Targets are scene objects that can move, rotate, switch mesh frames, and carry a
radio material. The package surface is intentionally small:
``TargetConfig`` stores YAML/API configuration, while ``TargetManager`` owns the
live Sionna ``SceneObject``. Lower-level mesh conversion and metadata helpers
stay private to keep target asset compatibility details out of the package API.
"""

from typing import TYPE_CHECKING

from .config import TargetConfig

if TYPE_CHECKING:
    from .manager import TargetManager

__all__ = [
    "TargetConfig",
    "TargetManager",
]


def __getattr__(name: str):
    if name == "TargetManager":
        # Delay Sionna/Mitsuba imports until scene construction actually needs
        # a live target manager. Config parsing can then import TargetConfig in
        # lightweight environments.
        from .manager import TargetManager

        return TargetManager
    raise AttributeError(name)
