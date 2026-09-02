"""pygfx/wgpu renderer backend.

``PygfxRenderer`` is the package entry point. The large renderer class composes
mixins for lifecycle/runtime, named geometry, materials, MPC payloads, overlays,
picking, and transform-gizmo behavior.
"""

from .renderer import PygfxRenderer

__all__ = ["PygfxRenderer"]
