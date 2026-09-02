"""Open3D/Filament backend for the visualizer renderer protocol.

``Open3DRenderer`` is the package entry point. The implementation is split
across mixins for runtime event pumping, camera controls, geometry conversion,
materials/lighting, MPC payloads, nodes, overlays, and trajectories.
"""

from .renderer import Open3DRenderer

__all__ = ["Open3DRenderer"]
