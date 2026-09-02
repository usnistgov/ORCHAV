"""Select renderer backends through one renderer-neutral contract.

Use :func:`create_renderer` at application composition boundaries and
:class:`RendererProtocol` in shared visualizer code. ``open3d`` and ``pygfx``
contain backend implementations; ``shared`` contains implementation policy
that is safe for both.
"""

from .factory import create_renderer
from .mpc_path_inspection import MpcPathInspectionSnapshot
from .protocol import (
    MpcPathSelectionCallback,
    RendererCapabilities,
    RendererProtocol,
    renderer_capabilities,
)
from .registry import (
    CANONICAL_RENDERER_IDS,
    DEFAULT_RENDERER_ID,
    RendererBackend,
    canonicalize_renderer_id,
    renderer_choices,
)

__all__ = [
    "CANONICAL_RENDERER_IDS",
    "DEFAULT_RENDERER_ID",
    "MpcPathInspectionSnapshot",
    "MpcPathSelectionCallback",
    "RendererBackend",
    "RendererCapabilities",
    "RendererProtocol",
    "canonicalize_renderer_id",
    "create_renderer",
    "renderer_capabilities",
    "renderer_choices",
]
