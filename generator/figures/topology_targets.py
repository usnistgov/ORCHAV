"""Display geometry for native RCS targets without loading the optional solver.

The native target's existing preview contract supplies the proxy dimensions.
These boxes describe placement/occlusion, not scattering from metal surfaces.
"""

from pathlib import Path

import numpy as np

from .mesh_geometry import _read_mesh


def native_proxy_geometry(config, metadata: dict, config_path: Path):
    """Return the configured rigid proxy and the source that defines its shape."""
    geometry = metadata["geometry"]
    if geometry["type"] == "mesh":
        path = Path(config.resolved_mesh_path)
        vertices, faces = _read_mesh(path)
        return vertices * config.scale, faces, path
    if geometry["type"] != "box":
        raise ValueError(f"Unsupported native proxy geometry: {geometry['type']}")
    dimensions = np.array([geometry[key] for key in ("length_m", "width_m", "height_m")])
    vertices = (
        np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ]
        )
        * dimensions
        / 2
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
    )
    return vertices, faces, config_path
