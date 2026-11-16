"""Optical detail maps from authored metric relief, without mesh displacement."""

from __future__ import annotations

import numpy as np


def relief_occlusion(height: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Approximate shallow local cavities from eight sampled relief horizons.

    This is a restrained texture-space approximation, not scene-wide ambient
    occlusion. Edge padding prevents unrelated atlas tiles casting into a bay.
    ``spacing`` gives the physical pixel pitch along image rows and columns.
    """
    height = np.asarray(height, dtype=np.float32)
    horizon = np.zeros_like(height)
    pad = np.pad(height, 7, mode="edge")
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)):
        direction = np.zeros_like(height)
        for radius in (1, 3, 7):
            neighbour = pad[
                7 + dy * radius : 7 + dy * radius + height.shape[0],
                7 + dx * radius : 7 + dx * radius + height.shape[1],
            ]
            distance = radius * np.hypot(dy * spacing[0], dx * spacing[1])
            direction = np.maximum(direction, (neighbour - height) / distance)
        horizon += np.arctan(direction)
    # At most 30% cavity darkening; flat/open surfaces remain exactly white.
    visibility = np.clip(1.0 - 0.45 * horizon / 8, 0.7, 1.0)
    return np.repeat(np.rint(visibility[..., None] * 255).astype(np.uint8), 3, axis=2)


def relief_normal(height: np.ndarray, spacing: tuple[float, float]) -> np.ndarray:
    """Encode tangent normals using metric height and physical pixel pitch."""
    dy, dx = np.gradient(np.asarray(height, dtype=np.float32), *spacing)
    normal = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.rint((normal + 1) * 127.5).astype(np.uint8)
