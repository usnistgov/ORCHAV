"""Generate original, seamless illustrative water maps; no external artwork.

A 256 m tile combines broad surface variation with short ripples, so detail
survives city-view minification. The maps alter visual lighting only: there is
no displaced geometry, fluid model, animation, or RF roughness change.
"""

from pathlib import Path

import numpy as np
from PIL import Image

SIZE_PX = 2048
TILE_METRES = 256.0
# Integer frequencies make each map periodic. Mixed directions and phases
# avoid aligned crests. Amplitudes construct optical normals, not displaced
# vertices or a site-specific wave-height estimate.
WAVES = (
    (2, 1, 0.260, 0.3),
    (-1, 3, 0.200, 1.8),
    (4, 1, 0.150, 4.1),
    (-3, 5, 0.110, 0.7),
    (5, 6, 0.085, 2.6),
    (-7, 4, 0.065, 5.1),
    (9, 2, 0.055, 3.2),
    (19, 11, 0.050, 1.1),
    (-27, 17, 0.038, 4.8),
    (43, 29, 0.020, 2.3),
    (67, -31, 0.012, 5.8),
    (127, 57, 0.0050, 0.9),
    (-163, 97, 0.0040, 3.7),
    (233, 149, 0.0025, 2.1),
    (311, -187, 0.0018, 5.2),
)


def generate(output: Path) -> None:
    """Write deterministic albedo, normal and roughness maps at 2048² pixels."""
    x = np.arange(SIZE_PX, dtype=np.float32)[None, :] / SIZE_PX
    y = np.arange(SIZE_PX, dtype=np.float32)[:, None] / SIZE_PX
    dx = np.zeros((SIZE_PX, SIZE_PX), dtype=np.float32)
    dy = np.zeros_like(dx)
    for kx, ky, amplitude_m, phase in WAVES:
        angle = 2 * np.pi * (kx * x + ky * y) + phase
        slope = amplitude_m * 2 * np.pi / TILE_METRES * np.cos(angle)
        dx += kx * slope
        dy += ky * slope
    normals = np.stack((-dx, -dy, np.ones_like(dx)), axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    normal_rgb = np.round((normals * 0.5 + 0.5) * 255).astype(np.uint8)
    # Broad roughness variation changes the light response at a distance;
    # it does not paint bright crests or suggest depth/bathymetry in the albedo.
    modulation = np.tanh(
        0.8 * np.sin(2 * np.pi * (2 * x - 3 * y) + 1.9)
        + 0.55 * np.cos(2 * np.pi * (5 * x + 7 * y) + 4.2)
        + 0.3 * np.sin(2 * np.pi * (11 * x - 4 * y) + 0.4)
        + 0.15 * np.cos(2 * np.pi * (37 * x + 19 * y) + 2.4)
    )
    tone = 1.6 * modulation + 0.4 * np.sin(2 * np.pi * (43 * x + 29 * y) + 2.3)
    albedo = np.round(np.clip(np.array([36, 63, 68]) + tone[..., None], 0, 255)).astype(np.uint8)
    roughness = np.round((0.52 + 0.11 * modulation) * 255).astype(np.uint8)
    output.mkdir(parents=True, exist_ok=True)
    Image.fromarray(albedo).save(output / "albedo.png")
    Image.fromarray(normal_rgb).save(output / "normal.png")
    Image.fromarray(roughness).save(output / "roughness.png")


if __name__ == "__main__":
    generate(Path(__file__).parent)
