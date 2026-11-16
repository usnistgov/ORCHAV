"""Published atlas paths stay readable while retained materials reference them."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from visualizer.src.materials import scene_variation


@pytest.mark.parametrize("kind", ["roads", "facades"])
def test_referenced_atlas_survives_other_scene_palettes(tmp_path, monkeypatch, kind):
    monkeypatch.setenv("ORCHAV_UV_CACHE_DIR", str(tmp_path / "uv"))
    # Pixel generation is covered by the facade/road tests. Small real PNGs
    # exercise publication and retention without rendering forty full atlases.
    monkeypatch.setattr(
        scene_variation,
        "atlas_pixels",
        lambda kind, rgb, theme: np.full((4, 4, 3), rgb, dtype=np.uint8),
    )
    referenced = Path(scene_variation._atlas(kind, (76, 76, 76), "mixed"))
    for shade in range(40):
        scene_variation._atlas("facades", (shade, shade, shade), "european")

    # A retained material uses this original path directly: calling _atlas()
    # again before checking would regenerate a missing file and hide the bug.
    assert referenced.is_file()
    with Image.open(referenced) as image:
        image.load()
        assert image.getpixel((0, 0)) == (76, 76, 76)
