"""Authored optical detail stays deterministic and independent of RF geometry."""

import numpy as np

from visualizer.src.materials.surface_detail import relief_normal, relief_occlusion


def test_relief_occlusion_preserves_open_surfaces_and_shades_only_cavities():
    flat = np.zeros((32, 32), dtype=np.float32)
    assert np.all(relief_occlusion(flat, (0.01, 0.01)) == 255)
    np.testing.assert_array_equal(relief_normal(flat, (0.01, 0.01))[16, 16], [128, 128, 255])
    cavity = flat.copy()
    cavity[12:20, 12:20] = -0.02
    ao = relief_occlusion(cavity, (0.01, 0.01))
    assert ao[15, 15, 0] < 250
    assert ao.min() >= 178
    assert ao[0, 0, 0] == 255
    np.testing.assert_array_equal(flat, 0)
