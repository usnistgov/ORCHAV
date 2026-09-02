"""Opt-in production-pygfx visual smoke for selected MPC presentation."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1",
        reason="set ORCHAV_RUN_PYGFX_TESTS=1 to run GPU-backed visual checks",
    ),
]


def test_selected_path_visual_contains_visible_colored_overlay(tmp_path) -> None:
    """Capture halo/pulse/number-label presentation through the real renderer."""
    pytest.importorskip("pygfx")
    image_module = pytest.importorskip("PIL.Image")

    from scripts.capture_mpc_explorer_visual import capture_mpc_explorer_visual

    output = capture_mpc_explorer_visual(
        tmp_path / "mpc_selected_path.png",
        width=720,
        height=480,
    )
    image = np.asarray(image_module.open(output).convert("RGB"), dtype=np.uint8)

    assert image.shape == (480, 720, 3)
    assert int(image.max()) > 100
    assert float(np.std(image.astype(np.float32))) > 5.0
    # The selected path is cyan while bounce markers deliberately add warm and
    # violet pixels; checking both channels catches an empty/context-only draw.
    channels = image.astype(np.int16)
    assert np.count_nonzero(channels[..., 2] > channels[..., 0] + 35) > 100
    assert np.count_nonzero(channels[..., 0] > channels[..., 1] + 25) > 10
