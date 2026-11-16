"""GPU endpoint checks for sunset over both a solid backdrop and a resident HDR."""

import os
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = [
    pytest.mark.pygfx_runtime,
    pytest.mark.skipif(
        os.getenv("ORCHAV_RUN_PYGFX_TESTS", "0") != "1",
        reason="set ORCHAV_RUN_PYGFX_TESTS=1 to run GPU-backed visual checks",
    ),
]


@pytest.mark.parametrize("skybox", [False, True])
def test_sunset_zero_and_disable_preserve_native_pixels(qapp, tmp_path, skybox):
    import pygfx as gfx
    from PIL import Image

    from visualizer.src.renderers.pygfx.renderer import PygfxRenderer
    from visualizer.src.types.camera_state import CameraState

    renderer = PygfxRenderer(SimpleNamespace())
    try:
        renderer.initialize_visualizer(width=320, height=240)
        renderer.set_ibl("neutral_outdoor")
        renderer.show_skybox(skybox)
        material = gfx.MeshStandardMaterial(
            map=gfx.Texture(np.full((1, 1, 3), 150, np.uint8), dim=2),
            emissive=(0, 0, 0),
        )
        source = gfx.Mesh(gfx.box_geometry(), material)
        renderer._scene.add(source)
        renderer._objects["target:probe::mesh"] = source
        renderer._materials["target:probe::mesh"] = SimpleNamespace(texture_path="probe.png")
        renderer._ibl_manager.apply_to_material(material)
        renderer.set_camera_state(CameraState((4, -5, 3), (0, 0, 0), (0, 0, 1), 55))
        renderer.configure_dusk(
            {
                "targets": [
                    {
                        "object_id": "target:probe::mesh",
                        "light": {"position": [2, 0, 0], "lookat": [5, 0, 0]},
                        "palette_slots": 1,
                        "emissive_slots": {0: [1.0, 0.5, 0.2]},
                    }
                ],
            }
        )

        def capture(name):
            qapp.processEvents()
            path = tmp_path / f"{name}.png"
            assert renderer.export_screenshot(str(path))
            return np.asarray(Image.open(path)).copy()

        original = capture("original")
        renderer.set_dusk_progress(0)
        prepared = capture("prepared")
        np.testing.assert_array_equal(prepared, original)
        emission_map = material.emissive_map
        assert emission_map is not None and material.emissive_intensity == 0
        background = renderer._dusk_background
        for index, progress in enumerate((0.25, 0.5, 0.85, 1.0)):
            renderer.set_dusk_progress(progress)
            assert renderer._dusk_background is background
            assert material.emissive_map is emission_map
            assert np.count_nonzero(capture(f"phase_{index}") != original) > 100
        renderer.set_dusk_progress(0)
        np.testing.assert_array_equal(capture("returned_to_zero"), original)
        renderer.set_dusk_enabled(False)
        np.testing.assert_array_equal(capture("disabled"), original)
        assert renderer.get_skybox_visible() == skybox
    finally:
        renderer.close()
