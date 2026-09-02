from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.materials.properties import collect_scene_radio_material_properties


def test_collect_scene_radio_material_properties_exports_scalar_fields():
    class TensorScalar:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return np.array([self.value], dtype=np.float32)

    scene = SimpleNamespace(
        radio_materials={
            "itu_concrete": SimpleNamespace(
                name="itu_concrete",
                relative_permittivity=TensorScalar(5.2),
                conductivity=0.03,
                scattering_coefficient=0.4,
                vector_value=[1.0, 2.0],
            ),
            "mat-itu_glass": SimpleNamespace(
                name="mat-itu_glass",
                relative_permittivity=6.4,
                conductivity=0.0,
            ),
        }
    )

    block = collect_scene_radio_material_properties(scene)

    assert block["schema_version"] == 1
    assert block["source"] == "sionna.rt.Scene.radio_materials"
    assert block["properties"]["itu_concrete"] == pytest.approx(
        {
            "relative_permittivity": 5.2,
            "conductivity": 0.03,
            "scattering_coefficient": 0.4,
        }
    )
    assert block["properties"]["mat-itu_glass"] == pytest.approx(
        {
            "relative_permittivity": 6.4,
            "conductivity": 0.0,
        }
    )
