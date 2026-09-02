from __future__ import annotations

import pytest

from visualizer.src.materials.appearance import (
    AppearanceIntent,
    MaterialDisplayMode,
    VisualMaterialBinding,
    VisualMaterialSource,
    resolve_appearance,
)
from visualizer.src.types.render_payloads import MaterialPayload, SurfaceColorSource


@pytest.mark.parametrize(
    ("overrides", "expected_visible", "expected_highlighted"),
    [
        ({}, True, False),
        ({"manual_visible": False}, False, False),
        ({"runtime_visible": False}, False, False),
        ({"frame_visible": False}, False, False),
        ({"pov_visible": False}, False, False),
        ({"global_visible": False}, False, False),
        ({"material_mode": MaterialDisplayMode.HIDDEN}, False, False),
        ({"manual_highlight": True}, True, True),
        ({"selected": True}, True, True),
        ({"material_mode": MaterialDisplayMode.HIGHLIGHTED}, True, True),
        ({"manual_visible": False, "manual_highlight": True}, False, False),
        (
            {"material_mode": MaterialDisplayMode.HIDDEN, "manual_highlight": True},
            False,
            False,
        ),
    ],
)
def test_appearance_truth_table(overrides, expected_visible, expected_highlighted) -> None:
    resolved = resolve_appearance(AppearanceIntent(**overrides))
    assert resolved.visible is expected_visible
    assert resolved.highlighted is expected_highlighted


def test_uniform_highlight_becomes_red_and_restores_base_color() -> None:
    material = MaterialPayload(base_color=(0.2, 0.4, 0.6, 0.7))
    highlighted = resolve_appearance(AppearanceIntent(material=material, manual_highlight=True))
    normal = resolve_appearance(AppearanceIntent(material=material))

    assert highlighted.material.base_color == (1.0, 0.0, 0.0, 0.7)
    assert highlighted.material.color_multiplier == (1.0, 1.0, 1.0)
    assert normal.material == material


@pytest.mark.parametrize(
    ("source", "texture_path"),
    [
        (SurfaceColorSource.MATERIAL, "authored_albedo.png"),
        (SurfaceColorSource.VERTEX, None),
    ],
)
def test_authored_rgb_highlight_uses_multiplier(source, texture_path) -> None:
    material = MaterialPayload(
        base_color=(1.0, 1.0, 1.0, 0.8),
        texture_path=texture_path,
    )
    resolved = resolve_appearance(
        AppearanceIntent(material=material, color_source=source, selected=True)
    )

    assert resolved.material.base_color == material.base_color
    assert resolved.material.texture_path == texture_path
    assert resolved.material.color_multiplier == (1.0, 0.3, 0.3)


def test_manual_highlight_is_latent_while_material_is_hidden() -> None:
    hidden = resolve_appearance(
        AppearanceIntent(
            manual_highlight=True,
            material_mode=MaterialDisplayMode.HIDDEN,
        )
    )
    restored = resolve_appearance(AppearanceIntent(manual_highlight=True))

    assert hidden.visible is False
    assert hidden.highlighted is False
    assert restored.visible is True
    assert restored.highlighted is True


def test_visual_material_bindings_keep_explicit_assignments_separate() -> None:
    assert VisualMaterialBinding().source is VisualMaterialSource.FOLLOW_EM
    profile = VisualMaterialBinding(
        source=VisualMaterialSource.PROFILE,
        preset="Skin",
        overrides={"roughness": 0.6},
    )
    manual = VisualMaterialBinding(
        source=VisualMaterialSource.MANUAL,
        material_type="copper",
    )
    assert profile.preset == "Skin"
    assert manual.material_type == "copper"
    with pytest.raises(TypeError):
        profile.overrides["roughness"] = 0.2


@pytest.mark.parametrize("value", ["normal", "hidden", "highlighted"])
def test_material_display_mode_accepts_exact_serialized_values(value) -> None:
    assert MaterialDisplayMode.coerce(value) in set(MaterialDisplayMode)
