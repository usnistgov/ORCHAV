from __future__ import annotations

from unittest.mock import Mock

import pytest

from visualizer.src.scene.target_materials import (
    MINIMAL_TARGET_PBR_PROPS,
    resolve_effective_target_entry_pbr_props,
    resolve_target_pbr_props,
    target_entry_pbr_fields,
    target_handle_color,
)


def test_resolve_target_pbr_props_prefers_visual_profile() -> None:
    profile = Mock()
    profile.resolve = Mock(
        return_value={
            "color": [0.1, 0.2, 0.3],
            "roughness": 0.4,
            "metallic": 0.5,
            "reflectance": 0.6,
            "alpha": 0.7,
        }
    )
    resolved = resolve_target_pbr_props(
        target_name="target_1",
        material_type="custom",
        visual_profile_service=profile,
    )

    assert resolved.visual_profile_matched is True
    assert resolved.props["color"] == [0.1, 0.2, 0.3]


def test_resolve_target_pbr_props_uses_catalog_then_fallback() -> None:
    resolved = resolve_target_pbr_props(
        target_name="target_1",
        material_type="custom",
        visual_profile_service=None,
    )

    assert resolved.visual_profile_matched is False
    assert resolved.props["color"] == [0.8, 0.6, 0.5]
    assert resolved.props["roughness"] == pytest.approx(0.6)

    fallback = resolve_target_pbr_props(
        target_name="target_1",
        material_type="glass",
        visual_profile_service=None,
    )
    assert fallback.props["color"] == [0.168, 0.139, 0.509]
    assert fallback.props["reflectance"] == pytest.approx(0.95)


def test_resolve_target_pbr_props_can_preserve_minimal_on_demand_fallback() -> None:
    resolved = resolve_target_pbr_props(
        target_name="target_1",
        material_type="glass",
        visual_profile_service=None,
        default_props=MINIMAL_TARGET_PBR_PROPS,
        use_material_fallbacks=False,
    )

    assert resolved.props == MINIMAL_TARGET_PBR_PROPS
    assert "reflectance" not in resolved.props


def test_target_entry_pbr_fields_and_handle_color() -> None:
    props = {
        "color": [0.2, 0.3, 0.4],
        "roughness": 0.5,
        "metallic": 0.6,
        "reflectance": 0.7,
        "alpha": 0.8,
    }

    fields = target_entry_pbr_fields(props)

    assert fields["pbr_color"] == [0.2, 0.3, 0.4]
    assert fields["pbr_properties"]["alpha"] == pytest.approx(0.8)
    assert target_handle_color(props, has_vertex_texture=False) == [0.2, 0.3, 0.4]
    assert target_handle_color(props, has_vertex_texture=True) == [1.0, 1.0, 1.0]


def test_resolve_effective_target_entry_pbr_props_uses_material_service() -> None:
    entry = {
        "material_type": "concrete",
        "color": [0.2, 0.3, 0.4],
        "pbr_roughness": 0.5,
        "pbr_metallic": 0.0,
        "pbr_reflectance": 0.6,
        "pbr_alpha": 0.7,
    }
    material_service = Mock()
    material_service.get_effective_entry_properties = Mock(
        return_value={"color": [0.9, 0.8, 0.7], "roughness": 0.1}
    )

    resolved = resolve_effective_target_entry_pbr_props(entry, material_service)

    assert resolved == {"color": [0.9, 0.8, 0.7], "roughness": 0.1}
