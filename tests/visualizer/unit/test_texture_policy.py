from __future__ import annotations

import os
from pathlib import Path

from visualizer.src.materials.catalog import pbr_props_to_kwargs
from visualizer.src.materials.texture_policy import (
    apply_texture_launch_policy,
    apply_texture_policy_to_props,
    resolve_texture_policy,
    textures_globally_enabled,
)


def _touch(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"texture")
    return str(path)


def test_active_albedo_locks_color_and_forces_white_base(tmp_path: Path) -> None:
    albedo = _touch(tmp_path / "albedo.png")

    policy = resolve_texture_policy(
        {"texture_path": albedo, "alpha": 0.6},
        color=[0.2, 0.3, 0.4],
        textures_enabled=True,
        context="brick",
    )

    assert policy.active_albedo_path == albedo
    assert policy.color_editable is False
    assert policy.renderer_base_color == (1.0, 1.0, 1.0, 0.6)


def test_detail_only_maps_keep_color_editable(tmp_path: Path) -> None:
    normal = _touch(tmp_path / "normal.png")
    roughness = _touch(tmp_path / "roughness.png")

    policy = resolve_texture_policy(
        {
            "normal_map_path": normal,
            "roughness_map_path": roughness,
            "alpha": 1.0,
        },
        color=[0.2, 0.3, 0.4],
        textures_enabled=True,
        context="detail-only",
    )

    assert policy.active_albedo_path is None
    assert policy.active_maps["normal_map_path"] == normal
    assert policy.active_maps["roughness_map_path"] == roughness
    assert policy.color_editable is True
    assert policy.renderer_base_color == (0.2, 0.3, 0.4, 1.0)


def test_default_textures_strip_maps_but_keep_color(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    albedo = _touch(tmp_path / "albedo.png")

    props, policy = apply_texture_policy_to_props(
        {"color": [0.3, 0.4, 0.5], "texture_path": albedo, "roughness": 0.8},
        context="disabled",
    )

    assert policy.textures_enabled is False
    assert props["texture_path"] is None
    assert props["roughness"] == 0.8
    assert policy.color_editable is True
    assert policy.renderer_base_color == (0.3, 0.4, 0.5, 1.0)


def test_enable_env_activates_texture_maps(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    albedo = _touch(tmp_path / "albedo.png")

    policy = resolve_texture_policy(
        {"texture_path": albedo},
        color=[0.3, 0.4, 0.5],
        context="enabled-env",
    )

    assert textures_globally_enabled() is True
    assert policy.textures_enabled is True
    assert policy.active_albedo_path == albedo
    assert policy.color_editable is False


def test_disable_env_overrides_enable_env(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    albedo = _touch(tmp_path / "albedo.png")

    policy = resolve_texture_policy(
        {"texture_path": albedo},
        color=[0.3, 0.4, 0.5],
        context="disabled-env",
    )

    assert textures_globally_enabled() is False
    assert policy.textures_enabled is False
    assert policy.active_albedo_path is None
    assert policy.color_editable is True


def test_apply_texture_launch_policy_enables_and_disables(monkeypatch) -> None:
    monkeypatch.delenv("ORCHAV_ENABLE_TEXTURES", raising=False)
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)

    apply_texture_launch_policy(enable_textures=True)

    assert textures_globally_enabled() is True
    assert "ORCHAV_DISABLE_TEXTURES" not in os.environ

    apply_texture_launch_policy(disable_textures=True)

    assert textures_globally_enabled() is False
    assert "ORCHAV_ENABLE_TEXTURES" not in os.environ
    assert os.environ["ORCHAV_DISABLE_TEXTURES"] == "1"


def test_missing_albedo_warns_and_falls_back_to_editable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.png"

    policy = resolve_texture_policy(
        {"texture_path": str(missing)},
        color=[0.1, 0.2, 0.3],
        textures_enabled=True,
        context="missing-case",
    )

    assert policy.active_albedo_path is None
    assert policy.color_editable is True
    assert policy.renderer_base_color == (0.1, 0.2, 0.3, 1.0)
    assert policy.warnings
    assert str(missing) in policy.warnings[0]


def test_pbr_kwargs_passes_active_maps_and_white_color(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    albedo = _touch(tmp_path / "albedo.png")
    normal = _touch(tmp_path / "normal.png")
    roughness = _touch(tmp_path / "roughness.png")
    ao = _touch(tmp_path / "ao.png")
    metallic = _touch(tmp_path / "metallic.png")

    kwargs = pbr_props_to_kwargs(
        [0.2, 0.3, 0.4],
        {
            "material_type": "full-pack",
            "alpha": 0.75,
            "roughness": 0.6,
            "metallic": 0.2,
            "texture_path": albedo,
            "normal_map_path": normal,
            "roughness_map_path": roughness,
            "ao_map_path": ao,
            "metallic_map_path": metallic,
        },
    )

    assert kwargs["color"] == [1.0, 1.0, 1.0]
    assert kwargs["alpha"] == 0.75
    assert kwargs["texture_path"] == albedo
    assert kwargs["normal_map_path"] == normal
    assert kwargs["roughness_map_path"] == roughness
    assert kwargs["ao_map_path"] == ao
    assert kwargs["metallic_map_path"] == metallic


def test_pbr_kwargs_detail_only_keeps_material_color(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ORCHAV_ENABLE_TEXTURES", "1")
    monkeypatch.delenv("ORCHAV_DISABLE_TEXTURES", raising=False)
    normal = _touch(tmp_path / "normal.png")

    kwargs = pbr_props_to_kwargs(
        [0.2, 0.3, 0.4],
        {
            "material_type": "detail-only",
            "normal_map_path": normal,
        },
    )

    assert kwargs["color"] == [0.2, 0.3, 0.4]
    assert kwargs["texture_path"] is None
    assert kwargs["normal_map_path"] == normal
