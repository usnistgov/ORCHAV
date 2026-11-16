"""Optical dusk is reversible and target headlights inherit recorded motion."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pygfx as gfx
import pytest

from shared.scenarios.model import DuskViewModel
from visualizer.src.renderers.pygfx.dusk import PygfxDuskMixin, _palette_pixels, _window_pixels


class Owner(PygfxDuskMixin):
    def __init__(self, skybox=False):
        self._gfx = gfx
        self._scene = gfx.Scene()
        self._objects = {}
        self._materials = {}
        self._clear_color = (0.3, 0.4, 0.5, 1)
        self._skybox = skybox
        self._ibl_intensity = 4321
        self._ibl_manager = SimpleNamespace(set_intensity=Mock())
        self._ambient_light = gfx.AmbientLight((0.3, 0.4, 0.9))
        self._key_light = gfx.DirectionalLight((0.9, 0.4, 0.3))
        self._fill_light = gfx.DirectionalLight((0.4, 0.5, 0.6))
        self._head_light = gfx.DirectionalLight()
        self._base_ambient_intensity = 2.0
        self._base_key_intensity = 3.0
        self._base_fill_intensity = 1.0
        self._headlight_enabled = True
        self._headlight_intensity = 1.2
        self._lighting_profile_name = "custom"
        self._request_canvas_draw = Mock()
        self._remove_solid_background = Mock()
        self._apply_light_intensities()

    def get_light_rig_state(self):
        return dict(
            lighting_profile=self._lighting_profile_name,
            ambient_intensity=self._base_ambient_intensity,
            key_intensity=self._base_key_intensity,
            fill_intensity=self._base_fill_intensity,
            headlight_enabled=self._headlight_enabled,
            headlight_intensity=self._headlight_intensity,
            shadows_enabled=True,
        )

    def _apply_light_intensities(self):
        self._ambient_light.intensity = self._base_ambient_intensity
        self._key_light.intensity = self._base_key_intensity
        self._fill_light.intensity = self._base_fill_intensity
        self._head_light.intensity = self._headlight_intensity if self._headlight_enabled else 0

    def get_ibl_intensity(self):
        return self._ibl_intensity

    def get_skybox_visible(self):
        return self._skybox

    def show_skybox(self, value):
        self._skybox = value

    def set_background_color(self, value):
        self._clear_color = tuple(value)

    def restore_light_rig_state(self, value, *, ibl):
        self._lighting_profile_name = value["lighting_profile"]
        for field in ("ambient_intensity", "key_intensity", "fill_intensity"):
            setattr(self, f"_base_{field}", value[field])
        self._headlight_enabled = value["headlight_enabled"]
        self._headlight_intensity = value["headlight_intensity"]
        self._apply_light_intensities()
        self._ibl_intensity = ibl


@pytest.fixture
def recipe():
    # Self-contained optical fixture: two wall lights and one 19-slot actor.
    return {
        "facade_lights": [
            {"position": [0, 0, 2], "lookat": [0, 3, 1], "intensity": 4},
            {"position": [4, 0, 2], "lookat": [4, 3, 1], "intensity": 5},
        ],
        "targets": [
            {
                "object_id": "fixture_vehicle",
                "light": {"position": [0, 0, 1], "lookat": [0, 5, 1], "intensity": 3},
                "palette_slots": 19,
                "emissive_slots": {3: [1, 0.7, 0.2], 9: [1, 0.1, 0.1], 11: [1, 1, 1]},
            }
        ],
    }


def source_mesh(owner, name):
    texture = gfx.Texture(np.ones((1, 19, 3), dtype=np.uint8), dim=2)
    material = gfx.MeshStandardMaterial(
        map=texture, emissive=(0.1, 0.2, 0.3), emissive_intensity=0.4
    )
    mesh = gfx.Mesh(gfx.box_geometry(), material)
    owner._scene.add(mesh)
    owner._objects[name] = mesh
    owner._materials[name] = SimpleNamespace(texture_path="bmw/albedo.png")
    return mesh


@pytest.mark.parametrize("skybox", [False, True])
def test_disable_restores_exact_optical_state_and_releases_all_lights(recipe, skybox):
    owner = Owner(skybox)
    source = source_mesh(owner, recipe["targets"][0]["object_id"])
    original = (tuple(source.material.emissive), source.material.emissive_intensity)
    rig, color, ibl = owner.get_light_rig_state(), owner._clear_color, owner._ibl_intensity
    colors = [
        tuple(getattr(owner, name).color)
        for name in ("_ambient_light", "_key_light", "_fill_light")
    ]
    owner.configure_dusk(recipe)
    assert not owner.get_dusk_state()["enabled"]
    assert owner.set_dusk_enabled(True)
    assert owner.get_dusk_state()["facade_lights"] == 2
    assert owner.get_dusk_state()["headlights"] == 1
    assert all(not light.cast_shadow for light in owner._dusk_facade_lights)
    assert source.material.emissive_map is not None
    assert owner.set_dusk_enabled(True)
    assert owner.get_dusk_state()["facade_lights"] == 2
    owner.set_dusk_enabled(False)
    assert owner.get_light_rig_state() == rig
    assert owner._clear_color == color and owner._ibl_intensity == ibl and owner._skybox == skybox
    assert [
        tuple(getattr(owner, name).color)
        for name in ("_ambient_light", "_key_light", "_fill_light")
    ] == colors
    assert tuple(source.material.emissive) == original[0]
    assert source.material.emissive_intensity == original[1]
    assert source.material.emissive_map is None
    assert list(owner._scene.children) == [source]
    assert not source.children


def test_headlights_follow_source_transform_visibility_and_replacement(recipe):
    owner = Owner()
    name = recipe["targets"][0]["object_id"]
    source = source_mesh(owner, name)
    owner.configure_dusk(recipe)
    owner.set_dusk_enabled(True)
    light = owner._dusk_target_bindings[name][1]
    local = np.asarray(light.local.position).copy()
    source.local.position = (4, 5, 6)
    np.testing.assert_allclose(light.world.position, local + (4, 5, 6))
    source.visible = False
    assert owner.get_dusk_state()["visible_headlights"] == 0
    replacement = source_mesh(owner, name)
    owner._scene.remove(source)
    owner._sync_dusk()
    assert not source.children
    assert owner._dusk_target_bindings[name][0] is replacement
    assert owner.get_dusk_state()["visible_headlights"] == 1
    owner.configure_dusk(None)
    assert not replacement.children
    assert not owner.get_dusk_state()["available"]


def test_material_color_and_wireframe_do_not_keep_emission(recipe):
    owner = Owner()
    source = source_mesh(owner, recipe["targets"][0]["object_id"])
    owner.configure_dusk(recipe)
    owner.set_dusk_enabled(True)
    source.material.wireframe = True
    owner._sync_dusk()
    assert source.material.emissive_map is None
    source.material.wireframe = False
    owner._sync_dusk()
    assert source.material.emissive_map is not None
    source.material.map = None
    owner._sync_dusk()
    assert source.material.emissive_map is None


def test_replaced_material_is_restored_and_released_immediately(recipe):
    owner = Owner()
    name = recipe["targets"][0]["object_id"]
    source = source_mesh(owner, name)
    original = source.material
    owner.configure_dusk(recipe)
    owner.set_dusk_enabled(True)
    replacement = gfx.MeshStandardMaterial(map=original.map)
    source.material = replacement
    owner._sync_dusk()
    assert original.emissive_map is None
    assert len(owner._dusk_emissions) == 1
    assert replacement.emissive_map is not None
    owner.set_dusk_enabled(False)
    assert source.material is replacement
    assert tuple(replacement.emissive) == (0, 0, 0, 1)


def test_transparent_material_color_keeps_its_original_emission(recipe):
    owner = Owner()
    source = source_mesh(owner, recipe["targets"][0]["object_id"])
    original = tuple(source.material.emissive)
    source.material.color = (1, 0, 0, 0.5)
    assert source.material.opacity == 1 and source.material.depth_write
    owner.configure_dusk(recipe)
    owner.set_dusk_enabled(True)
    assert source.material.emissive_map is None
    assert tuple(source.material.emissive) == original
    source.material.color = (1, 0, 0, 1)
    owner._sync_dusk()
    assert source.material.emissive_map is not None
    source.material.color = (1, 0, 0, 0.5)
    owner._sync_dusk()
    assert source.material.emissive_map is None
    assert tuple(source.material.emissive) == original


def test_off_sync_does_not_touch_objects_or_materials():
    owner = Owner()
    del owner._objects, owner._materials
    owner._sync_dusk()
    assert not owner._request_canvas_draw.called


@pytest.mark.parametrize("skybox", [False, True])
def test_progress_zero_preserves_daylight_and_existing_emission(recipe, skybox):
    owner = Owner(skybox)
    name = recipe["targets"][0]["object_id"]
    source = source_mesh(owner, name)
    original = (
        tuple(source.material.emissive),
        source.material.emissive_intensity,
        source.material.emissive_map,
    )
    background = gfx.Background.from_color("#abcdef")
    owner._scene.add(background)
    rig = owner.get_light_rig_state()
    colors = [
        tuple(light.color) for light in (owner._ambient_light, owner._key_light, owner._fill_light)
    ]
    owner.configure_dusk(recipe)
    owner.set_dusk_progress(0, request_redraw=False)
    assert owner.get_dusk_state()["enabled"] and owner.get_dusk_state()["progress"] == 0
    assert owner.get_light_rig_state() == rig
    assert owner.get_skybox_visible() == skybox
    assert owner._dusk_background.material.opacity == 0
    assert background.parent is owner._scene
    assert (
        tuple(source.material.emissive),
        source.material.emissive_intensity,
        source.material.emissive_map,
    ) == original
    assert all(light.intensity == 0 for light in owner._dusk_facade_lights)
    assert owner._dusk_target_bindings[name][1].intensity == 0
    owner.set_dusk_progress(1, request_redraw=False)
    owner.set_dusk_progress(0, request_redraw=False)
    assert owner.get_light_rig_state() == rig
    assert [
        tuple(light.color) for light in (owner._ambient_light, owner._key_light, owner._fill_light)
    ] == colors
    assert (
        tuple(source.material.emissive),
        source.material.emissive_intensity,
        source.material.emissive_map,
    ) == original
    assert background.parent is owner._scene
    assert not owner._request_canvas_draw.called
    owner.set_dusk_enabled(False)
    assert background.parent is owner._scene


def test_sunset_prebinds_maps_and_reuses_resources_without_per_tick_draws(recipe, monkeypatch):
    owner = Owner()
    name = recipe["targets"][0]["object_id"]
    source = source_mesh(owner, name)
    source.material.emissive = (0, 0, 0)
    owner.configure_dusk(recipe)
    owner.set_dusk_progress(0, request_redraw=False)
    texture_map = source.material.emissive_map
    assert texture_map is not None and source.material.emissive_intensity == 0
    background = owner._dusk_background
    lights = tuple(owner._dusk_facade_lights) + (owner._dusk_target_bindings[name][1],)
    material = source.material
    with monkeypatch.context() as patch:
        patch.setattr(
            gfx, "Texture", Mock(side_effect=AssertionError("a transition must reuse textures"))
        )
        for value in (0.1, 0.35, 0.5, 0.7, 0.85, 1.0, 0.5, 0.0):
            owner.set_dusk_progress(value, request_redraw=False)
            assert source.material is material and source.material.emissive_map is texture_map
            assert owner._dusk_background is background
            assert (
                tuple(owner._dusk_facade_lights) + (owner._dusk_target_bindings[name][1],) == lights
            )
    assert not owner._request_canvas_draw.called
    owner.set_dusk_progress(0.2)
    owner._request_canvas_draw.assert_called_once()
    owner.set_dusk_progress(0.2)
    owner._request_canvas_draw.assert_called_once()


def test_architectural_lights_precede_vehicle_lamps_and_full_endpoint_matches(recipe):
    owner = Owner()
    name = recipe["targets"][0]["object_id"]
    source = source_mesh(owner, name)
    owner.configure_dusk(recipe)
    owner.set_dusk_progress(0.5)
    assert 0 < owner._dusk_facade_lights[0].intensity < recipe["facade_lights"][0]["intensity"]
    assert owner._dusk_target_bindings[name][1].intensity == 0
    assert source.material.emissive_intensity == 0
    owner.set_dusk_progress(0.85)
    assert (
        0
        < owner._dusk_target_bindings[name][1].intensity
        < recipe["targets"][0]["light"]["intensity"]
    )
    owner.set_dusk_enabled(True)
    assert owner.get_dusk_state()["progress"] == 1
    assert owner._base_ambient_intensity == 0.55 and owner._base_key_intensity == 0.42
    assert owner._base_fill_intensity == 0.22 and not owner._headlight_enabled
    assert owner.get_ibl_intensity() == 700
    assert owner._dusk_background.material.opacity == 1
    assert source.material.emissive_intensity == pytest.approx(1.2)
    assert owner._dusk_target_bindings[name][1].intensity == pytest.approx(
        recipe["targets"][0]["light"]["intensity"]
    )


def test_replacement_actor_inherits_current_sunset_phase(recipe):
    owner = Owner()
    name = recipe["targets"][0]["object_id"]
    old = source_mesh(owner, name)
    owner.configure_dusk(recipe)
    owner.set_dusk_progress(0.85)
    intensity = old.material.emissive_intensity
    light_intensity = owner._dusk_target_bindings[name][1].intensity
    replacement = source_mesh(owner, name)
    owner._scene.remove(old)
    owner._sync_dusk()
    assert replacement.material.emissive_intensity == intensity
    assert owner._dusk_target_bindings[name][1].intensity == light_intensity
    assert old.material.emissive_map is None and not old.children


@pytest.mark.parametrize("value", [-0.001, 1.001, float("nan"), float("inf")])
def test_invalid_progress_leaves_original_rig_untouched(recipe, value):
    owner = Owner()
    owner.configure_dusk(recipe)
    rig = owner.get_light_rig_state()
    with pytest.raises(ValueError, match="progress"):
        owner.set_dusk_progress(value)
    assert not owner.get_dusk_state()["enabled"]
    assert owner.get_light_rig_state() == rig and not owner._scene.children


def test_emission_masks_do_not_light_bodywork_or_opaque_window_frames(recipe):
    target = DuskViewModel.model_validate(recipe).model_dump()["targets"][0]
    pixels = _palette_pixels(target)
    assert np.flatnonzero(pixels.sum(axis=(0, 2))).tolist() == [3, 9, 11]
    atlas = np.zeros((4096, 1536, 3), dtype=np.uint8)
    atlas[100, 100] = (74, 92, 99)
    atlas[100, 101] = (55, 59, 58)  # Balcony iron.
    mask = _window_pixels(atlas)
    assert mask[100, 100].any() and not mask[100, 101].any()
    assert np.count_nonzero(mask) == 3


@pytest.mark.parametrize("case", ["nan", "coincident", "lights", "palette"])
def test_invalid_recipe_rejected(recipe, case):
    if case == "nan":
        recipe["facade_lights"][0]["position"][0] = float("nan")
    elif case == "coincident":
        recipe["facade_lights"][0]["lookat"] = recipe["facade_lights"][0]["position"]
    elif case == "lights":
        recipe["facade_lights"] *= 4
    else:
        recipe["targets"][0]["emissive_slots"][19] = [1, 1, 1]
    with pytest.raises(ValueError):
        DuskViewModel.model_validate(recipe)
