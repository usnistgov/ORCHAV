"""OSM alignment, bounded acquisition and independent geographic map controls."""

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from PySide6.QtCore import QObject, Signal

from visualizer.src.panels.map_layer_controls import MapLayerControls
from visualizer.src.renderers.protocol import RendererCapabilities
from visualizer.src.scene.map_reference import (
    load_scene_map,
    mercator_bounds,
    project_map,
    read_map,
    tile_layout,
)
from visualizer.src.scene.visibility_policy import effective_entry_visibility
from visualizer.src.services.map_layer_service import MapLayerService
from visualizer.src.services.session_service import SessionService


@pytest.fixture
def metadata():
    return {
        "bbox": [48.8675, 2.3265, 48.876, 2.3385],
        "utm_crs": "EPSG:32631",
        "origin_utm": [451052.28302045795, 5413413.766453487, 0],
        "osm_config": {"ground_pad_m": 40},
    }


def saved_map(root, metadata):
    root.mkdir(parents=True, exist_ok=True)
    bounds = mercator_bounds(metadata)
    (root / "map.json").write_text(json.dumps(project_map(metadata, bounds)))
    Image.new("RGB", (256, 256), "#59af55").save(root / "map.png")
    return root / "map.json"


@pytest.mark.parametrize("damage", ["svg", "jpeg", "truncated_png"])
def test_saved_map_rejects_non_png_and_incomplete_pixels(tmp_path, metadata, damage):
    path = saved_map(tmp_path, metadata)
    texture = path.with_name("map.png")
    if damage == "svg":
        texture.write_text(
            '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16">'
            '<rect width="16" height="16" fill="green"/></svg>'
        )
    elif damage == "jpeg":
        Image.new("RGB", (16, 16), "green").save(texture, format="JPEG")
    else:
        # Retain the PNG signature and dimensions; header-only validation
        # would accept this file without noticing missing pixel data.
        texture.write_bytes(texture.read_bytes()[:48])
    with pytest.raises(ValueError, match="readable PNG"):
        read_map(path)


def test_saved_map_pixel_guard_uses_normal_invalid_map_recovery(tmp_path, metadata, monkeypatch):
    path = saved_map(tmp_path, metadata)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 128)
    with pytest.raises(ValueError, match="readable PNG"):
        read_map(path)


def test_compiler_warns_about_invalid_map_and_recovers_after_repair(tmp_path, metadata):
    from visualizer.src.authoring.compiler import IssueSeverity, ScenarioCompiler
    from visualizer.src.authoring.domain import (
        ActorRole,
        AuthoringActor,
        AuthoringScenario,
        ScenarioSourceSnapshot,
        SceneReference,
    )
    from visualizer.src.authoring.map_overlay import MAP_NAME

    path = saved_map(tmp_path / "builder_map", metadata)
    path.with_name("map.png").write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    scenario = AuthoringScenario(
        scene=SceneReference("library", "empty/empty.xml"),
        actors=(
            AuthoringActor.create(ActorRole.TX, "TX"),
            AuthoringActor.create(ActorRole.RX, "RX", position=(1, 0, 0)),
        ),
        source_snapshot=ScenarioSourceSnapshot.from_mapping(
            {"visualizer": {"scenario_builder": {"map_reference": "builder_map/map.json"}}}
        ),
    )
    compiler = ScenarioCompiler(Path(__file__).resolve().parents[3])
    result = compiler.compile(scenario, scenario_directory=tmp_path)
    assert result.valid, result.issues
    issue = next(
        issue for issue in result.issues if issue.code == "scene.map_reference.unavailable"
    )
    assert issue.severity is IssueSeverity.WARNING and "readable PNG" in issue.message
    assert not any(asset.name == MAP_NAME for asset in result.scene_assets)

    Image.new("RGB", (16, 16), "green").save(path.with_name("map.png"))
    repaired = compiler.compile(scenario, scenario_directory=tmp_path)
    assert repaired.valid, repaired.issues
    assert not any(issue.code == "scene.map_reference.unavailable" for issue in repaired.issues)
    assert any(asset.name == MAP_NAME for asset in repaired.scene_assets)


def test_projection_uses_actual_utm_origin_and_image_north(metadata):
    from pyproj import Transformer

    original = deepcopy(metadata)
    bounds = mercator_bounds(metadata)
    geometry = project_map(metadata, bounds)
    vertices, uvs = np.array(geometry["vertices"]), np.array(geometry["uvs"])
    transform = Transformer.from_crs(3857, metadata["utm_crs"], always_xy=True)
    for index, xy in ((0, bounds[:2]), (-1, bounds[2:])):
        actual = np.array(transform.transform(*xy)) - metadata["origin_utm"][:2]
        np.testing.assert_allclose(vertices[index, :2], actual, atol=1e-8)
    np.testing.assert_array_equal(uvs[0], [0, 1])
    np.testing.assert_array_equal(uvs[-1], [1, 0])
    # UTM convergence must remain visible: the projected map is not axis-aligned.
    assert abs(vertices[12, 1] - vertices[0, 1]) > 1
    assert metadata == original
    zoom, x0, y0, x1, y1 = tile_layout(bounds)
    assert 0 <= zoom <= 19 and (x1 - x0) * (y1 - y0) <= 16


@pytest.mark.parametrize("damage", ["origin", "bbox", "crs", "nan", "wrap"])
def test_invalid_georeferencing_never_uses_default_location(metadata, damage):
    if damage == "origin":
        metadata.pop("origin_utm")
    if damage == "bbox":
        metadata["bbox"] = [0, 0, 0, 0]
    if damage == "crs":
        metadata["utm_crs"] = "EPSG:4326"
    if damage == "nan":
        metadata["bbox"][0] = float("nan")
    if damage == "wrap":
        metadata["bbox"] = [40, 179.9, 41, -179.9]
    with pytest.raises(ValueError):
        mercator_bounds(metadata)


def test_worker_fetches_one_bounded_level_reuses_http_cache_and_cancels(tmp_path, metadata):
    from tests.visualizer.authoring.test_osm_builder import tile_server

    output = tmp_path / "job"
    output.mkdir()
    with tile_server() as (url, requests):
        config = {
            "metadata": metadata,
            "cache": str(tmp_path / "tiles"),
            "tile_url": url + "/{z}/{x}/{y}.png",
        }
        first = load_scene_map(config, output, lambda: False, lambda _: None)
        assert 1 <= first["loaded_tiles"] <= 16
        assert len(requests) == first["loaded_tiles"]
        second = load_scene_map(config, output, lambda: False, lambda _: None)
        assert second["cached_tiles"] == first["loaded_tiles"]
        assert len(requests) == first["loaded_tiles"]
        mesh, texture, credit = read_map(second["map_path"])
        assert len(mesh.triangles) == 288 and Path(texture).is_file()
        assert credit == "© OpenStreetMap contributors"
        with pytest.raises(InterruptedError):
            load_scene_map(config, output, lambda: True, lambda _: None)


class Job(QObject):
    progress = Signal(str)
    finished = Signal(object)

    def __init__(self, operation, config, parent):
        super().__init__(parent)
        self.operation, self.config = operation, config
        self.root = Path(config["cache"]).parent / f"job-{id(self)}"
        self.root.mkdir(parents=True)
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.cancel()
        self.finished.emit({"event": "cancelled"})


@pytest.fixture
def service(qapp, tmp_path, metadata, monkeypatch):
    monkeypatch.setenv("ORCHAV_CACHE_DIR", str(tmp_path / "cache"))
    renderer = SimpleNamespace(
        capabilities=RendererCapabilities(viewport_hud=True),
        ensure_object=Mock(return_value=True),
        remove_object=Mock(),
        set_map_attribution=Mock(),
        request_redraw=Mock(),
        get_camera_state=Mock(return_value="previous-camera"),
        set_camera_state=Mock(),
    )
    viz = SimpleNamespace(
        renderer=renderer,
        mesh_entries=[
            {"name": "building", "visible": True},
            {"name": "road", "visible": True, "_geographic_layer_visible": False},
            {"name": "hidden", "visible": False},
        ],
        target_entries=[{"name": "target", "entry_type": "target", "visible": True}],
        camera_controller=SimpleNamespace(set_overview_view=Mock()),
        object_appearance_service=SimpleNamespace(refresh_object_visibility_batch=Mock()),
    )
    service = MapLayerService(viz, job_factory=Job)
    viz.scene_service = SimpleNamespace(map_layer=service)
    (tmp_path / "osm_metadata.json").write_text(json.dumps(metadata))
    service.set_scene(tmp_path / "scene.xml")
    yield service
    service.close()


def finish(service, metadata):
    job = service._job
    saved_map(job.root, metadata)
    job.finished.emit(
        {
            "event": "complete",
            "result": {
                "map_path": str(job.root / "map.json"),
                "loaded_tiles": 9,
                "cached_tiles": 9,
            },
        }
    )


def test_map_is_independent_of_scene_camera_and_reuses_geometry(service, metadata):
    entries, targets = service.viz.mesh_entries, deepcopy(service.viz.target_entries)
    source = deepcopy(entries)
    service.set_enabled(True)
    assert service._job is not None
    finish(service, metadata)
    assert entries == source and service.viz.target_entries == targets
    service.viz.camera_controller.set_overview_view.assert_not_called()
    rendered = service.viz.renderer.ensure_object.call_args.args[0]
    assert rendered.visible and not rendered.metadata["pickable"]
    mesh = rendered.payload
    service.set_height(1.5)
    service.set_opacity(0.4)
    rendered = service.viz.renderer.ensure_object.call_args.args[0]
    assert rendered.payload is mesh
    assert rendered.transform.translation[2] == 1.5
    assert rendered.material_payload.base_color[3] == 0.4
    service.set_enabled(False)
    assert entries == source
    service.viz.renderer.set_camera_state.assert_not_called()


def test_stale_cancelled_result_cannot_hide_or_overlay_next_scene(service, tmp_path, metadata):
    service.set_enabled(True)
    old = service._job
    service.set_scene(tmp_path / "other" / "scene.xml")
    assert old.cancelled and not service.available
    saved_map(old.root, metadata)
    old.finished.emit(
        {
            "event": "complete",
            "result": {
                "map_path": str(old.root / "map.json"),
                "loaded_tiles": 9,
                "cached_tiles": 0,
            },
        }
    )
    assert service._mesh is None and not old.root.exists()
    assert effective_entry_visibility(service.viz.mesh_entries[0])
    service.viz.renderer.ensure_object.assert_not_called()


def test_failure_keeps_scene_and_retry_is_possible(service):
    service.set_enabled(True)
    service._job.finished.emit({"event": "failed", "error": "offline"})
    assert not service.enabled and "offline" in service.status
    assert effective_entry_visibility(service.viz.mesh_entries[0])
    service.set_enabled(True)
    assert service._job is not None
    service.set_enabled(False)
    assert service._job is None


def test_saved_reference_is_offline_and_workspace_preserves_settings(service, tmp_path, metadata):
    service.saved_map = saved_map(tmp_path / "saved", metadata)
    record = json.loads(service.saved_map.read_text())
    record["attribution"] = (
        '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    )
    service.saved_map.write_text(json.dumps(record))
    service.set_enabled(True)
    assert service._job is None and service.enabled
    service.viz.renderer.set_map_attribution.assert_called_with("© OpenStreetMap contributors")
    service.set_opacity(0.55)
    service.set_height(0.75)
    service.set_enabled(True)
    session = SessionService(service.viz)
    settings = session._get_rendering_state()
    assert settings["osm_map"] == {"enabled": True, "opacity": 0.55, "height": 0.75}
    service.clear()
    service.set_scene(tmp_path / "scene.xml")
    service.saved_map = tmp_path / "saved/map.json"
    service.viz.camera_controller.set_overview_view.reset_mock()
    session._restore_rendering_state(settings)
    assert service.enabled and service.height == 0.75 and service.opacity == 0.55
    service.viz.camera_controller.set_overview_view.assert_not_called()


def test_invalid_saved_map_keeps_scene_and_can_retry_after_repair(service, tmp_path, metadata):
    service.saved_map = saved_map(tmp_path / "saved", metadata)
    texture = service.saved_map.with_name("map.png")
    texture.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    service.metadata = {}  # An offline reference has no acquisition fallback.
    entries = deepcopy(service.viz.mesh_entries)
    service.set_enabled(True)
    assert not service.enabled and service._mesh is None
    assert service._job is None and "Map unavailable" in service.status
    service.viz.renderer.ensure_object.assert_not_called()
    assert service.viz.mesh_entries == entries

    Image.new("RGB", (16, 16), "green").save(texture)
    service.set_enabled(True)
    assert service.enabled and service._job is None
    assert service.status == "Saved map reference"
    service.viz.renderer.ensure_object.assert_called_once()
    assert service.viz.mesh_entries == entries


def test_invalid_worker_map_is_not_published_and_allows_retry(service, metadata):
    service.set_enabled(True)
    job = service._job
    path = saved_map(job.root, metadata)
    path.with_name("map.png").write_bytes(b"not a PNG")
    job.finished.emit(
        {
            "event": "complete",
            "result": {"map_path": str(path), "loaded_tiles": 1, "cached_tiles": 0},
        }
    )
    assert not service.enabled and service._mesh is None
    assert not job.root.exists() and "readable PNG" in service.status
    service.viz.renderer.ensure_object.assert_not_called()
    service.set_enabled(True)
    finish(service, metadata)
    assert service.enabled and service._mesh is not None


def test_failed_renderer_publication_keeps_environment_visible(service, tmp_path, metadata):
    service.saved_map = saved_map(tmp_path / "saved", metadata)
    service.viz.renderer.ensure_object.return_value = False
    service.set_enabled(True)
    assert not service.enabled
    assert effective_entry_visibility(service.viz.mesh_entries[0])
    assert "renderer" in service.status


def test_controls_reflect_map_loading_and_publish_sliders(qtbot, service, metadata):
    controls = MapLayerControls(service)
    qtbot.addWidget(controls)
    controls.enabled.click()
    assert controls.enabled.isChecked() and controls.enabled.isEnabled()
    assert "Loading" in controls.status.text()
    finish(service, metadata)
    controls.opacity.setValue(50)
    controls.height.setValue(0.8)
    assert service.opacity == 0.5 and service.height == 0.8
    mesh = service._mesh
    for widget, text in ((controls.height, "12.50"), (controls.opacity, "45")):
        widget.setFocus()
        widget.selectAll()
        service.viz.renderer.request_redraw.reset_mock()
        qtbot.keyClicks(widget, text)
        # No Enter, focus loss, frame advance or camera gesture is required.
        service.viz.renderer.request_redraw.assert_called()
    assert service.height == 12.5 and service.opacity == 0.45
    controls.height.stepUp()
    assert service.height == 13.5
    rendered = service.viz.renderer.ensure_object.call_args.args[0]
    assert rendered.transform.translation[2] == 13.5 - mesh.vertices[0, 2]
    assert rendered.material_payload.base_color[3] == 0.45
    assert rendered.payload is mesh and service._job is None
    controls.enabled.click()
    assert not service.enabled
