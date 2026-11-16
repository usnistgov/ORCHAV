"""OSM authoring behavior using local HTTP fixtures, never public tile traffic."""

from __future__ import annotations

import io
import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import requests
from PIL import Image
from PySide6.QtCore import QObject, Signal

from generator.osm.metadata import normalize_osm_config, should_reuse_generated_scene
from visualizer.src.authoring.compiler import ScenarioCompiler, canonical_scenario_mapping
from visualizer.src.authoring.document import ScenarioDocument
from visualizer.src.authoring.domain import ActorRole, ScenarioSourceSnapshot
from visualizer.src.authoring.osm_jobs import OsmJob
from visualizer.src.authoring.osm_panel import OsmScenePanel
from visualizer.src.authoring.osm_region import GroundMap, parse_location, validate_bbox
from visualizer.src.authoring.osm_worker import build_scene, fetch_tile, tile_expiry
from visualizer.src.authoring.persistence import load_for_authoring, save_document
from visualizer.src.authoring.viewport_port import (
    HitResult,
    KeyboardInput,
    PointerInput,
    PointerPhase,
)

PROJECT = Path(__file__).resolve().parents[3]


@contextmanager
def tile_server(*, max_age=604800, delay=0, image_format="PNG"):
    records = []
    image = Image.new("RGB", (256, 256), "#f2ebd9")
    image.paste("#59af55", (0, 0, 128, 128))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    data = buffer.getvalue()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            records.append(dict(self.headers))
            if delay:
                import time

                time.sleep(delay)
            code = 304 if self.headers.get("If-None-Match") == '"fixture"' else 200
            self.send_response(code)
            self.send_header("Cache-Control", f"max-age={max_age}")
            self.send_header("ETag", '"fixture"')
            self.send_header("Content-Length", str(len(data) if code == 200 else 0))
            try:
                self.end_headers()
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                return
            if code == 200:
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", records
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


class FakeJob(QObject):
    progress = Signal(str)
    finished = Signal(object)

    def __init__(self, operation, config, parent=None):
        super().__init__(parent)
        self.operation, self.config = operation, config
        self.cancelled = False

    def start(self):
        pass

    def cancel(self):
        self.cancelled = True

    def close(self):
        self.cancel()


def complete_scene(root, config):
    root.mkdir(parents=True)
    (root / "scene.xml").write_text(
        '<scene version="3.0.0"><shape type="cube" id="building"/></scene>', encoding="utf-8"
    )
    (root / "osm_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": 5,
                "osm_config": normalize_osm_config(config),
                "acquisition": {"complete": True},
                "artifacts": {
                    "scene_xml": "scene.xml",
                    "street_network_graphml": None,
                    "mesh_files": [],
                },
            }
        ),
        encoding="utf-8",
    )
    return root


def test_map_coordinates_and_reverse_drag_roundtrip():
    ground = GroundMap(38.8895, -77.025, 16)
    p = ground.world(38.891, -77.022)
    assert ground.geographic(*p) == pytest.approx((38.891, -77.022), abs=1e-10)
    assert ground.selection((200, 100), (-100, -150)) == ground.selection((-100, -150), (200, 100))
    west, south, east, north = ground.bounds
    assert west < 0 < east and south < 0 < north
    assert east - west == pytest.approx(north - south)
    assert GroundMap(80, 179.9).default_selection()[2] == 80
    assert GroundMap(-80, -179.9).default_selection()[0] == -80
    assert parse_location("https://www.openstreetmap.org/#map=17/38.89/-77.02") == (
        38.89,
        -77.02,
        17,
    )


@pytest.mark.parametrize("value", ["London", "nan, 2", "81, 0", "0, 181", "1,2,3"])
def test_invalid_location(value):
    with pytest.raises(ValueError):
        parse_location(value)


@pytest.mark.parametrize(
    "bbox", [(0, 0, 0, 1), (0, 0, 1, 1), (0, 179, 1, -179), (0, 0, float("nan"), 1)]
)
def test_invalid_or_excessive_region(bbox):
    with pytest.raises(ValueError):
        validate_bbox(bbox)


def test_tile_cache_and_conditional_revalidation(tmp_path):
    with tile_server(max_age=0) as (url, records), requests.Session() as session:
        tile, reused = fetch_tile(session, url + "/tile", tmp_path, lambda: False)
        assert not reused and tile.getpixel((0, 0)) == (89, 175, 85)
        _, reused = fetch_tile(session, url + "/tile", tmp_path, lambda: False)
        assert reused and records[-1]["If-None-Match"] == '"fixture"'
        assert records[0]["User-Agent"].startswith("ORCHAV/")
    assert tile_expiry({}, 100) == 100 + 7 * 86400
    assert tile_expiry({"Cache-Control": "no-cache"}, 100) == 100


def test_tile_network_rejects_jpeg_without_caching(tmp_path):
    with tile_server(image_format="JPEG") as (url, records), requests.Session() as session:
        with pytest.raises(Image.UnidentifiedImageError):
            fetch_tile(session, url + "/tile.png", tmp_path, lambda: False)
        assert len(records) == 1
    assert not list(tmp_path.iterdir())


def test_corrupt_fresh_png_cache_discards_validators_and_fetches_png(tmp_path):
    with tile_server() as (url, records), requests.Session() as session:
        fetch_tile(session, url + "/tile.png", tmp_path, lambda: False)
        cached = next(tmp_path.glob("*.png"))
        Image.new("RGB", (256, 256), "red").save(cached, format="JPEG")
        tile, reused = fetch_tile(session, url + "/tile.png", tmp_path, lambda: False)
        assert not reused and len(records) == 2
        assert "If-None-Match" not in records[-1]
        assert tile.getpixel((0, 0)) == (89, 175, 85)
        with Image.open(cached) as image:
            assert image.format == "PNG"


def test_revalidated_cache_rejects_jpeg_bytes(tmp_path):
    with tile_server(max_age=0) as (url, records), requests.Session() as session:
        fetch_tile(session, url + "/tile.png", tmp_path, lambda: False)
        cached = next(tmp_path.glob("*.png"))
        Image.new("RGB", (256, 256), "red").save(cached, format="JPEG")
        with pytest.raises(Image.UnidentifiedImageError):
            fetch_tile(session, url + "/tile.png", tmp_path, lambda: False)
        assert records[-1]["If-None-Match"] == '"fixture"'


def test_map_job_requests_only_nine_visible_tiles_then_reuses_cache(qtbot, tmp_path, monkeypatch):
    monkeypatch.setenv("ORCHAV_CACHE_DIR", str(tmp_path))
    with tile_server() as (url, records):
        config = {
            "map": {"latitude": 38.8895, "longitude": -77.025, "zoom": 16},
            "cache": str(tmp_path / "tiles"),
            "tile_url": url + "/{z}/{x}/{y}.png",
        }
        for expected_cached in (0, 9):
            job = OsmJob("map", config, cache_root=tmp_path)
            results = []
            job.finished.connect(results.append)
            job.start()
            qtbot.waitUntil(lambda: bool(results), timeout=30000)
            assert results[0]["event"] == "complete", results
            assert results[0]["result"]["cached_tiles"] == expected_cached
            assert Path(results[0]["result"]["texture"]).is_file()
            job.close()
        assert len(records) == 9


def test_panel_selection_invalid_drag_cancel_and_stale_map(qtbot):
    panel = OsmScenePanel(job_factory=FakeJob)
    qtbot.addWidget(panel)
    assert panel.checks["green_vegetation"].isChecked()
    panel.sync_config({"green_vegetation": False})
    assert not panel.checks["green_vegetation"].isChecked()
    panel.sync_config({})
    assert panel.checks["green_vegetation"].isChecked()
    panel.show_map()
    assert not panel.build_button.isEnabled()
    panel.build()
    assert panel._build_job is None
    previous_job = panel._map_job
    first_bbox = panel.bbox
    panel.location.setText("48.85, 2.35")
    panel.show_map()
    previous_job.finished.emit({"event": "complete", "result": {"texture": "wrong.png"}})
    assert panel._texture is None and panel.bbox != first_bbox
    port = SimpleNamespace(begin_drag_plane=lambda *_: None, end_drag=lambda: None)
    original = panel.bbox
    for phase, point in ((PointerPhase.DOWN, (-200, -200, 0)), (PointerPhase.MOVE, (100, 100, 0))):
        panel.handle_input(PointerInput(phase, (0, 0), button=1), port)
        panel.handle_input(HitResult(point), port)
    assert panel.bbox != original
    panel.handle_input(KeyboardInput("Escape", True), port)
    assert panel.bbox == original
    for phase, point in ((PointerPhase.DOWN, (0, 0, 0)), (PointerPhase.UP, (1, 1, 0))):
        panel.handle_input(PointerInput(phase, (0, 0), button=1), port)
        panel.handle_input(HitResult(point), port)
    assert panel.bbox is None and not panel.build_button.isEnabled()
    panel.close_jobs()


def test_document_adoption_preserves_actors_unknown_settings_and_undo(tmp_path):
    from dataclasses import replace

    document = ScenarioDocument.new()
    target = document.add_default_actor(ActorRole.TARGET)
    document._scenario = replace(
        document.scenario,
        source_snapshot=ScenarioSourceSnapshot.from_mapping({"debug_level": "WARNING"}),
    )
    original = document.scenario
    config = {
        "bbox": list(GroundMap(38.8895, -77.025).default_selection()),
        "height_enrichment": False,
    }
    root = complete_scene(tmp_path / "built", config)
    document.install_osm_scene(config, root, project_root=PROJECT)
    assert document.actors == (target,)
    mapping = canonical_scenario_mapping(document.scenario)
    assert mapping["scene"]["osm"] == config and mapping["debug_level"] == "WARNING"
    assert {d.relative_path for d in document.scenario.dependencies} == {
        "scene.xml",
        "osm_metadata.json",
    }
    document.undo()
    assert document.scenario == original
    document.redo()
    assert document.scenario.scene.source == "osm"
    (root / "scene.xml").unlink()
    before = document.scenario
    with pytest.raises(ValueError, match="incomplete"):
        document.install_osm_scene(config, root, project_root=PROJECT)
    assert document.scenario == before


@pytest.mark.parametrize("refresh_source", ["environment", "config"])
def test_refreshed_scene_completion_adoption_and_offline_save(
    tmp_path, monkeypatch, refresh_source
):
    from dataclasses import replace

    import generator.osm.scenario_extension as extension

    monkeypatch.delenv("OSM_REFRESH_CACHE", raising=False)
    config = {
        "bbox": list(GroundMap(38.8895, -77.025).default_selection()),
        "height_enrichment": False,
    }
    if refresh_source == "environment":
        monkeypatch.setenv("OSM_REFRESH_CACHE", "1")
    else:
        config["refresh_cache"] = True

    def completed_build(config, root, **_kwargs):
        return complete_scene(root, config) / "scene.xml"

    monkeypatch.setattr(extension, "build_configured_scene", completed_build)
    result = build_scene(config, tmp_path, lambda: False, lambda _event: None)
    root = Path(result["scene_root"])
    assert should_reuse_generated_scene(root, config) == (
        False,
        "explicit OSM cache refresh requested",
    )

    # Completion must not bypass refresh on later generator runs, but authoring
    # previews and saves must use the artifacts already produced by this build.
    monkeypatch.setattr(
        extension, "build_configured_scene", lambda *_a, **_k: pytest.fail("Implicit OSM rebuild")
    )
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    before = document.scenario
    document.install_osm_scene(config, root, project_root=PROJECT)
    document.undo()
    assert document.scenario == before
    document.redo()
    compiler = ScenarioCompiler(PROJECT)
    compiled = compiler.compile(document, scenario_directory=tmp_path)
    assert compiled.valid, compiled.issues
    assert compiled.resolved_scene_path == str(root / "scene.xml")
    assert compiled.mapping["scene"]["osm"] == config
    # An already-local scene also validates without a staged dependency bundle.
    unstaged = compiler.compile(
        replace(document.scenario, dependencies=()), scenario_directory=root
    )
    assert unstaged.valid, unstaged.issues
    destination = save_document(document, tmp_path / "saved", compiler=compiler)
    reopened = load_for_authoring(destination).document
    assert reopened is not None
    assert canonical_scenario_mapping(reopened.scenario)["scene"]["osm"] == config
    assert compiler.compile(reopened, scenario_directory=destination.parent).valid
    assert not should_reuse_generated_scene(destination.parent, config)[0]
    monkeypatch.delenv("OSM_REFRESH_CACHE", raising=False)
    assert should_reuse_generated_scene(destination.parent, {**config, "refresh_cache": False})[0]


@pytest.mark.parametrize(
    "defect",
    ["malformed_metadata", "incomplete_acquisition", "missing_mesh", "failed_rebuild"],
)
def test_refresh_does_not_accept_invalid_completed_artifacts(tmp_path, monkeypatch, defect):
    import generator.osm.scenario_extension as extension

    monkeypatch.setenv("OSM_REFRESH_CACHE", "1")
    config = {"bbox": list(GroundMap(38.8895, -77.025).default_selection())}

    def invalid_build(config, root, **_kwargs):
        complete_scene(root, config)
        metadata_path = root / "osm_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if defect == "malformed_metadata":
            metadata_path.write_text("{unfinished", encoding="utf-8")
        elif defect == "incomplete_acquisition":
            metadata["acquisition"]["complete"] = False
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        elif defect == "missing_mesh":
            metadata["artifacts"]["mesh_files"] = ["mesh/missing.ply"]
            metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
        else:
            (root / "osm_acquisition_attempt.json").write_text(
                json.dumps({"outcome": "failed", "osm_config": normalize_osm_config(config)}),
                encoding="utf-8",
            )
        return root / "scene.xml"

    monkeypatch.setattr(extension, "build_configured_scene", invalid_build)
    with pytest.raises(RuntimeError, match="incomplete"):
        build_scene(config, tmp_path, lambda: False, lambda _event: None)
    document = ScenarioDocument.new()
    before = document.scenario
    with pytest.raises(ValueError, match="incomplete"):
        document.install_osm_scene(config, tmp_path / "scene", project_root=PROJECT)
    assert document.scenario == before


def test_incomplete_scene_validation_never_starts_download(tmp_path, monkeypatch):
    from dataclasses import replace

    import generator.osm.scenario_extension as extension
    from visualizer.src.authoring.domain import SceneReference

    document = ScenarioDocument.new()
    config = {"bbox": list(GroundMap(38.8895, -77.025).default_selection())}
    root = complete_scene(tmp_path / "built", config)
    (root / "osm_metadata.json").unlink()
    document._scenario = replace(
        document.scenario,
        scene=SceneReference("osm", "scene.xml"),
        source_snapshot=ScenarioSourceSnapshot.from_mapping(
            {"scene": {"source": "osm", "id": "scene.xml", "osm": config}}
        ),
    )
    monkeypatch.setattr(
        extension, "build_configured_scene", lambda *_a, **_k: pytest.fail("Implicit OSM download")
    )
    result = ScenarioCompiler(PROJECT).compile(document, scenario_directory=root)
    assert "scene.osm.build_required" in {issue.code for issue in result.issues}


@pytest.mark.parametrize("refresh_cache", [False, True])
def test_real_osm_worker_save_reopen_and_offline_reuse(qtbot, tmp_path, monkeypatch, refresh_cache):
    pytest.importorskip("osmnx")
    pytest.importorskip("open3d")
    import generator.osm.scenario_extension as extension
    from tests.generator.osm_fixture import fixture_server, synthetic_payload

    monkeypatch.setenv("ORCHAV_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("OSM_REFRESH_CACHE", raising=False)
    if refresh_cache:
        monkeypatch.setenv("OSM_REFRESH_CACHE", "1")
    fixture = synthetic_payload()
    with fixture_server(fixture) as (endpoint, records):
        ground = GroundMap(38.8887, -77.0261)
        config = {
            "bbox": list(ground.selection((-150, -150), (150, 150))),
            "ground_pad_m": 30,
            "height_enrichment": False,
            "osm_server": endpoint,
        }
        image_path = tmp_path / "visible-map.png"
        Image.new("RGB", (768, 768), "#c3e4ac").save(image_path)
        from dataclasses import asdict

        worker_config = {
            **config,
            "_ground_map": {"texture": str(image_path), "map": asdict(ground)},
        }
        job = OsmJob("build", worker_config, cache_root=tmp_path)
        results = []
        job.finished.connect(results.append)
        job.start()
        qtbot.waitUntil(lambda: bool(results), timeout=90000)
        assert results[0]["event"] == "complete", (
            results,
            (job.root / "worker.log").read_text(encoding="utf-8", errors="replace"),
        )
        scene_root = Path(results[0]["result"]["scene_root"])
        assert sum(r["method"] == "POST" for r in records) == 2
        job.close()
    # The fixture server is now closed; saving and previewing must remain offline.
    monkeypatch.setattr(
        extension, "build_configured_scene", lambda *_a, **_k: pytest.fail("Implicit OSM rebuild")
    )
    document = ScenarioDocument.new()
    document.add_default_actor(ActorRole.TX)
    document.add_default_actor(ActorRole.RX)
    document.install_osm_scene(config, scene_root, project_root=PROJECT)
    compiler = ScenarioCompiler(PROJECT)
    result = compiler.compile(document, scenario_directory=tmp_path)
    assert result.scene_assets and result.valid, result.issues
    import xml.etree.ElementTree as ET

    from visualizer.src.materials.catalog import material_preset

    vegetation = next(asset for asset in result.scene_assets if asset.name.endswith("_vegetation"))
    assert vegetation.material.base_color[:3] == tuple(material_preset("vegetation")["color"])
    xml = ET.parse(scene_root / "scene.xml")
    shape = xml.find(".//shape[@id='mesh-vegetation']")
    assert shape is not None and shape.find("ref").get("id") == "mat-itu_wood"
    destination = save_document(document, tmp_path / "saved", compiler=compiler)
    reopened = load_for_authoring(destination).document
    assert reopened is not None
    assert canonical_scenario_mapping(reopened.scenario)["scene"]["osm"] == config
    assert should_reuse_generated_scene(destination.parent, config)[0] is (not refresh_cache)
    assert all(
        d.source_path.is_relative_to(destination.parent) for d in reopened.scenario.dependencies
    )
    assert compiler.compile(reopened, scenario_directory=destination.parent).valid
    assert (destination.parent / "builder_map/map.png").is_file()
    from visualizer.src.authoring.map_overlay import MAP_NAME

    compiled = compiler.compile(reopened, scenario_directory=destination.parent)
    overlay = next(asset for asset in compiled.scene_assets if asset.name == MAP_NAME)
    assert overlay.material.shader_variant == "reference_image"
    assert Path(overlay.material.texture_path).is_relative_to(destination.parent)
    metadata = json.loads((destination.parent / "osm_metadata.json").read_text())
    from pyproj import Transformer

    transform = Transformer.from_crs("EPSG:4326", metadata["utm_crs"], always_xy=True)
    lat, lon = ground.geographic(-180, -180)
    x, y = transform.transform(lon, lat)
    assert overlay.mesh.vertices[0, :2] == pytest.approx(
        [x - metadata["origin_utm"][0], y - metadata["origin_utm"][1]], abs=1e-3
    )
    assert np.ptp(overlay.mesh.vertices[:, :2], axis=0).max() < 380
    west, south, east, north = ground.bounds
    assert overlay.mesh.triangle_uvs[0] == pytest.approx(
        [(-180 - west) / (east - west), 1 - (-180 - south) / (north - south)]
    )
    reopened.set_scene(None)
    assert not any(d.kind == "builder_map" for d in reopened.scenario.dependencies)
    reopened.undo()
    assert any(d.kind == "builder_map" for d in reopened.scenario.dependencies)


def test_reference_map_displays_with_surface_textures_disabled(tmp_path, monkeypatch):
    from visualizer.src.backends.pygfx_scene_helpers import apply_texture_policy_to_material_payload
    from visualizer.src.types.render_payloads import MaterialPayload

    path = tmp_path / "map.png"
    Image.new("RGB", (256, 256), "green").save(path)
    monkeypatch.setenv("ORCHAV_DISABLE_TEXTURES", "1")
    surface = MaterialPayload(shader="unlit", texture_path=str(path))
    reference = MaterialPayload(
        shader="unlit", texture_path=str(path), shader_variant="reference_image"
    )
    assert apply_texture_policy_to_material_payload(surface).texture_path is None
    assert apply_texture_policy_to_material_payload(reference).texture_path == str(path)


def test_cancel_real_map_process_stops_job_without_publishing(qtbot, tmp_path):
    from PySide6.QtCore import QProcess

    with tile_server(delay=4) as (url, records):
        job = OsmJob(
            "map",
            {
                "map": {"latitude": 38.8895, "longitude": -77.025, "zoom": 16},
                "cache": str(tmp_path / "tiles"),
                "tile_url": url + "/{z}/{x}/{y}.png",
            },
            cache_root=tmp_path,
        )
        results = []
        job.finished.connect(results.append)
        job.start()
        qtbot.waitUntil(lambda: bool(records), timeout=30000)
        job.cancel()
        qtbot.waitUntil(lambda: bool(results), timeout=7000)
        assert results[0]["event"] == "cancelled"
        assert job.process.state() == QProcess.NotRunning
        assert not (job.root / "map.png").exists()
        job.close()
