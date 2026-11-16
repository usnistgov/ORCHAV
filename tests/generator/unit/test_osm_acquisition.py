"""Acquisition correctness tests: real local HTTP, no public provider traffic."""

import json
import logging
import time
from pathlib import Path

import pytest

pytest.importorskip("osmnx")
from generator.osm import scene_builder as b
from generator.osm.acquisition import (
    Acquisition,
    AcquisitionCancelled,
    AcquisitionError,
    acquisition_scope,
    validate_overpass,
)
from generator.osm.publication import staged_scene
from tests.generator.osm_fixture import fixture_server, synthetic_payload

FIXTURE = synthetic_payload()


def test_unlimited_endpoint_does_not_wait_sixty_seconds(monkeypatch):
    acquisition = Acquisition()
    monkeypatch.setattr(
        acquisition,
        "_request",
        lambda *_args, **_kwargs: ({"status": 200}, b"Connected\nRate limit: 0\n"),
    )
    monkeypatch.setattr(
        acquisition, "wait", lambda *_: pytest.fail("Unlimited endpoint must not wait")
    )
    acquisition._slot()


@pytest.mark.parametrize("failure", ["timeout", "missing", "malformed"])
def test_unavailable_advisory_status_still_allows_data_request(monkeypatch, failure):
    import osmnx as ox

    monkeypatch.setattr(ox.settings, "overpass_rate_limit", True)
    acquisition = Acquisition()
    calls = []

    def request(url, **kwargs):
        calls.append(url)
        if url.endswith("/status"):
            assert kwargs["timeout"] == 8
            if failure == "timeout":
                raise AcquisitionError("Status deadline exceeded")
            return {"status": 404 if failure == "missing" else 200}, b"Unknown slot format"
        return {"status": 200}, b'{"elements":[]}'

    monkeypatch.setattr(acquisition, "_request", request)
    monkeypatch.setattr(acquisition, "wait", lambda *_: pytest.fail("No provider wait advertised"))
    assert acquisition.overpass({"data": "[timeout:45];node(1);out;"}) == {"elements": []}
    assert len(calls) == 2 and calls[-1].endswith("/interpreter")


@pytest.mark.parametrize("retry_after, delay", [(None, 30), ("0", 30), ("23", 30), ("45", 45)])
def test_status_rate_limit_still_respects_retry_after(monkeypatch, retry_after, delay):
    acquisition = Acquisition()
    waits = []
    monkeypatch.setattr(
        acquisition,
        "_request",
        lambda *_a, **_k: ({"status": 429, "retry_after": retry_after}, b""),
    )
    monkeypatch.setattr(acquisition, "wait", waits.append)
    acquisition._slot()
    assert waits == [delay]


@pytest.fixture
def acquisition_clock(monkeypatch):
    from types import SimpleNamespace

    from generator.osm import acquisition as module

    clock = SimpleNamespace(now=0.0)

    def sleep(seconds):
        clock.now += seconds

    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep))
    return clock


@pytest.mark.parametrize("rate_limit", [False, True])
@pytest.mark.parametrize("retry_after, delay", [(None, 30), ("0", 30), ("45", 45)])
def test_final_429_waits_before_layer_fallback(
    monkeypatch, acquisition_clock, rate_limit, retry_after, delay
):
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", rate_limit)
    acquisition = Acquisition(timeout=200)
    calls = []
    posts = 0

    def request(url, **_kwargs):
        nonlocal posts
        acquisition.check()
        calls.append((url, acquisition_clock.now))
        if url.endswith("/status"):
            return {"status": 200}, b"2 slots available now."
        posts += 1
        if posts <= 2:
            return {"status": 429, "retry_after": retry_after}, b"private body must not be logged"
        return {"status": 200}, b'{"elements":[]}'

    monkeypatch.setattr(acquisition, "_request", request)
    with pytest.raises(AcquisitionError, match="HTTP 429.*rate limit"):
        acquisition.overpass({"data": "[timeout:45];out;"})
    assert acquisition_clock.now == pytest.approx(2 * delay)
    assert acquisition.remaining == pytest.approx(200 - 2 * delay)
    assert acquisition.overpass({"data": "[timeout:45];out;"}) == {"elements": []}
    for index, (url, sent) in enumerate(calls):
        if url.endswith("/interpreter") and index < len(calls) - 1:
            assert calls[index + 1][1] - sent == pytest.approx(delay)
    assert posts == 3  # Two attempts for the failed layer, one for the next layer.


def test_status_429_waits_before_data_without_double_charging(monkeypatch, acquisition_clock):
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", True)
    acquisition = Acquisition(timeout=100)
    calls = []

    def request(url, **_kwargs):
        calls.append((url, acquisition_clock.now))
        if url.endswith("/status"):
            return {"status": 429, "retry_after": "0"}, b""
        return {"status": 200}, b'{"elements":[]}'

    monkeypatch.setattr(acquisition, "_request", request)
    assert acquisition.overpass({"data": "[timeout:45];out;"}) == {"elements": []}
    assert len(calls) == 2
    assert calls[1][1] - calls[0][1] == pytest.approx(30)
    assert acquisition.remaining == pytest.approx(70)


@pytest.mark.parametrize("status_endpoint", [False, True])
def test_429_wait_exhaustion_preserves_reason_and_prevents_fallback(
    monkeypatch, acquisition_clock, status_endpoint
):
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", status_endpoint)
    acquisition = Acquisition(timeout=10)
    calls = []

    def request(url, **_kwargs):
        acquisition.check()
        calls.append(url)
        return {"status": 429, "retry_after": "0"}, b"private body"

    monkeypatch.setattr(acquisition, "_request", request)
    with pytest.raises(AcquisitionError, match="HTTP 429.*remaining network budget"):
        acquisition.overpass({"data": "[timeout:45];out;"})
    assert acquisition.remaining == 0
    assert acquisition_clock.now == 0  # Do not spend time when the required wait cannot fit.
    with pytest.raises(AcquisitionError, match="budget exhausted"):
        acquisition.overpass({"data": "[timeout:45];out;"})
    assert len(calls) == 1


@pytest.mark.parametrize("status_endpoint", [False, True])
def test_429_wait_is_cancellable(monkeypatch, acquisition_clock, status_endpoint):
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", status_endpoint)
    acquisition = Acquisition(timeout=100, cancel=lambda: acquisition_clock.now >= 0.3)
    calls = []

    def request(url, **_kwargs):
        calls.append(url)
        return {"status": 429}, b""

    monkeypatch.setattr(acquisition, "_request", request)
    with pytest.raises(AcquisitionCancelled):
        acquisition.overpass({"data": "[timeout:45];out;"})
    assert len(calls) == 1
    assert acquisition_clock.now == pytest.approx(0.3)
    assert acquisition.remaining == pytest.approx(99.7)


CAPACITY_BODY = (
    b"This Overpass API server is currently at capacity. All query workers are busy\n"
    b"and the request queue is full."
)


@pytest.mark.parametrize(
    "status, body, explanation",
    [
        (502, CAPACITY_BODY, "server is at capacity"),
        (502, b"<html>private error details</html>", "temporarily unavailable"),
        (503, b"upstream unavailable", "temporarily unavailable"),
        (504, b"gateway timeout", "temporarily unavailable"),
        (502, b"x" * 4096 + CAPACITY_BODY, "temporarily unavailable"),
    ],
    ids=["capacity", "unrecognized-502", "unrecognized-503", "unrecognized-504", "bounded-prefix"],
)
def test_http_errors_explain_server_failure_without_raw_body(
    monkeypatch, caplog, status, body, explanation
):
    caplog.set_level(logging.INFO, logger="generator.osm.acquisition")
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", False)
    events = []
    acquisition = Acquisition(progress=events.append)
    monkeypatch.setattr(
        acquisition, "_request", lambda *_a, **_k: ({"status": status, "retry_after": "0"}, body)
    )
    with pytest.raises(AcquisitionError) as caught:
        acquisition.overpass({"data": "[timeout:45];out;"})
    for text in (str(caught.value), caplog.text, json.dumps(events)):
        assert f"HTTP {status}" in text
        assert acquisition.endpoint in text
        assert explanation in text
        assert "Try again later" in text
        assert "private error details" not in text
        assert "<html>" not in text
        if explanation != "server is at capacity":
            assert "server is at capacity" not in text


def test_busy_advisory_status_is_explained_but_data_can_succeed(monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="generator.osm.acquisition")
    monkeypatch.setattr(b.ox.settings, "overpass_rate_limit", True)
    events = []
    acquisition = Acquisition(progress=events.append)

    def request(url, **_kwargs):
        if url.endswith("/status"):
            return {"status": 502}, CAPACITY_BODY
        return {"status": 200}, b'{"elements":[]}'

    monkeypatch.setattr(acquisition, "_request", request)
    assert acquisition.overpass({"data": "[timeout:45];out;"}) == {"elements": []}
    for text in (caplog.text, json.dumps(events)):
        assert "server is at capacity" in text
        assert acquisition.endpoint + "/status" in text
        assert "trying the data request" in text


def test_real_http_capacity_response_is_explained_and_bounded(caplog):
    caplog.set_level(logging.INFO, logger="generator.osm.acquisition")
    events = []
    with fixture_server(FIXTURE, mode="capacity") as (endpoint, requests):
        with acquisition_scope(
            endpoint=endpoint, timeout=5, request_timeout=2, progress=events.append
        ) as acquisition:
            with pytest.raises(AcquisitionError, match="HTTP 502.*server is at capacity"):
                acquisition.overpass({"data": "[out:json][timeout:2];out;"})
    assert sum(r["method"] == "POST" for r in requests) == 2
    assert "server is at capacity" in caplog.text
    assert "server is at capacity" in json.dumps(events)


@pytest.mark.parametrize("mode", ["429-broken", "429-stall"])
def test_broken_429_body_cannot_hide_rate_limit_or_allow_retry(caplog, mode):
    caplog.set_level(logging.INFO, logger="generator.osm.acquisition")
    with fixture_server(FIXTURE, mode=mode) as (endpoint, requests):
        with acquisition_scope(endpoint=endpoint, timeout=5, request_timeout=1) as acquisition:
            with pytest.raises(AcquisitionError, match="HTTP 429.*remaining network budget"):
                acquisition.overpass({"data": "[out:json][timeout:2];out;"})
            assert acquisition.remaining == 0
            with pytest.raises(AcquisitionError, match="budget exhausted"):
                acquisition.overpass({"data": "[out:json][timeout:2];out;"})
            rate_limit = next(r for r in acquisition.requests if r.get("status") == 429)
            assert rate_limit["bytes"] == rate_limit["transfer_bytes"] == 0
            assert rate_limit["retry_after"] == "0"
    assert sum(r["method"] == "POST" for r in requests) == 1
    assert "server rate limit reached" in caplog.text
    assert "private invalid gzip body" not in caplog.text
    assert "BadGzipFile" not in caplog.text


@pytest.mark.parametrize("variant", ["valid", "truncated", "oversized"])
def test_gzip_http_keeps_length_validation_and_decoded_size_limit(tmp_path, variant):
    import gzip
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    payload = b'{"elements":[]}' * 20000
    compressed = gzip.compress(payload)
    observed = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_GET(self):
            observed.append(self.headers.get("Accept-Encoding"))
            self.send_response(200)
            self.send_header("Content-Encoding", "gzip")
            self.send_header("Content-Length", str(len(compressed)))
            self.end_headers()
            self.wfile.write(compressed if variant != "truncated" else compressed[:-8])
            self.close_connection = True

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    events = []
    acquisition = Acquisition(timeout=5, progress=events.append)
    path = tmp_path / "body"
    try:
        url = f"http://127.0.0.1:{server.server_port}/data"
        if variant == "valid":
            result = acquisition.download(url, path)
            assert path.read_bytes() == payload
            assert result["transfer_bytes"] == len(compressed) < result["bytes"]
        else:
            with pytest.raises(AcquisitionError):
                acquisition.download(
                    url, path, max_bytes=100 if variant == "oversized" else len(payload)
                )
        assert observed == ["gzip"]
        assert events and "Contacting" in events[0]["message"]
        assert not acquisition.layers  # Transport heartbeats are not layer outcomes.
    finally:
        acquisition.close()
        server.shutdown()
        server.server_close()
        thread.join()


def args(tmp_path):
    return dict(
        label="water",
        bbox=(2, 48, 2.1, 48.1),
        tags={"water": True},
        serializer=b.OsmSerializer(lambda v, p: p.write_text(v), lambda p: p.read_text(), ".txt"),
        optional=True,
        no_data_value=None,
        cache_dir=tmp_path,
        refresh=False,
    )


def test_optional_failure_recovers_without_false_empty(tmp_path):
    calls = []

    def fail():
        calls.append(1)
        raise ConnectionError("temporary outage")

    assert b._cached_osm_query(fetch_fn=fail, **args(tmp_path)) is None
    assert len(calls) == 3
    assert not list(tmp_path.glob("*.empty"))
    assert b._cached_osm_query(fetch_fn=lambda: "water", **args(tmp_path)) == "water"


def test_legacy_empty_revalidated_positive_preserved(tmp_path):
    kw = args(tmp_path)
    name = b._osm_cache_filename("water", kw["bbox"], kw["tags"], ".txt")
    marker = tmp_path / (name + ".empty")
    marker.write_text("")
    assert b._cached_osm_query(fetch_fn=lambda: "water", **kw) == "water"
    assert not marker.exists()
    assert (tmp_path / name).read_text() == "water"
    assert b._cached_osm_query(fetch_fn=lambda: pytest.fail("unexpected download"), **kw) == "water"


def test_legacy_empty_migrates_to_confirmed_empty(tmp_path):
    kw = args(tmp_path)
    name = b._osm_cache_filename("water", kw["bbox"], kw["tags"], ".txt")
    marker = tmp_path / (name + ".empty")
    marker.write_text("")
    b._cached_osm_query(fetch_fn=lambda: None, **kw)
    assert json.loads(marker.read_text())["schema_version"] == 2
    b._cached_osm_query(
        fetch_fn=lambda: pytest.fail("empty response should be reusable offline"), **kw
    )


def test_failed_refresh_keeps_payload_but_retries(tmp_path):
    kw = args(tmp_path)
    b._cached_osm_query(fetch_fn=lambda: "old", **kw)

    def fail():
        raise ConnectionError("outage")

    b._cached_osm_query(fetch_fn=fail, **{**kw, "refresh": True})
    assert list(tmp_path.glob("*.txt"))[0].read_text() == "old"
    assert b._cached_osm_query(fetch_fn=lambda: "new", **kw) == "new"
    assert not list(tmp_path.glob("*.retry"))


@pytest.mark.parametrize(
    "payload",
    [
        {},
        [],
        {"elements": [], "remark": "Query timed out"},
        {"elements": [], "error": "bad query"},
        {"elements": [{"type": "way"}]},
    ],
)
def test_bad_overpass_response_is_not_empty(payload):
    with pytest.raises(AcquisitionError):
        validate_overpass(payload)


@pytest.mark.parametrize("mode", ["html", "partial", "429"])
def test_real_http_errors_leave_cache_retryable(tmp_path, mode):
    with fixture_server(FIXTURE, mode=mode) as (endpoint, requests):
        with acquisition_scope(endpoint=endpoint, timeout=2, request_timeout=1):
            result = b._cached_osm_query(
                fetch_fn=lambda: b.ox.features_from_bbox(
                    (-77.028, 38.887, -77.024, 38.891), {"water": True}
                ),
                **args(tmp_path),
            )
            assert result is None
    assert not list(tmp_path.glob("*.empty"))
    assert not list(tmp_path.glob("*.txt"))
    assert sum(r["method"] == "POST" for r in requests) <= 2


def test_download_deadline_kills_worker_before_cleanup(tmp_path):
    with fixture_server(FIXTURE, mode="stall") as (endpoint, requests):
        c = Acquisition(timeout=1, request_timeout=0.25)
        start = time.monotonic()
        with pytest.raises(AcquisitionError, match="deadline"):
            c.download(endpoint + "/dataset", tmp_path / "temporary")
        assert time.monotonic() - start < 1.0
        assert not (tmp_path / "temporary").exists()


@pytest.mark.parametrize("mode", ["504", "504-once"])
def test_504_retry_is_visible_and_bounded(mode, caplog):
    caplog.set_level(logging.INFO, logger="generator.osm.acquisition")
    with fixture_server(FIXTURE, mode=mode) as (endpoint, requests):
        with acquisition_scope(endpoint=endpoint, timeout=5, request_timeout=2) as acquisition:
            if mode == "504":
                with pytest.raises(AcquisitionError, match="Overpass HTTP 504"):
                    acquisition.overpass({"data": "[out:json][timeout:2];out;"})
            else:
                assert acquisition.overpass({"data": "[out:json][timeout:2];out;"}) == FIXTURE
    assert sum(r["method"] == "POST" for r in requests) == 2
    assert "attempt 1/2" in caplog.text
    assert "attempt 2/2" in caplog.text
    assert "request limit" in caplog.text
    assert "HTTP 504" in caplog.text
    assert "retrying once" in caplog.text
    assert "Waiting 0.0 s" in caplog.text


def test_cancel_download_stops_worker(tmp_path):
    start = time.monotonic()
    with fixture_server(FIXTURE, mode="stall") as (endpoint, requests):
        c = Acquisition(timeout=3, request_timeout=2, cancel=lambda: time.monotonic() - start > 0.2)
        with pytest.raises(AcquisitionCancelled):
            c.download(endpoint + "/dataset", tmp_path / "temporary")
        assert time.monotonic() - start < 1
        assert not (tmp_path / "temporary").exists()


def test_adapter_restored_after_cancellation():
    from osmnx import _overpass

    original = _overpass._overpass_request
    endpoint = b.ox.settings.overpass_url
    with pytest.raises(AcquisitionCancelled):
        with acquisition_scope(endpoint="http://localhost:1234", cancel=lambda: True) as c:
            c.check()
    assert _overpass._overpass_request is original
    assert b.ox.settings.overpass_url == endpoint


def test_custom_endpoint_has_own_cache_namespace():
    key = b._osm_cache_filename("buildings", (1, 2, 3, 4), {"building": True}, ".geojson")
    with acquisition_scope(endpoint="http://internal.example/api"):
        assert (
            b._osm_cache_filename("buildings", (1, 2, 3, 4), {"building": True}, ".geojson") != key
        )


def test_staging_preserves_old_scene_and_target_on_failure(tmp_path):
    (tmp_path / "scene.xml").write_text("old scene")
    (tmp_path / "target.ply").write_text("user target")
    with pytest.raises(RuntimeError):
        with staged_scene(tmp_path) as stage:
            (stage / "scene.xml").write_text("new scene")
            raise RuntimeError("required download failed")
    assert (tmp_path / "scene.xml").read_text() == "old scene"
    assert (tmp_path / "target.ply").read_text() == "user target"
    assert not list(tmp_path.glob(".osm-build-*"))


def test_publication_rolls_back_replacement_error(tmp_path, monkeypatch):
    import generator.osm.publication as pub

    (tmp_path / "scene.xml").write_text("old scene")
    (tmp_path / "a.ply").write_text("old mesh")
    original = pub.os.replace

    def fail(source, target):
        if Path(source).name == "scene.xml" and "new" in Path(source).parts:
            raise OSError("injected publication error")
        return original(source, target)

    monkeypatch.setattr(pub.os, "replace", fail)
    with pytest.raises(OSError):
        with staged_scene(tmp_path) as stage:
            (stage / "scene.xml").write_text("new scene")
            (stage / "a.ply").write_text("new mesh")
    assert (tmp_path / "scene.xml").read_text() == "old scene"
    assert (tmp_path / "a.ply").read_text() == "old mesh"


def test_full_build_batches_preserves_holes_parts_and_warm_reuse(tmp_path, monkeypatch):
    # Native OSMnx parser, graph builder, projection, triangulation and PLY writer.
    monkeypatch.setattr(
        b, "_GlobalMLHeightIndex", lambda *a: pytest.fail("all fixture heights are tagged")
    )
    with fixture_server(FIXTURE) as (endpoint, requests):
        common = dict(
            lat=38.8887,
            lon=-77.0261,
            size_m=300,
            osm_server=endpoint,
            osm_cache_dir=tmp_path / "cache",
            height_enrichment=True,
            ground_pad_m=30,
        )
        b.build_scene_from_center(output_dir=str(tmp_path / "cold"), **common)
        assert sum(r["method"] == "POST" for r in requests) == 2
        meta = json.loads((tmp_path / "cold/osm_metadata.json").read_text())
        assert meta["acquisition"]["complete"]
        assert meta["acquisition"]["layers"]["height_enrichment"]["outcome"] == "not_needed"
        # Verify the canonical OSMnx representation retains the multipolygon hole.
        cache = list((tmp_path / "cache").glob("buildings_*.geojson"))[0]
        buildings = b._geojson_to_gdf(cache)
        assert any(len(p.interiors) == 1 for p in buildings.geometry if p.geom_type == "Polygon")
        assert list((tmp_path / "cache").glob("building_parts_*.geojson"))
        requests.clear()
        b.build_scene_from_center(output_dir=str(tmp_path / "warm"), **common)
        assert requests == []
        for file in (tmp_path / "cold/mesh").glob("*.ply"):
            assert file.read_bytes() == (tmp_path / "warm/mesh" / file.name).read_bytes()
        assert json.loads((tmp_path / "warm/osm_metadata.json").read_text())["osm_cache_stats"] == {
            "hits": 7,
            "misses": 0,
        }


def test_retry_only_failed_layer_in_partial_scene(tmp_path, monkeypatch):
    original = b._FeatureBatch.get

    def fail_water(self, label, tags):
        if label == "water":
            raise AcquisitionError("temporary water service failure")
        return original(self, label, tags)

    with fixture_server(FIXTURE) as (endpoint, requests):
        common = dict(
            lat=38.8887,
            lon=-77.0261,
            size_m=300,
            osm_server=endpoint,
            osm_cache_dir=tmp_path / "cache",
            height_enrichment=False,
            ground_pad_m=30,
        )
        monkeypatch.setattr(b._FeatureBatch, "get", fail_water)
        b.build_scene_from_center(output_dir=str(tmp_path / "scene"), **common)
        partial = json.loads((tmp_path / "scene/osm_metadata.json").read_text())
        assert partial["acquisition"]["complete"] is False
        assert partial["acquisition"]["layers"]["water"]["outcome"] == "failed"
        monkeypatch.setattr(b._FeatureBatch, "get", original)
        requests.clear()
        b.build_scene_from_center(output_dir=str(tmp_path / "scene"), **common)
        recovered = json.loads((tmp_path / "scene/osm_metadata.json").read_text())
        assert recovered["acquisition"]["complete"] is True
        assert recovered["osm_cache_stats"] == {"hits": 6, "misses": 1}
        assert sum(r["method"] == "POST" for r in requests) == 1
        assert "building" not in next(r["query"] for r in requests if r["method"] == "POST")


def test_globalml_processed_lookup_avoids_second_gzip_scan(tmp_path, monkeypatch):
    import gzip

    from pyproj import Transformer

    tile = tmp_path / "tile.csv.gz"
    record = {
        "properties": {"height": 15},
        "geometry": {
            "type": "Polygon",
            "coordinates": [
                [[2.0001, 48.0001], [2.0002, 48.0001], [2.0002, 48.0002], [2.0001, 48.0001]]
            ],
        },
    }
    with gzip.open(tile, "wt") as f:
        f.write(json.dumps(record) + "\n")
    monkeypatch.setattr(b._GlobalMLHeightIndex, "_find_tile_url", lambda *a: "fixture")
    monkeypatch.setattr(b._GlobalMLHeightIndex, "_download_tile", lambda *a: str(tile))
    transform = Transformer.from_crs("EPSG:4326", "EPSG:32631", always_xy=True)
    first = b._GlobalMLHeightIndex((2, 48, 2.001, 48.001), transform)
    assert first._tree is not None
    monkeypatch.setattr(gzip, "open", lambda *a, **k: pytest.fail("warm lookup rescanned gzip"))
    second = b._GlobalMLHeightIndex((2, 48, 2.001, 48.001), transform)
    assert second._heights.tolist() == first._heights.tolist()


def test_invalid_height_tags_still_request_enrichment():
    assert b._has_tagged_height({"height": "12"})
    assert b._has_tagged_height({"building:levels": "3"})
    for value in [None, "nan", "inf", "0", "-1", "unknown"]:
        assert not b._has_tagged_height({"height": value})


def test_provider_wait_beyond_budget_never_retries_early():
    c = Acquisition(timeout=10)
    with pytest.raises(AcquisitionError):
        c.wait(120)
    assert c.remaining == 0


@pytest.mark.parametrize("tile_size", [128, 512])
def test_terrain_cache_reuses_validated_tile(tmp_path, monkeypatch, tile_size):
    import numpy as np
    from PIL import Image

    from generator.osm.heightmap import HeightMap

    height = object.__new__(HeightMap)
    height._cache = tmp_path
    file = tmp_path / "14/1/2.tif"
    file.parent.mkdir(parents=True)
    Image.fromarray(np.full((tile_size, tile_size), 42, dtype=np.float32)).save(file)
    monkeypatch.setattr(
        Acquisition, "download", lambda *a, **k: pytest.fail("warm terrain downloaded")
    )
    result = height._download_tile(14, 1, 2)
    assert result.shape == (512, 512)
    assert np.all(result == 42)


def test_trickling_body_cannot_evade_overall_deadline(tmp_path):
    from generator.osm.acquisition import download_file

    with fixture_server(FIXTURE, mode="trickle") as (endpoint, requests):
        start = time.monotonic()
        with pytest.raises(AcquisitionError, match="deadline"):
            download_file(endpoint + "/dataset", tmp_path / "published", timeout=0.3)
        assert time.monotonic() - start < 1.0
        assert list(tmp_path.iterdir()) == []


def test_truncated_body_does_not_replace_valid_dataset(tmp_path):
    from generator.osm.acquisition import download_file

    target = tmp_path / "published"
    target.write_bytes(b"previous valid data")
    with fixture_server(FIXTURE, mode="truncated") as (endpoint, requests):
        with pytest.raises(AcquisitionError):
            download_file(endpoint + "/dataset", target, timeout=2)
    assert target.read_bytes() == b"previous valid data"
    assert list(tmp_path.iterdir()) == [target]


def test_failed_required_refresh_retains_scene_and_retry_intent(tmp_path, monkeypatch):
    import generator.osm as osm
    from generator.osm.metadata import should_reuse_generated_scene

    with fixture_server(FIXTURE) as (endpoint, requests):
        cfg = {
            "center": [38.8887, -77.0261],
            "size_m": 300,
            "ground_pad_m": 30,
            "height_enrichment": False,
            "osm_server": endpoint,
        }
        kwargs = dict(
            lat=38.8887,
            lon=-77.0261,
            size_m=300,
            ground_pad_m=30,
            height_enrichment=False,
            osm_server=endpoint,
            osm_cache_dir=tmp_path / "cache",
            output_dir=str(tmp_path / "scene"),
        )
        osm.build_scene_from_center(**kwargs)
        original = (tmp_path / "scene/scene.xml").read_bytes()

        def fail(self, label, tags):
            raise AcquisitionError("service unavailable")

        monkeypatch.setattr(b._FeatureBatch, "get", fail)
        with pytest.raises(RuntimeError, match="buildings query failed"):
            osm.build_scene_from_center(osm_refresh_cache=True, **kwargs)
        assert (tmp_path / "scene/scene.xml").read_bytes() == original
        assert not should_reuse_generated_scene(tmp_path / "scene", cfg)[0]
        attempt = json.loads((tmp_path / "scene/osm_acquisition_attempt.json").read_text())
        assert attempt["outcome"] == "failed"


def test_dataset_cap_applies_inside_longer_build_budget(tmp_path):
    from generator.osm.acquisition import download_file

    with fixture_server(FIXTURE, mode="stall") as (endpoint, requests):
        with acquisition_scope(endpoint=endpoint, timeout=5, request_timeout=3):
            started = time.monotonic()
            with pytest.raises(AcquisitionError, match="deadline"):
                download_file(endpoint + "/dataset", tmp_path / "dataset", timeout=0.25)
            assert time.monotonic() - started < 1
    assert list(tmp_path.iterdir()) == []


def test_worker_pins_dns_for_status_and_post(monkeypatch):
    from generator.osm import _http_worker

    calls = []

    def resolve(*args):
        calls.append(args)
        return [(2, 1, 6, "", ("127.0.0.1", 443))]

    monkeypatch.setattr(_http_worker.socket, "getaddrinfo", resolve)
    _http_worker.pin_dns()
    _http_worker.socket.getaddrinfo("overpass.example", 443)
    _http_worker.socket.getaddrinfo("overpass.example", 443)
    assert len(calls) == 1
