"""Unit tests for the _cached_osm_query wrapper in generator/osm/scene_builder.py.

Focused on behaviours the integration smoke test does not cover:

- Empty / no_data results are cached via a sidecar .empty marker so
  sparse scenes (desert, open water, rural) stay fully offline on
  rebuilds.
- Successful results are cached as the real serializer file.
- The cache HIT path short-circuits the fetch_fn entirely (zero network
  calls on the second invocation).
- ``refresh=True`` clears a stale empty marker on the transition
  empty -> non-empty and vice versa.
"""

from __future__ import annotations

from pathlib import Path

import pytest

osmnx = pytest.importorskip("osmnx")

from generator.osm.scene_builder import (  # noqa: E402
    OsmSerializer,
    _cached_osm_query,
    _configure_osmnx_overpass_timeout,
)


def _make_text_serializer() -> OsmSerializer:
    """Serializer that writes/reads plain text — keeps tests dependency-light."""

    def to_disk(value, path: Path) -> None:
        Path(path).write_text(str(value))

    def from_disk(path: Path):
        return Path(path).read_text()

    return OsmSerializer(to_disk=to_disk, from_disk=from_disk, ext=".txt")


class _CountingFetch:
    def __init__(self, return_value):
        self.return_value = return_value
        self.calls = 0

    def __call__(self):
        self.calls += 1
        return self.return_value


def test_successful_result_roundtrips_through_cache(tmp_path):
    fetch = _CountingFetch(return_value="hello_buildings")
    serializer = _make_text_serializer()

    # First call — cache miss, fetch runs once.
    result = _cached_osm_query(
        "buildings",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"building": True},
        fetch_fn=fetch,
        serializer=serializer,
        optional=False,
        no_data_value=None,
        cache_dir=tmp_path,
        refresh=False,
    )
    assert result == "hello_buildings"
    assert fetch.calls == 1
    cache_files = sorted(p.name for p in tmp_path.iterdir())
    assert any(name.startswith("buildings_") and name.endswith(".txt") for name in cache_files)
    assert not any(name.endswith(".empty") for name in cache_files)

    # Second call — cache hit, fetch is NOT invoked.
    result2 = _cached_osm_query(
        "buildings",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"building": True},
        fetch_fn=fetch,
        serializer=serializer,
        optional=False,
        no_data_value=None,
        cache_dir=tmp_path,
        refresh=False,
    )
    assert result2 == "hello_buildings"
    assert fetch.calls == 1  # unchanged — served from disk


def test_empty_result_is_cached_via_sidecar_marker(tmp_path):
    """A definitive 'no features in this bbox+tags' answer must be cached."""
    fetch = _CountingFetch(return_value=None)
    serializer = _make_text_serializer()
    stats: dict = {}

    # First call — genuine empty, fetch runs, writes a .empty marker.
    result = _cached_osm_query(
        "barriers",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"barrier": ["wall"]},
        fetch_fn=fetch,
        serializer=serializer,
        optional=True,
        no_data_value=None,
        cache_dir=tmp_path,
        refresh=False,
        stats=stats,
    )
    assert result is None
    assert fetch.calls == 1
    markers = [p for p in tmp_path.iterdir() if p.name.endswith(".empty")]
    assert len(markers) == 1, f"expected one .empty marker, got {markers}"
    # The payload file should NOT exist — empty results don't produce one.
    payloads = [p for p in tmp_path.iterdir() if p.suffix == ".txt"]
    assert payloads == []
    assert stats == {"misses": 1}

    # Second call — warm rebuild of a sparse scene, fetch is NOT invoked.
    result2 = _cached_osm_query(
        "barriers",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"barrier": ["wall"]},
        fetch_fn=fetch,
        serializer=serializer,
        optional=True,
        no_data_value=None,
        cache_dir=tmp_path,
        refresh=False,
        stats=stats,
    )
    assert result2 is None
    assert fetch.calls == 1  # unchanged — served from the empty marker
    assert stats == {"misses": 1, "hits": 1}


def test_refresh_clears_stale_empty_marker_when_data_appears(tmp_path):
    """A previously empty bbox that now returns data must invalidate the marker."""
    serializer = _make_text_serializer()
    common_kwargs = dict(
        label="vegetation",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"natural": ["tree"]},
        serializer=serializer,
        optional=True,
        no_data_value=None,
        cache_dir=tmp_path,
    )

    # Cold run — empty response writes the marker.
    _cached_osm_query(fetch_fn=lambda: None, refresh=False, **common_kwargs)
    markers = [p for p in tmp_path.iterdir() if p.name.endswith(".empty")]
    assert len(markers) == 1

    # Refresh with a non-empty result: marker must be cleared, payload written.
    _cached_osm_query(fetch_fn=lambda: "new_trees", refresh=True, **common_kwargs)
    markers_after = [p for p in tmp_path.iterdir() if p.name.endswith(".empty")]
    payloads_after = [p for p in tmp_path.iterdir() if p.suffix == ".txt"]
    assert markers_after == [], "stale .empty marker must be cleared on refresh"
    assert len(payloads_after) == 1


def test_refresh_clears_stale_payload_when_data_disappears(tmp_path):
    """And the reverse: a previously populated bbox that now returns empty
    must drop the stale payload and write a fresh marker."""
    serializer = _make_text_serializer()
    common_kwargs = dict(
        label="water",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"water": True},
        serializer=serializer,
        optional=True,
        no_data_value=None,
        cache_dir=tmp_path,
    )

    _cached_osm_query(fetch_fn=lambda: "seine", refresh=False, **common_kwargs)
    payloads = [p for p in tmp_path.iterdir() if p.suffix == ".txt"]
    assert len(payloads) == 1

    _cached_osm_query(fetch_fn=lambda: None, refresh=True, **common_kwargs)
    payloads_after = [p for p in tmp_path.iterdir() if p.suffix == ".txt"]
    markers_after = [p for p in tmp_path.iterdir() if p.name.endswith(".empty")]
    assert payloads_after == [], "stale payload must be cleared when data disappears"
    assert len(markers_after) == 1


def test_configure_overpass_timeout_sets_requests_timeout_only():
    """OSMnx 2.1.0 reads settings.requests_timeout, not settings.timeout.

    The helper should set the modern attribute and not attempt to
    maintain a legacy alias that no longer exists on the settings
    module.
    """
    original = osmnx.settings.requests_timeout
    try:
        _configure_osmnx_overpass_timeout(240)
        assert osmnx.settings.requests_timeout == 240

        # Verify the Overpass header builder actually picks it up.
        from osmnx._overpass import _make_overpass_settings

        assert "timeout:240" in _make_overpass_settings()
    finally:
        osmnx.settings.requests_timeout = original


def test_cache_dir_none_falls_through_to_plain_query(tmp_path):
    """When cache_dir is None, the wrapper must bypass all cache logic."""
    fetch = _CountingFetch(return_value="raw_buildings")
    serializer = _make_text_serializer()

    result = _cached_osm_query(
        "buildings",
        bbox=(0.0, 0.0, 0.1, 0.1),
        tags={"building": True},
        fetch_fn=fetch,
        serializer=serializer,
        optional=False,
        no_data_value=None,
        cache_dir=None,
        refresh=False,
    )
    assert result == "raw_buildings"
    assert fetch.calls == 1
    # No cache dir was passed → nothing should land on disk.
    assert list(tmp_path.iterdir()) == []
