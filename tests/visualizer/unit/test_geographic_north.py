"""Compass availability and map-projection convergence come from actual metadata."""

import builtins
import math

import pytest

from visualizer.src.scene.geographic_north import geographic_north


@pytest.mark.parametrize(
    "metadata",
    [
        None,
        {},
        {"artifacts": {"street_network_graphml": "streets.graphml"}},
        {"center_latlon": [48.8, 2.3]},
    ],
)
def test_non_geographic_sources_never_infer_north(metadata):
    result = geographic_north(metadata)
    assert result.direction_xy is None and "no geographic" in result.reason


@pytest.mark.parametrize(
    "lon,lat,epsg", [(2.3325, 48.87175, 32631), (3, 48.8, 32631), (18.4, -33.9, 32734)]
)
def test_geographic_north_matches_crs_convergence_at_actual_origin(lon, lat, epsg):
    pyproj = pytest.importorskip("pyproj")
    x, y = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform(lon, lat)
    # No graph and no center coordinates are necessary. A stale center must
    # not override the actual local coordinate origin.
    result = geographic_north(
        {"utm_crs": f"EPSG:{epsg}", "origin_utm": [x, y, 0], "center_latlon": [0, 0]}
    )
    assert result.direction_xy is not None and not result.reason
    assert math.hypot(*result.direction_xy) == pytest.approx(1)
    convergence = pyproj.Proj(f"EPSG:{epsg}").get_factors(lon, lat).meridian_convergence
    assert math.degrees(math.atan2(*result.direction_xy)) == pytest.approx(-convergence, abs=3e-6)
    assert result.label == "Geographic north at scene origin"


@pytest.mark.parametrize(
    "metadata",
    [
        {"utm_crs": "EPSG:32631"},
        {"utm_crs": "EPSG:32631", "origin_utm": [float("nan"), 0]},
        {"utm_crs": "EPSG:4326", "origin_utm": [2.3, 48.8]},
        {"utm_crs": "EPSG:2263", "origin_utm": [1000000, 200000]},  # Feet, not local metres.
        {"utm_crs": "not a CRS", "origin_utm": [0, 0]},
        {"utm_crs": "EPSG:32631", "origin_utm": [1e300, 1e300]},
    ],
)
def test_invalid_or_nonmetric_geographic_metadata_is_unavailable(metadata):
    result = geographic_north(metadata)
    assert result.direction_xy is None and result.reason


def test_missing_optional_geographic_dependency_has_explicit_reason(monkeypatch):
    original = builtins.__import__

    def without_pyproj(name, *args, **kwargs):
        if name == "pyproj":
            raise ImportError("pyproj unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_pyproj)
    result = geographic_north({"utm_crs": "EPSG:32631", "origin_utm": [451052, 5413414]})
    assert result.direction_xy is None and "pyproj" in result.reason
