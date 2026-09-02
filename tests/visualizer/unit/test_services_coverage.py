from types import SimpleNamespace
from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.coverage.hdf5 import CoverageHDF5Reader
from shared.coverage.schema import (
    COVERAGE_FRAME_GENERATION_ID_ATTR,
    COVERAGE_FRAME_SET_ID_ATTR,
    COVERAGE_HDF5_STORAGE_LAYOUT,
)
from shared.frames.provider_base import DataProvider, ProviderInfo
from visualizer.src.services.coverage_service import CoverageService


def test_coverage_service_cache_put_get():
    service = CoverageService(max_cache_size=2)
    key = "test_key"
    vertices = np.zeros((2, 3))
    triangles = np.zeros((2, 3), dtype=np.int32)
    colors = np.ones((2, 3))

    assert service.get_mesh(key) is None
    service.put_mesh(key, vertices, triangles, colors)
    cached = service.get_mesh(key)
    assert cached is not None
    assert isinstance(cached, tuple)
    assert service.stats()["hits"] >= 1


def test_coverage_service_clear_and_stats():
    service = CoverageService(max_cache_size=1)
    key = "a"
    service.put_mesh(key, np.zeros((1, 3)), np.zeros((1, 3), dtype=np.int32), np.zeros((1, 3)))
    stats_before = service.stats()
    assert stats_before["cache_size"] >= 1
    service.clear()
    stats_after = service.stats()
    assert stats_after["cache_size"] == 0


def test_compute_cache_key_variation():
    service = CoverageService()
    coverage_data = {
        "grid_origin": np.zeros(3),
        "grid_spacing": np.ones(3),
        "grid_shape": np.array([1, 1, 1]),
        "values": np.array([0.0]),
        "value_min": 0,
        "value_max": 1,
        "metric_name": "signal",
    }
    key_a = service.compute_cache_key(coverage_data, 0, "nearest")
    key_b = service.compute_cache_key(coverage_data, 1, "nearest")
    coverage_data["metric_name"] = "other_signal"
    key_c = service.compute_cache_key(coverage_data, 0, "nearest")
    assert key_a != key_b
    assert key_a != key_c


def test_isoline_cache_is_bounded_and_cleared_with_mesh_cache():
    service = CoverageService(max_cache_size=1)
    points = np.asarray([[0.0, 0.0, 1.5], [1.0, 1.0, 1.5]])
    lines = np.asarray([[0, 1]], dtype=np.int32)
    colors = np.asarray([[0.1, 0.1, 0.1]])
    key_a = service.compute_isoline_cache_key("mesh-a", (1.0, 2.0), 1.5)
    key_b = service.compute_isoline_cache_key("mesh-b", (1.0, 2.0), 1.5)

    service.put_isolines(key_a, points, lines, colors, (1.0, 2.0))
    cached = service.get_isolines(key_a)
    assert cached is not None
    assert cached[3] == (1.0, 2.0)
    service.put_isolines(key_b, points, lines, colors, (1.0, 2.0))

    assert service.get_isolines(key_a) is None
    assert service.stats()["isoline_evictions"] == 1
    assert service.stats()["isoline_cache_size"] == 1
    service.clear()
    assert service.stats()["isoline_cache_size"] == 0


def test_interpolate_values_delegates(monkeypatch):
    service = CoverageService()
    called = {}

    def fake_interpolate(values, interpolation):
        called["args"] = (values.copy(), interpolation)
        return np.full_like(values, 2.0)

    monkeypatch.setattr(service.cache, "interpolate_coverage_values", fake_interpolate)
    data = np.array([[1.0, 2.0]])

    result = service.interpolate_values(data, "linear")
    assert called["args"][1] == "linear"
    assert np.all(result == 2.0)


def test_serving_tx_metric_range_ignores_no_service_sentinel():
    service = CoverageService()
    coverage_data = {
        "tx_names": ["TX1", "TX2"],
        "metric_layers": {
            "serving_tx": np.array([[[-1, 0], [1, -1]]], dtype=np.int16),
        },
    }

    service.select_metric_layer(coverage_data, "serving_tx")

    assert coverage_data["serving_tx_count"] == 2
    assert coverage_data["value_min"] == 0.0
    assert coverage_data["value_max"] == 1.0


def test_log_stats_delegates(monkeypatch):
    service = CoverageService()
    called = {"logged": False}

    def fake_log():
        called["logged"] = True

    monkeypatch.setattr(service.cache, "log_stats", fake_log)
    service.log_stats()
    assert called["logged"]


def test_load_v2_coverage_hdf5_exposes_stored_and_derived_layers(tmp_path, monkeypatch):
    coverage_file = tmp_path / "coverage_maps.h5"
    path_gain = np.ones((1, 2, 2, 2, 3), dtype=np.float32) * 1e-8
    rss_w = np.ones_like(path_gain) * 1e-11
    rss_w[:, :, 1] = 2e-11

    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([3, 2, 2], dtype=np.int32),
            "heights": np.array([1.5, 10.0], dtype=np.float32),
            "path_gain_linear": path_gain,
            "stored_values": {"path_gain_linear": path_gain, "rss_w": rss_w},
            "derived": {
                "best_path_loss_db": np.ones((1, 2, 2, 3), dtype=np.float32) * 80.0,
                "path_loss_db": np.ones((1, 2, 2, 2, 3), dtype=np.float32) * 90.0,
                "serving_tx": np.zeros((1, 2, 2, 3), dtype=np.int16),
            },
            "metric_name": "best_path_loss_db",
            "tx_positions": np.array(
                [[0.0, 0.0, 10.0], [10.0, 0.0, 10.0]],
                dtype=np.float32,
            ),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["TX1", "TX2"],
            "rx_names": [],
            "tx_power_dbm": np.array([0.0, 3.0], dtype=np.float32),
            "value_min": 80.0,
            "value_max": 80.0,
            "metadata": {
                "tx_mode": "per_tx",
                "metrics_store": ["path_gain_linear", "rss_w"],
                "metrics_derived": ["path_loss_db", "sinr_db", "serving_tx"],
                "noise_power_w": 1e-12,
                "bandwidth_hz": 2e9,
                "temperature_k": 293.0,
            },
        },
        coverage_file,
        compression=None,
    )
    with h5py.File(coverage_file, "r") as f:
        assert f.attrs["coverage_storage_layout"] == COVERAGE_HDF5_STORAGE_LAYOUT
        assert list(f["values"].keys()) == ["path_gain_linear"]
        assert list(f["derived"].keys()) == ["serving_tx"]
        available = f["metadata"].attrs["available_metrics"]
        assert "rss_w/TX2" in available
        assert "path_loss_db/TX1" in available

    metric_reads = []
    original_read_metric = CoverageHDF5Reader.read_metric

    def observed_read_metric(reader, metric_name, **kwargs):
        metric_reads.append((metric_name, kwargs.get("height_indices")))
        return original_read_metric(reader, metric_name, **kwargs)

    monkeypatch.setattr(CoverageHDF5Reader, "read_metric", observed_read_metric)
    service = CoverageService()
    data = service._load_v2_coverage_hdf5(coverage_file)

    assert data["metric_name"] == "best_path_loss_db"
    assert data["values_3d"].shape == (1, 2, 3)
    assert data["_active_height_index"] == 0
    assert metric_reads == [("best_path_loss_db", 0)]
    assert "path_gain_linear/TX1" in data["available_metrics"]
    assert "serving_path_gain_linear" in data["available_metrics"]
    assert "rss_w/TX2" in data["available_metrics"]
    assert "path_loss_db/TX1" in data["available_metrics"]
    assert "sinr_db" in data["available_metrics"]
    assert "sinr_db/TX1" in data["available_metrics"]
    assert "sinr_db/TX2" in data["available_metrics"]
    assert "serving_tx" in data["available_metrics"]

    service.select_metric_layer(data, "rss_w/TX2")

    assert data["metric_name"] == "rss_w/TX2"
    assert data["values_3d"].shape == (1, 2, 3)
    assert np.allclose(data["values_3d"], 2e-11)
    assert all(height_index is not None for _, height_index in metric_reads)
    metric_range = (data["value_min"], data["value_max"])

    service.select_height_layer(data, 1)

    assert data["_active_height_index"] == 1
    assert data["values_3d"].shape == (1, 2, 3)
    assert (data["value_min"], data["value_max"]) == metric_range
    assert metric_reads[-1] == ("rss_w/TX2", 1)


@pytest.mark.parametrize(
    ("stored_primary", "expected_primary"),
    [
        ("path_gain_linear", "path_gain_linear/BaseStation"),
        ("path_gain_db", "path_gain_db/BaseStation"),
        ("path_loss_db", "path_loss_db/BaseStation"),
        ("rss_w", "rss_w/BaseStation"),
        ("rss_dbm", "rss_dbm/BaseStation"),
        ("best_path_loss_db", "path_loss_db/BaseStation"),
        ("best_rss_dbm", "rss_dbm/BaseStation"),
        ("sum_rss_dbm", "rss_dbm/BaseStation"),
        ("sinr_db", "sinr_db/BaseStation"),
        ("serving_tx", "path_gain_linear/BaseStation"),
    ],
)
def test_load_single_tx_coverage_hides_redundant_metrics_and_maps_primary(
    tmp_path,
    stored_primary,
    expected_primary,
):
    coverage_file = tmp_path / f"{stored_primary}.h5"
    path_gain = np.ones((1, 1, 1, 1, 2), dtype=np.float32) * 1e-8
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([2, 1, 1], dtype=np.int32),
            "heights": np.array([1.5], dtype=np.float32),
            "path_gain_linear": path_gain,
            "derived": {
                "serving_tx": np.zeros((1, 1, 1, 2), dtype=np.int16),
            },
            "metric_name": stored_primary,
            "tx_positions": np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["BaseStation"],
            "rx_names": [],
            "tx_power_dbm": np.array([0.0], dtype=np.float32),
            "value_min": 0.0,
            "value_max": 0.0,
            "metadata": {
                "metrics_store": ["path_gain_linear"],
                "metrics_derived": [
                    "path_loss_db",
                    "rss_dbm",
                    "sum_rss_dbm",
                    "sinr_db",
                    "serving_tx",
                    "tx_margin_db",
                ],
            },
        },
        coverage_file,
        compression=None,
    )

    service = CoverageService()
    data = service._load_v2_coverage_hdf5(coverage_file)

    assert data["available_metrics"][0] == "path_gain_linear/BaseStation"
    assert expected_primary in data["available_metrics"]
    assert not {
        "best_path_loss_db",
        "best_rss_dbm",
        "sum_rss_dbm",
        "serving_tx",
        "tx_margin_db",
    }.intersection(data["available_metrics"])
    assert data["metric_name"] == expected_primary


def test_load_coverage_map_uses_fixed_scenario_coverage_path(tmp_path, monkeypatch):
    scenario_root = tmp_path / "scenario"
    coverage_file = scenario_root / "coverage" / "coverage_maps.h5"
    coverage_file.parent.mkdir(parents=True)
    coverage_file.touch()
    loaded_paths = []
    availability = []
    service = CoverageService()
    payload = {
        "grid_origin": np.asarray([0.0, 0.0, 1.5]),
        "grid_spacing": np.asarray([1.0, 1.0, 1.0]),
        "grid_shape": np.asarray([1, 1, 1]),
        "heights": [1.5],
        "available_metrics": ["signal"],
        "metric_name": "signal",
        "metric_layers": {"signal": np.asarray([[[2.0]]], dtype=np.float32)},
        "value_min": 2.0,
        "value_max": 2.0,
    }

    def _load(path):
        loaded_paths.append(path)
        return payload

    monkeypatch.setattr(service, "_load_v2_coverage_hdf5", _load)
    viz = SimpleNamespace(
        app_state=SimpleNamespace(),
        set_state=Mock(),
        ui_manager=SimpleNamespace(
            panels={},
            set_coverage_data_available=lambda available: availability.append(bool(available)),
        ),
    )

    loaded = service.load_coverage_map(scenario_root, viz)

    assert loaded is True
    assert loaded_paths == [coverage_file]
    assert viz.coverage_metric_name == "signal"
    assert availability[-1] is True


@pytest.mark.parametrize(
    "coverage_binding",
    [
        {
            COVERAGE_FRAME_GENERATION_ID_ATTR: "other-generation",
            COVERAGE_FRAME_SET_ID_ATTR: "active-frame-set",
        },
        {COVERAGE_FRAME_GENERATION_ID_ATTR: "active-generation"},
    ],
    ids=["mismatched-generation", "missing-frame-set"],
)
def test_load_coverage_map_ignores_coverage_not_bound_to_active_frames(
    tmp_path,
    monkeypatch,
    coverage_binding,
):
    scenario_root = tmp_path / "scenario"
    coverage_file = scenario_root / "coverage" / "coverage_maps.h5"
    coverage_file.parent.mkdir(parents=True)
    with h5py.File(coverage_file, "w") as coverage_h5:
        for name, value in coverage_binding.items():
            coverage_h5.attrs[name] = value

    provider = Mock(spec=DataProvider)
    provider.info = ProviderInfo(
        name="identified-test-provider",
        source="test",
        generation_id="active-generation",
        frame_set_id="active-frame-set",
    )
    availability = []
    viz = SimpleNamespace(
        app_state=SimpleNamespace(),
        frame_loader=SimpleNamespace(provider=provider),
        set_state=Mock(),
        ui_manager=SimpleNamespace(
            panels={},
            set_coverage_data_available=lambda available: availability.append(bool(available)),
        ),
    )
    service = CoverageService()
    load_coverage = Mock(side_effect=AssertionError("unbound coverage must not be loaded"))
    warning = Mock()
    monkeypatch.setattr(service, "_load_v2_coverage_hdf5", load_coverage)
    monkeypatch.setattr(
        "visualizer.src.services.coverage_service.logger.warning",
        warning,
    )

    loaded = service.load_coverage_map(scenario_root, viz)

    assert loaded is False
    load_coverage.assert_not_called()
    assert viz.coverage_data is None
    assert availability[-1] is False
    warning.assert_called_once()
    assert "does not match" in warning.call_args.args[0]


def test_load_coverage_map_accepts_matching_remote_frame_set_binding(tmp_path, monkeypatch):
    scenario_root = tmp_path / "scenario"
    coverage_file = scenario_root / "coverage" / "coverage_maps.h5"
    coverage_file.parent.mkdir(parents=True)
    with h5py.File(coverage_file, "w") as coverage_h5:
        coverage_h5.attrs[COVERAGE_FRAME_SET_ID_ATTR] = "active-frame-set"

    provider = Mock(spec=DataProvider)
    provider.info = ProviderInfo(
        name="remote-test-provider",
        source="test",
        generation_id=None,
        frame_set_id="active-frame-set",
    )
    payload = {
        "grid_origin": np.asarray([0.0, 0.0, 1.5]),
        "grid_spacing": np.asarray([1.0, 1.0, 1.0]),
        "grid_shape": np.asarray([1, 1, 1]),
        "heights": [1.5],
        "available_metrics": ["signal"],
        "metric_name": "signal",
        "metric_layers": {"signal": np.asarray([[[2.0]]], dtype=np.float32)},
        "value_min": 2.0,
        "value_max": 2.0,
    }
    viz = SimpleNamespace(
        app_state=SimpleNamespace(),
        frame_loader=SimpleNamespace(provider=provider),
        set_state=Mock(),
        ui_manager=SimpleNamespace(
            panels={},
            set_coverage_data_available=Mock(),
        ),
    )
    service = CoverageService()
    load_coverage = Mock(return_value=payload)
    monkeypatch.setattr(service, "_load_v2_coverage_hdf5", load_coverage)

    loaded = service.load_coverage_map(scenario_root, viz)

    assert loaded is True
    load_coverage.assert_called_once_with(coverage_file)
    assert viz.coverage_metric_name == "signal"


def test_reset_runtime_state_delegates_controller_reset_and_clears_coverage_state():
    service = CoverageService()
    service.put_isolines(
        "iso",
        np.zeros((2, 3)),
        np.zeros((1, 2), dtype=np.int32),
        np.zeros((1, 3)),
        (1.0,),
    )
    reset_controller_state = Mock()
    panel = SimpleNamespace(reset_view_state=Mock(), update_coverage_status=Mock())
    ui_manager = SimpleNamespace(
        panels={"coverage": panel},
        set_coverage_data_available=Mock(),
    )
    viz = SimpleNamespace(
        app_state=SimpleNamespace(show_coverage=True, coverage_height_index=2),
        set_state=Mock(),
        ui_controller=SimpleNamespace(
            reset_coverage_runtime_state=reset_controller_state,
        ),
        ui_manager=ui_manager,
        coverage_data={"metric_name": "signal"},
        coverage_heights=[1.5, 3.0],
        coverage_height_index=1,
        coverage_opacity=0.4,
        coverage_interpolation_method="cubic",
        coverage_metric_name="signal",
        coverage_threshold_enabled=True,
        coverage_threshold_value=5.0,
        coverage_threshold_mask_enabled=True,
        coverage_isolines_enabled=True,
        coverage_isoline_count=11,
    )

    service.reset_runtime_state(viz)

    reset_controller_state.assert_called_once_with()
    assert viz.coverage_data is None
    assert viz.coverage_heights == []
    assert viz.coverage_opacity == 1.0
    assert viz.coverage_interpolation_method == "none"
    assert viz.coverage_metric_name is None
    assert viz.coverage_threshold_enabled is False
    assert viz.coverage_threshold_mask_enabled is False
    assert viz.coverage_isolines_enabled is False
    viz.set_state.assert_called_once_with(show_coverage=False, coverage_height_index=0)
    panel.reset_view_state.assert_called_once_with()
    panel.update_coverage_status.assert_called_once_with(False)
    ui_manager.set_coverage_data_available.assert_called_once_with(False)
    assert service.stats()["isoline_cache_size"] == 0
