from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shared.frames.directory_ownership import capture_frame_directory
from shared.frames.provider_base import DataProvider, ProviderCapability, ProviderInfo
from shared.frames.types import StandardMPCFrame
from shared.scenarios.actors import TimelineSpec
from shared.statistics.themes import theme_manager
from tests.visualizer.fixtures.packed_v2 import write_identity_only_frame_set
from visualizer.src.io.scenario_config import Scenario
from visualizer.src.panels.statistics_panel import (
    _GRAPH_EXPLANATIONS,
    StatisticsPanel,
    _bucket_reflection_orders,
    _constant_pair_state_message,
    _unique_export_path,
)
from visualizer.src.services.statistics_cache_service import StatisticsCacheService


def _sample_stats() -> dict:
    return {
        "total_mpcs": 12,
        "total_frames": 3,
        "unique_tx_count": 1,
        "unique_rx_count": 1,
        "unique_tx_rx_pairs": 1,
        "avg_mpcs_per_frame": 4.0,
        "overall_delay_spread": 2.5,
        "mpc_count_variation_coeff": 0.1,
        "path_loss_stats": {"min": 10.0, "max": 30.0, "mean": 20.0, "median": 20.0},
        "delay_stats": {"min": 1.0, "max": 3.0, "mean": 2.0, "median": 2.0},
        "reflection_order_dist": {0: 5, 1: 7},
        "mpc_type_dist": {0: 5, 1: 7},
        "frame_indices": np.array([0, 1, 2], dtype=np.int64),
        "mpc_evolution": np.array([3, 4, 5], dtype=np.int32),
        "delay_spread_evolution": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "path_loss_values": np.array([10.0, 20.0, 30.0], dtype=np.float32),
        "delay_values": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "aod_az_values": np.array([-45.0, 0.0, 45.0], dtype=np.float32),
        "aod_el_values": np.array([-10.0, 0.0, 10.0], dtype=np.float32),
        "aoa_az_values": np.array([-90.0, 0.0, 90.0], dtype=np.float32),
        "aoa_el_values": np.array([-20.0, 0.0, 20.0], dtype=np.float32),
        "reflection_order_evolution_per_frame": {0: [1, 2, 2], 1: [2, 2, 3]},
        "mpc_type_evolution_per_frame": {0: [1, 2, 2], 1: [2, 2, 3]},
        "pair_visibility_counts": {
            "direct_path_present": 2,
            "indirect_only": 1,
            "no_path": 0,
        },
        "pair_visibility_evolution": {
            "direct_path_present": [1, 0, 1],
            "indirect_only": [0, 1, 0],
            "no_path": [0, 0, 0],
        },
        "pair_visibility_summary": {
            "direct_path_present": {"count": 2, "percent": 66.6666666667},
            "indirect_only": {"count": 1, "percent": 33.3333333333},
            "no_path": {"count": 0, "percent": 0.0},
        },
        "direct_path_pair_share_evolution": np.array([1.0, 0.0, 1.0]),
        "pair_aggregate_path_gain_db_values": np.array([-20.0, -30.0, -25.0]),
        "pair_rms_delay_spread_ns_values": np.array([1.0, 3.0, 2.0]),
        "pair_aggregate_path_gain_stats": {
            "count": 3,
            "min": -30.0,
            "p10": -29.0,
            "median": -25.0,
            "p90": -21.0,
            "max": -20.0,
        },
        "pair_rms_delay_spread_stats": {
            "count": 3,
            "min": 1.0,
            "p10": 1.2,
            "median": 2.0,
            "p90": 2.8,
            "max": 3.0,
        },
        "strongest_single_path_loss_evolution": np.array([10.0, 15.0, 12.0], dtype=np.float32),
        "tx_rx_pairs": [(0, 0)],
    }


def test_statistics_panel_combines_every_high_interaction_order() -> None:
    assert _bucket_reflection_orders({0: 2, 5: 3, 6: 4, 8: 5}) == {
        0: 2,
        1: 0,
        2: 0,
        3: 0,
        4: 0,
        5: 3,
        6: 9,
    }


def _write_frame_files(frames_dir: Path) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for name in ("mpc_frames_00000-00099.h5", "frames_manifest.json"):
        (frames_dir / name).write_text("test")


def _file_scenario(root: Path, directory: str | None = None) -> Scenario:
    data_spec: dict = {"mode": "files"}
    if directory is not None:
        data_spec["files"] = {"directory": directory}
    return Scenario(
        root=root,
        scene_spec={},
        data_mode="files",
        data_spec=data_spec,
        view_defaults={},
        timeline=TimelineSpec(steps=1, duration_s=0.0),
    )


def _provider(
    generation_id: str = "generation-a",
    frame_set_id: str = "frame-set-a",
) -> SimpleNamespace:
    return SimpleNamespace(
        info=ProviderInfo(
            name="test",
            source="snapshot",
            generation_id=generation_id,
            frame_set_id=frame_set_id,
        )
    )


def test_statistics_cache_uses_configured_frame_directory(tmp_path: Path) -> None:
    scenario = _file_scenario(tmp_path / "scenario", "selected/frames")
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    cache_path = service.save_cached_stats(_sample_stats())

    assert cache_path == service._cache_path()
    assert cache_path is not None
    assert cache_path.parent == scenario.frames_dir.parent
    assert cache_path.name.startswith(service.CACHE_FILENAME_PREFIX)
    assert not cache_path.is_relative_to(scenario.frames_dir)
    assert cache_path.is_file()
    assert service._frame_set_fingerprint() is not None
    assert not (scenario.frames_dir / "scenario_stats_cache.npz").exists()
    assert service.load_cached_stats() is not None


def test_statistics_cache_uses_default_directory_when_data_files_are_omitted(
    tmp_path: Path,
) -> None:
    scenario = _file_scenario(tmp_path / "scenario")
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    cache_path = service.save_cached_stats(_sample_stats())

    assert scenario.frames_dir == (scenario.root / "frames").resolve()
    assert cache_path == service._cache_path()
    assert cache_path is not None
    assert cache_path.parent == scenario.root
    assert not cache_path.is_relative_to(scenario.frames_dir)
    assert service._frame_set_fingerprint() is not None


def test_statistics_cache_absolute_path_is_deterministic_and_distinct(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    first_frames = (tmp_path / "external" / "first").resolve()
    second_frames = first_frames.with_name("second")
    first = _file_scenario(scenario_root, str(first_frames))
    second = _file_scenario(scenario_root, str(second_frames))
    service = StatisticsCacheService(SimpleNamespace(scenario=first))

    first_path = service._cache_path()
    repeated_path = service._cache_path()
    second_path = service._cache_path(second)

    assert first.frames_dir == first_frames
    assert first_path == repeated_path
    assert first_path is not None
    assert second_path is not None
    assert first_path.parent == second_path.parent == first_frames.parent
    assert first_path.name != second_path.name
    assert not first_path.is_relative_to(first_frames)
    assert not second_path.is_relative_to(second_frames)


def test_statistics_cache_save_does_not_create_frame_directory(tmp_path: Path) -> None:
    scenario = SimpleNamespace(
        root=tmp_path / "scenario",
        frames_dir=tmp_path / "scenario" / "frames",
    )
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    expected_path = service._cache_path()

    cache_path = service.save_cached_stats(
        _sample_stats(),
        provider=_provider(),
    )

    assert cache_path is not None
    assert cache_path == expected_path == service._cache_path()
    assert cache_path.is_file()
    assert cache_path.parent == scenario.root
    assert not scenario.frames_dir.exists()


def test_statistics_cache_save_does_not_change_managed_frame_snapshot(
    tmp_path: Path,
) -> None:
    scenario_root = tmp_path / "scenario"
    frames_dir = scenario_root / "frames"
    write_identity_only_frame_set(frames_dir)
    snapshot = capture_frame_directory(frames_dir)
    scenario = SimpleNamespace(root=scenario_root, frames_dir=frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    cache_path = service.save_cached_stats(
        _sample_stats(),
        provider=_provider(),
    )

    assert cache_path is not None
    assert cache_path.parent == scenario_root
    snapshot.revalidate()


def test_regenerated_provider_rejects_then_replaces_preserved_cache(
    tmp_path: Path,
) -> None:
    scenario = SimpleNamespace(
        root=tmp_path / "scenario",
        frames_dir=tmp_path / "scenario" / "frames",
    )
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    original_provider = _provider("generation-a", "frame-set-a")
    replacement_provider = _provider("generation-b", "frame-set-b")
    original_stats = {"total_mpcs": 11}
    replacement_stats = {"total_mpcs": 22}

    cache_path = service.save_cached_stats(
        original_stats,
        provider=original_provider,
    )
    assert cache_path is not None
    original_cache = cache_path.read_bytes()
    scenario.frames_dir.rename(scenario.root / "old-frames")
    _write_frame_files(scenario.frames_dir)

    assert cache_path.is_file()
    assert service.load_cached_stats(provider=replacement_provider) is None
    assert cache_path.read_bytes() == original_cache

    replacement_path = service.save_cached_stats(
        replacement_stats,
        provider=replacement_provider,
    )

    assert replacement_path == cache_path
    assert cache_path.read_bytes() != original_cache
    assert service.load_cached_stats(provider=original_provider) is None
    loaded = service.load_cached_stats(provider=replacement_provider)
    assert loaded is not None
    assert loaded["total_mpcs"] == 22


def test_concurrent_statistics_cache_saves_are_atomic_and_leave_no_temps(
    tmp_path: Path,
) -> None:
    scenario = SimpleNamespace(
        root=tmp_path / "scenario",
        frames_dir=tmp_path / "scenario" / "frames",
    )
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    provider = _provider()
    save_count = 8
    start = threading.Barrier(save_count)

    def save(value: int) -> Path | None:
        start.wait(timeout=5.0)
        return service.save_cached_stats(
            {"total_mpcs": value},
            provider=provider,
        )

    with ThreadPoolExecutor(max_workers=save_count) as executor:
        saved_paths = list(executor.map(save, range(save_count)))

    cache_path = service._cache_path()
    assert cache_path is not None
    assert saved_paths == [cache_path] * save_count
    with np.load(cache_path, allow_pickle=False) as payload:
        assert "metadata_json" in payload
    loaded = service.load_cached_stats(provider=provider)
    assert loaded is not None
    assert loaded["total_mpcs"] in range(save_count)
    assert not list(cache_path.parent.glob(f"{cache_path.stem}.*.tmp.npz"))


def test_failed_statistics_cache_save_removes_its_unique_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scenario = SimpleNamespace(
        root=tmp_path / "scenario",
        frames_dir=tmp_path / "scenario" / "frames",
    )
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    def fail_after_partial_write(path: Path, **_payload: object) -> None:
        Path(path).write_bytes(b"partial")
        raise OSError("injected statistics cache failure")

    monkeypatch.setattr(np, "savez", fail_after_partial_write)

    with pytest.raises(OSError, match="injected statistics cache failure"):
        service.save_cached_stats(
            _sample_stats(),
            provider=_provider(),
        )

    cache_path = service._cache_path()
    assert cache_path is not None
    assert not cache_path.exists()
    assert not list(cache_path.parent.glob(f"{cache_path.stem}.*.tmp.npz"))


def test_statistics_cache_round_trip(tmp_path: Path):
    scenario = SimpleNamespace(
        data_mode="files", root=tmp_path / "scenario", frames_dir=tmp_path / "scenario" / "frames"
    )
    _write_frame_files(scenario.frames_dir)

    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    stats = _sample_stats()

    cache_path = service.save_cached_stats(stats)
    assert cache_path is not None
    assert cache_path.exists()

    loaded = service.load_cached_stats()
    assert loaded is not None
    assert loaded["total_mpcs"] == stats["total_mpcs"]
    assert loaded["reflection_order_dist"] == stats["reflection_order_dist"]
    np.testing.assert_array_equal(loaded["frame_indices"], stats["frame_indices"])
    np.testing.assert_allclose(loaded["path_loss_values"], stats["path_loss_values"])
    np.testing.assert_allclose(loaded["aod_az_values"], stats["aod_az_values"])
    np.testing.assert_allclose(loaded["aod_el_values"], stats["aod_el_values"])
    np.testing.assert_allclose(loaded["aoa_az_values"], stats["aoa_az_values"])
    np.testing.assert_allclose(loaded["aoa_el_values"], stats["aoa_el_values"])
    np.testing.assert_allclose(
        loaded["direct_path_pair_share_evolution"],
        stats["direct_path_pair_share_evolution"],
    )
    np.testing.assert_allclose(
        loaded["pair_aggregate_path_gain_db_values"],
        stats["pair_aggregate_path_gain_db_values"],
    )
    np.testing.assert_allclose(
        loaded["pair_rms_delay_spread_ns_values"],
        stats["pair_rms_delay_spread_ns_values"],
    )
    np.testing.assert_allclose(
        loaded["strongest_single_path_loss_evolution"],
        stats["strongest_single_path_loss_evolution"],
    )

    with np.load(cache_path, allow_pickle=False) as payload:
        assert "aod_az_values" in payload
        assert payload["path_loss_values"].dtype == np.float32
        assert payload["aod_az_values"].dtype == np.float32
        assert payload["frame_indices"].dtype == np.int32
        assert payload["direct_path_pair_share_evolution"].dtype == np.float32
        assert payload["pair_aggregate_path_gain_db_values"].dtype == np.float32
        assert payload["pair_rms_delay_spread_ns_values"].dtype == np.float32
        metadata = json.loads(payload["metadata_json"].item())
        assert "aod_az_values" not in metadata["stats"]
        assert metadata["cache_key"]["cache_schema_version"] == service.CACHE_SCHEMA_VERSION
        assert metadata["cache_key"]["algorithm_schema_version"] >= 1


def test_statistics_cache_invalidates_on_frame_set_change(tmp_path: Path):
    scenario = SimpleNamespace(
        data_mode="files", root=tmp_path / "scenario", frames_dir=tmp_path / "scenario" / "frames"
    )
    _write_frame_files(scenario.frames_dir)

    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    service.save_cached_stats(_sample_stats())

    frame_file = scenario.frames_dir / "mpc_frames_00000-00099.h5"
    frame_file.write_text("changed")

    assert service.load_cached_stats() is None


def test_statistics_cache_fingerprint_includes_frame_metadata_files(tmp_path: Path):
    scenario = SimpleNamespace(
        data_mode="files", root=tmp_path / "scenario", frames_dir=tmp_path / "scenario" / "frames"
    )
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    initial = service._frame_set_fingerprint()
    (scenario.frames_dir / "frames_manifest.json").write_text("changed-manifest")

    assert service._frame_set_fingerprint() != initial


def test_statistics_cache_fingerprint_is_stat_based_for_frame_payloads(tmp_path: Path):
    scenario = SimpleNamespace(
        data_mode="files", root=tmp_path / "scenario", frames_dir=tmp_path / "scenario" / "frames"
    )
    scenario.frames_dir.mkdir(parents=True)
    frame_file = scenario.frames_dir / "mpc_frames_00000-00099.h5"
    frame_file.write_text("abcd")
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))

    initial = service._frame_set_fingerprint()
    original_stat = frame_file.stat()
    frame_file.write_text("wxyz")
    os.utime(
        frame_file,
        ns=(int(original_stat.st_atime_ns), int(original_stat.st_mtime_ns)),
    )

    assert service._frame_set_fingerprint() == initial


def test_statistics_panel_binds_provider_source():
    class _Provider(DataProvider):
        @property
        def info(self):
            return ProviderInfo(
                name="test",
                source="memory",
                generation_id="generation-a",
                frame_set_id="set-a",
            )

        def list_frames(self):
            return [0]

        def has_frame(self, step):
            return step == 0

        def load_frame(self, step) -> StandardMPCFrame:
            raise AssertionError(f"panel must not load full frame {step}")

    calls = []
    statistics_service = SimpleNamespace(
        start_collection=lambda provider, **kwargs: calls.append((provider, kwargs)) or 17,
        cancel_collection=lambda: None,
    )
    parent = SimpleNamespace(
        scenario=object(),
        scenario_statistics_service=statistics_service,
    )
    panel = StatisticsPanel(parent)
    provider = _Provider()

    assert panel.set_statistics_source(SimpleNamespace(provider=provider)) is True

    assert panel._statistics_provider is provider
    assert panel._provider_stats_generation == 17
    assert calls[0][0] is provider
    assert calls[0][1]["force"] is False


def test_statistics_panel_ignores_queued_result_from_stale_generation(monkeypatch):
    panel = StatisticsPanel(SimpleNamespace())
    panel._provider_stats_generation = 2
    applied = []
    monkeypatch.setattr(
        panel,
        "_apply_stats_result",
        lambda stats, *, status_text: applied.append((stats, status_text)),
    )

    panel._on_provider_stats_ready({"total_frames": 1}, False, 1)
    panel._on_provider_stats_ready({"total_frames": 2}, False, 2)

    assert applied == [({"total_frames": 2}, "Done")]


def test_statistics_panel_renders_scalar_coverage_figures_without_cached_mpc_stats(monkeypatch):
    class _FakeGroup:
        def __init__(self):
            self.visible = None

        def setVisible(self, visible):
            self.visible = bool(visible)

    class _FakeHost(_FakeGroup):
        pass

    class _FakeLabel:
        def __init__(self):
            self.text = ""

        def setText(self, text):
            self.text = text

    parent = SimpleNamespace(
        coverage_data={
            "metric_name": "path_loss_db",
            "values_3d": np.array([[[80.0, 100.0, np.nan]]], dtype=np.float32),
            "grid_spacing": np.array([1.0, 2.0], dtype=np.float32),
        },
        coverage_height_index=0,
        coverage_heights=[1.5],
    )
    panel = StatisticsPanel(parent)
    panel._graphs_panel_created = True
    panel._graphs_visible = True
    panel._coverage_graphs_group = _FakeGroup()
    panel._coverage_graphs_status = _FakeLabel()
    panel.widgets = {
        "coverage_distribution_chart": _FakeHost(),
        "coverage_success_chart": _FakeHost(),
    }
    rendered = []
    monkeypatch.setattr(panel, "_ensure_plot_dependencies", lambda: True)
    monkeypatch.setattr(
        panel,
        "_create_chart",
        lambda _widget, chart_type, data, title="": rendered.append((chart_type, data, title)),
    )

    panel._render_coverage_graphs()

    assert [item[0] for item in rendered] == [
        "coverage_histogram",
        "coverage_success_curve",
    ]
    assert "Path loss" in rendered[0][2]
    assert "1.50 m" in rendered[1][2]
    assert "No data 33.3%" in panel._coverage_graphs_status.text
    assert panel._coverage_graphs_group.visible is True


def test_statistics_panel_renders_serving_tx_share_without_categorical_curve(monkeypatch):
    class _FakeWidget:
        def __init__(self):
            self.visible = None

        def setVisible(self, visible):
            self.visible = bool(visible)

    class _FakeLabel:
        def setText(self, text):
            self.text = text

    parent = SimpleNamespace(
        coverage_data={
            "metric_name": "serving_tx",
            "tx_names": ["West", "East"],
            "serving_tx_count": 2,
            "values_3d": np.array([[[0.0, 1.0, -1.0, np.nan]]], dtype=np.float32),
        },
        coverage_height_index=0,
        coverage_heights=[2.0],
    )
    panel = StatisticsPanel(parent)
    panel._graphs_panel_created = True
    panel._graphs_visible = True
    panel._coverage_graphs_group = _FakeWidget()
    panel._coverage_graphs_status = _FakeLabel()
    distribution = _FakeWidget()
    success = _FakeWidget()
    panel.widgets = {
        "coverage_distribution_chart": distribution,
        "coverage_success_chart": success,
    }
    rendered = []
    monkeypatch.setattr(panel, "_ensure_plot_dependencies", lambda: True)
    monkeypatch.setattr(
        panel,
        "_create_chart",
        lambda _widget, chart_type, data, title="": rendered.append((chart_type, data, title)),
    )

    panel._render_coverage_graphs()

    assert len(rendered) == 1
    assert rendered[0][0] == "coverage_serving_share"
    assert rendered[0][1]["labels"] == ["West", "East", "No service"]
    assert rendered[0][1]["percentages"] == pytest.approx([25.0, 25.0, 50.0])
    assert distribution.visible is True
    assert success.visible is False


def test_statistics_panel_all_graph_hosts_have_central_explanations(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    graphs = panel.create_graphs_panel()
    qtbot.addWidget(graphs)
    expected_keys = {
        "delay_time_chart",
        "coverage_distribution_chart",
        "coverage_success_chart",
        "reflection_order_chart",
        "interaction_type_chart",
        "path_loss_histogram_chart",
        "delay_histogram_chart",
        "delay_cdf_chart",
        "path_loss_cdf_chart",
        "pair_gain_cdf_chart",
        "pair_delay_spread_cdf_chart",
        "aod_az_polar_chart",
        "aod_el_polar_chart",
        "aoa_az_polar_chart",
        "aoa_el_polar_chart",
        "mpc_evolution_chart",
        "delay_spread_trend_chart",
        "mpc_order_evolution_hist_chart",
        "mpc_type_evolution_chart",
        "pair_visibility_chart",
        "strongest_path_loss_chart",
    }

    assert set(_GRAPH_EXPLANATIONS) == expected_keys
    for key in expected_keys:
        assert panel.widgets[key].toolTip() == _GRAPH_EXPLANATIONS[key]


def _visible_chart(panel, qtbot, key, kind, data, title="Chart"):
    host = panel._create_chart_widget(key)
    qtbot.addWidget(host)
    host.resize(560, 320)
    host.show()
    panel._create_chart(host, kind, data, title)
    return host


@pytest.mark.parametrize(
    "kind,data",
    [
        (
            "coverage_histogram",
            {
                "values": [1.0, 2.0, 3.0],
                "xlabel": "SINR (linear)",
                "color_scale": "logarithmic",
                "percentiles": (1.2, 2.0, 2.8),
            },
        ),
        (
            "coverage_success_curve",
            {
                "thresholds": [1.0, 2.0, 3.0],
                "percentages": [75.0, 50.0, 25.0],
                "xlabel": "SINR (linear)",
                "comparator": ">=",
                "color_scale": "logarithmic",
                "total_cells": 4,
                "valid_cells": 3,
            },
        ),
        (
            "coverage_serving_share",
            {
                "labels": ["West", "East", "No service"],
                "percentages": [25.0, 50.0, 25.0],
                "cells": [1, 2, 1],
                "areas_m2": [2.0, 4.0, 2.0],
                "colors": ["#1f77b4", "#d62728", "#7f7f7f"],
            },
        ),
    ],
)
def test_statistics_panel_coverage_chart_renderers_create_pyqtgraph_plots(qtbot, kind, data):
    import pyqtgraph as pg

    panel = StatisticsPanel(SimpleNamespace())
    key = (
        "coverage_success_chart"
        if kind == "coverage_success_curve"
        else "coverage_distribution_chart"
    )
    host = _visible_chart(panel, qtbot, key, kind, data)
    assert isinstance(host.plot, pg.PlotWidget)
    assert host.chart_metadata["chart_type"] == kind
    assert host.plot.toolTip() == host.toolTip()
    # The same input also has a purely headless publication path.
    pytest.importorskip("matplotlib")
    figure = panel._create_publication_figure(kind, data)
    assert figure.canvas.__class__.__name__ == "FigureCanvasAgg"


def test_statistics_panel_interaction_charts_use_semantic_labels_and_active_series(qtbot):
    import pyqtgraph as pg

    panel = StatisticsPanel(SimpleNamespace())
    host = _visible_chart(
        panel,
        qtbot,
        "interaction_type_chart",
        "interaction_type",
        {0: 10, 99: 2, -1: 1},
        "MPCs by Initial Propagation Mechanism",
    )
    assert host.title.text() == "MPCs by Initial Propagation Mechanism"
    assert [label for _, label in host.plot.getAxis("left")._tickLevels[0]] == [
        "LoS",
        "Virtual",
        "Unknown",
    ]
    bars = next(item for item in host.plot.items() if isinstance(item, pg.BarGraphItem))
    assert [brush.color().name() for brush in bars.opts["brushes"]] == [
        "#ffd600",
        "#ff9933",
        "#808080",
    ]
    data = host.chart_metadata["data"]
    x, y = panel._find_nearest_chart_point(1.0, 1.0, "interaction_type", data)
    assert (x, y) == (2.0, 1.0)
    assert (
        panel._format_chart_tooltip(x, y, "interaction_type", data)
        == "Initial mechanism: Virtual\nCount: 2"
    )
    evolution = _visible_chart(
        panel,
        qtbot,
        "mpc_type_evolution_chart",
        "mpc_type_evolution",
        {"frame_indices": [0, 1], "type_data": {0: [3, 2], 8: [0, 0], 99: [0, 1], -1: [1, 0]}},
    )
    assert [series[0] for series in evolution._series] == ["LoS", "Virtual", "Unknown"]
    data = evolution.chart_metadata["data"]
    x, y = panel._find_nearest_chart_point(1.0, 1.0, "mpc_type_evolution", data)
    assert (x, y) == (1.0, 0.0)
    assert panel._format_chart_tooltip(x, y, "mpc_type_evolution", data).splitlines() == [
        "Frame: 1",
        "LoS: 2",
        "Virtual: 1",
        "Unknown: 0",
    ]


def test_statistics_panel_interaction_order_evolution_uses_active_symlog_lines(qtbot):
    from visualizer.src.panels.statistics_plot import transform

    SymmetricalLogTransform = pytest.importorskip("matplotlib.scale").SymmetricalLogTransform

    panel = StatisticsPanel(SimpleNamespace())
    data = {
        "frame_indices": [0, 1, 2],
        "order_data": {0: [1, 0, 1], 1: [100, 120, 110], 2: [0, 0, 0]},
    }
    host = _visible_chart(
        panel, qtbot, "mpc_order_evolution_hist_chart", "mpc_order_evolution", data
    )
    assert [series[0] for series in host._series] == ["Order 0", "Order 1"]
    assert host.plot.getAxis("left").display_scale == "symlog"
    values = np.array([-100, -1, -0.5, 0, 0.5, 1, 100, 120.0])
    np.testing.assert_allclose(
        transform(values, "symlog"), SymmetricalLogTransform(10, 1.0, 1.0).transform(values)
    )
    data = host.chart_metadata["data"]
    x, y = panel._find_nearest_chart_point(1.0, 100.0, "mpc_order_evolution", data)
    assert (x, y) == (1.0, 0.0)
    assert panel._format_chart_tooltip(x, y, "mpc_order_evolution", data).splitlines() == [
        "Frame: 1",
        "Order 0: 0",
        "Order 1: 120",
    ]


def test_statistics_panel_polar_rose_hover_reports_azimuth_bin(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    host = _visible_chart(
        panel,
        qtbot,
        "aod_az_polar_chart",
        "polar_rose",
        {"values": [-170.0, -10.0, 10.0, 170.0], "n_bins": 4},
    )
    data = host.chart_metadata["data"]
    x, y = panel._find_nearest_chart_point(np.deg2rad(10.0), 1.0, "polar_rose", data)
    assert x == pytest.approx(np.deg2rad(45.0))
    assert y == 1.0
    assert (
        panel._format_chart_tooltip(x, y, "polar_rose", data)
        == "Azimuth bin: 0 deg to 90 deg\nCount: 1"
    )
    assert host.plot.toolTip() == _GRAPH_EXPLANATIONS["aod_az_polar_chart"]
    pytest.importorskip("matplotlib")
    figure = panel._create_publication_figure("polar_rose", data)
    assert figure.axes[0].get_theta_offset() == 0
    assert figure.axes[0].get_theta_direction() == 1


def test_statistics_panel_hover_restores_graph_explanation(qtbot):
    from PySide6.QtCore import QPointF

    panel = StatisticsPanel(SimpleNamespace())
    host = _visible_chart(panel, qtbot, "reflection_order_chart", "reflection_order", {0: 4})
    host.plot.setToolTip("Interaction order: 0\nCount: 4")
    host._hover((QPointF(-1000, -1000),))
    assert host.plot.toolTip() == _GRAPH_EXPLANATIONS["reflection_order_chart"]


def test_statistics_panel_constant_pair_states_replace_chart_without_stale_widgets(qtbot):
    panel = StatisticsPanel(SimpleNamespace())
    constant = {
        "frame_indices": list(range(50)),
        "category_data": {
            "direct_path_present": [3] * 50,
            "indirect_only": [0] * 50,
            "no_path": [0] * 50,
        },
    }
    host = _visible_chart(
        panel, qtbot, "pair_visibility_chart", "pair_visibility_evolution", constant
    )
    assert host.plot is None
    assert "all have a direct path" in host.message.text()
    assert "50 frames" in host.message.text()
    assert host.message.toolTip() == _GRAPH_EXPLANATIONS["pair_visibility_chart"]
    changing = {
        "frame_indices": [0, 1, 2],
        "category_data": {
            "direct_path_present": [3, 2, 3],
            "indirect_only": [0, 1, 0],
            "no_path": [0, 0, 0],
        },
    }
    panel._create_chart(host, "pair_visibility_evolution", changing)
    plot = host.plot
    assert plot is not None and plot.isVisible()
    assert host.message.isHidden()
    panel._create_chart(host, "pair_visibility_evolution", constant)
    assert host.plot is plot and plot.isHidden()
    assert host.message.isVisible()
    panel._create_chart(host, "pair_visibility_evolution", changing)
    assert host.plot is plot and plot.isVisible()


def test_constant_pair_state_message_does_not_claim_pair_identity() -> None:
    message = _constant_pair_state_message(
        [0, 1],
        {
            "direct_path_present": [2, 2],
            "indirect_only": [1, 1],
            "no_path": [0, 0],
        },
    )

    assert message is not None
    assert "same path-state counts" in message
    assert "same pairs" not in message


def test_statistics_panel_renders_evolution_charts_from_cached_arrays(monkeypatch):
    class _FakeLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    class _FakeGroup:
        def __init__(self):
            self.visible = None

        def setVisible(self, visible):
            self.visible = visible

        def setToolTip(self, text):
            self.tooltip = text

    panel = StatisticsPanel(SimpleNamespace())
    panel.stats = _sample_stats()
    panel._evolution_group = _FakeGroup()
    panel._channel_evolution_group = _FakeGroup()
    panel._polar_group = _FakeGroup()
    panel._graphs_panel_created = True
    panel._graphs_visible = True
    panel._graphs_status_label = _FakeLabel()
    panel.widgets = {
        "total_mpcs": _FakeLabel(),
        "total_frames": _FakeLabel(),
        "unique_pairs": _FakeLabel(),
        "mpc_frame_ratio": _FakeLabel(),
        "overall_delay_spread": _FakeLabel(),
        "path_loss_range": _FakeLabel(),
        "mpc_count_cv": _FakeLabel(),
        "pair_gain_percentiles": _FakeLabel(),
        "pair_delay_spread_percentiles": _FakeLabel(),
        "pair_visibility_summary": _FakeLabel(),
        "reflection_order_chart": object(),
        "interaction_type_chart": _FakeGroup(),
        "path_loss_histogram_chart": object(),
        "delay_histogram_chart": object(),
        "delay_cdf_chart": object(),
        "path_loss_cdf_chart": object(),
        "pair_gain_cdf_chart": object(),
        "pair_delay_spread_cdf_chart": object(),
        "aod_az_polar_chart": object(),
        "aod_el_polar_chart": object(),
        "aoa_az_polar_chart": object(),
        "aoa_el_polar_chart": object(),
        "mpc_evolution_chart": object(),
        "delay_spread_trend_chart": object(),
        "mpc_order_evolution_hist_chart": object(),
        "mpc_type_evolution_chart": _FakeGroup(),
        "pair_visibility_chart": object(),
        "strongest_path_loss_chart": object(),
    }

    rendered_chart_types = []

    def _record_chart(widget, chart_type, data, title=""):
        rendered_chart_types.append(chart_type)

    monkeypatch.setattr(panel, "_create_chart", _record_chart)

    panel._update_statistics()

    assert "mpc_evolution" not in rendered_chart_types
    assert rendered_chart_types[8:12] == [
        "polar_rose",
        "angle_histogram",
        "polar_rose",
        "angle_histogram",
    ]
    assert "line_evolution" in rendered_chart_types
    assert "mpc_order_evolution" in rendered_chart_types
    assert "mpc_type_evolution" in rendered_chart_types
    assert "pair_visibility_evolution" in rendered_chart_types
    assert panel.widgets["pair_gain_percentiles"].text.endswith("(n=3)")
    assert panel.widgets["pair_delay_spread_percentiles"].text.endswith("(n=3)")
    assert "Direct 2 (66.7%)" in panel.widgets["pair_visibility_summary"].text


def test_statistics_panel_defers_graph_rendering_when_collapsed(monkeypatch):
    class _FakeLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    panel = StatisticsPanel(SimpleNamespace())
    panel.stats = _sample_stats()
    panel._graphs_panel_created = True
    panel._graphs_visible = False
    panel._graphs_status_label = _FakeLabel()
    panel.widgets = {
        "total_mpcs": _FakeLabel(),
        "total_frames": _FakeLabel(),
        "unique_pairs": _FakeLabel(),
        "mpc_frame_ratio": _FakeLabel(),
        "overall_delay_spread": _FakeLabel(),
        "path_loss_range": _FakeLabel(),
        "mpc_count_cv": _FakeLabel(),
        "pair_gain_percentiles": _FakeLabel(),
        "pair_delay_spread_percentiles": _FakeLabel(),
        "pair_visibility_summary": _FakeLabel(),
    }

    monkeypatch.setattr(
        panel,
        "_render_statistics_graphs",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("collapsed graphs should not render")
        ),
    )

    panel._update_statistics()

    assert panel._graphs_dirty is True
    assert panel._graphs_status_label.text == "Open this section to render graphs."


def test_statistics_panel_cdf_tooltip_reports_percentile():
    panel = StatisticsPanel(SimpleNamespace())
    data = {
        "_cdf_x": np.array([10.0, 20.0, 30.0, 40.0]),
        "_cdf_y": np.array([0.25, 0.5, 0.75, 1.0]),
        "_cdf_total": 4,
        "xlabel": "Path Loss (dB)",
    }

    x_val, y_val = panel._find_nearest_chart_point(20.1, 0.5, "cdf", data)
    assert x_val == 20.0
    assert y_val == 0.5

    tooltip = panel._format_chart_tooltip(x_val, y_val, "cdf", data)
    assert "Path Loss (dB): 20.00" in tooltip
    assert "CDF: 50.0%" in tooltip
    assert "Samples <= value: 2/4" in tooltip


def test_statistics_panel_histogram_tooltip_reports_bin_range():
    panel = StatisticsPanel(SimpleNamespace())
    data = {
        "values": np.array([0.0, 1.0, 2.0, 3.0]),
        "_hist_counts": np.array([2, 2]),
        "_hist_edges": np.array([0.0, 2.0, 4.0]),
    }

    x_val, y_val = panel._find_nearest_chart_point(1.1, 2.0, "histogram", data)
    assert x_val == 1.0
    assert y_val == 2

    tooltip = panel._format_chart_tooltip(x_val, y_val, "histogram", data)
    assert "Bin: 0.00 - 2.00" in tooltip
    assert "Center: 1.00" in tooltip
    assert "Count: 2" in tooltip


def test_statistics_panel_graph_status_explains_hidden_sections():
    text, transient = StatisticsPanel._graphs_render_status(
        has_angle_data=False,
        multi_frame=False,
    )

    assert transient is False
    assert "angular charts: no angular data" in text
    assert "evolution charts: single frame" in text


def test_statistics_panel_chart_export_uses_unique_paths(monkeypatch, tmp_path: Path):
    pytest.importorskip("matplotlib")

    class _FakeFigure:
        def __init__(self):
            self.saved_paths = []

        def savefig(self, path, **_kwargs):
            out_path = Path(path)
            out_path.write_text("new chart", encoding="utf-8")
            self.saved_paths.append(out_path)

        def clear(self):
            pass

    figure = _FakeFigure()
    existing = tmp_path / "reflection_order_chart.png"
    existing.write_text("existing chart", encoding="utf-8")
    panel = StatisticsPanel(SimpleNamespace())
    panel.stats = {"total_mpcs": 1}
    panel._chart_specs = {
        "reflection_order_chart": {
            "chart_type": "reflection_order",
            "data": {0: 1},
            "title": "Order counts",
        }
    }
    panel._graphs_dirty = panel._coverage_graphs_dirty = False
    panel._graphs_rendered = panel._graphs_panel_created = True
    monkeypatch.setattr(panel, "_create_publication_figure", lambda **_kwargs: figure)
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QFileDialog.getExistingDirectory",
        lambda *_args, **_kwargs: str(tmp_path),
    )
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QMessageBox.warning", lambda *_args, **_kwargs: None
    )
    panel._on_export_charts()
    assert existing.read_text(encoding="utf-8") == "existing chart"
    assert (tmp_path / "reflection_order_chart_1.png").read_text(encoding="utf-8") == "new chart"
    assert figure.saved_paths == [tmp_path / "reflection_order_chart_1.png"]


def test_statistics_csv_uses_semantic_virtual_and_unknown_labels(
    monkeypatch,
    qapp,
    tmp_path: Path,
) -> None:
    output = tmp_path / "statistics.csv"
    panel = StatisticsPanel(SimpleNamespace(scenario_path=tmp_path / "scenario.yaml"))
    panel.stats = {
        "total_mpcs": 3,
        "total_frames": 1,
        "unique_tx_rx_pairs": 1,
        "mpc_type_dist": {99: 2, -1: 1},
    }
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QFileDialog.getSaveFileName",
        lambda *_args, **_kwargs: (str(output), "CSV Files (*.csv)"),
    )
    monkeypatch.setattr(
        "visualizer.src.panels.statistics_panel.QMessageBox.information",
        lambda *_args, **_kwargs: None,
    )

    panel._on_export_csv()

    exported = output.read_text(encoding="utf-8")
    assert "Initial Propagation Mechanism Virtual,2" in exported
    assert "Initial Propagation Mechanism Unknown,1" in exported
    assert "Type 99" not in exported
    assert "Type -1" not in exported


def test_unique_export_path_counts_up(tmp_path: Path):
    base = tmp_path / "chart.png"
    base.write_text("0", encoding="utf-8")
    (tmp_path / "chart_1.png").write_text("1", encoding="utf-8")

    assert _unique_export_path(base) == tmp_path / "chart_2.png"


def test_statistics_panel_clears_transient_graph_status():
    class _FakeLabel:
        def __init__(self):
            self.text = None

        def setText(self, text):
            self.text = text

    panel = StatisticsPanel(SimpleNamespace())
    panel._graphs_status_label = _FakeLabel()

    panel._set_graphs_status("Graphs updated.", transient=False)
    panel._clear_transient_graphs_status()

    assert panel._graphs_status_label.text == ""


def test_statistics_panel_does_not_refresh_for_mpc_palette_changes():
    panel = StatisticsPanel(SimpleNamespace())
    panel.stats = _sample_stats()
    panel._graphs_dirty = False

    original = theme_manager.current.categorical_colormap
    replacement = "tab10" if original != "tab10" else "Set1"
    try:
        theme_manager.set_colormap("categorical", replacement)
        assert panel._graphs_dirty is False
    finally:
        theme_manager.set_colormap("categorical", original)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("with_identity", [False, True])
@pytest.mark.parametrize("capability", [ProviderCapability.STREAMING, ProviderCapability.OVERRIDES])
def test_mutable_source_never_borrows_or_overwrites_local_cache(
    tmp_path, wrapped, with_identity, capability
):
    scenario = _file_scenario(tmp_path)
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    path = service.save_cached_stats({"total_mpcs": 6})
    original = path.read_bytes()
    provider = _provider()
    provider.info.capabilities = capability | ProviderCapability.CACHING
    if not with_identity:
        provider.info.generation_id = provider.info.frame_set_id = None
    if wrapped:
        provider = SimpleNamespace(provider=provider)

    assert service.load_cached_stats(provider=provider) is None
    assert service.save_cached_stats({"total_mpcs": 4}, provider=provider) is None
    assert path.read_bytes() == original


def test_remote_snapshot_without_generation_reuses_and_invalidates(tmp_path):
    scenario = _file_scenario(tmp_path)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    remote = _provider(None, "remote-snapshot-a")
    path = service.save_cached_stats({"total_mpcs": 468}, provider=remote)
    assert path is not None
    assert not scenario.frames_dir.exists()
    assert service.load_cached_stats(provider=remote)["total_mpcs"] == 468
    assert service.load_cached_stats(provider=_provider(None, "remote-snapshot-b")) is None
    assert (
        service.load_cached_stats(provider=_provider("generation-a", "remote-snapshot-a")) is None
    )
    # A configured local destination is not part of this remote snapshot identity.
    _write_frame_files(scenario.frames_dir)
    assert service.load_cached_stats(provider=remote)["total_mpcs"] == 468


@pytest.mark.parametrize("mode", [None, "live_grpc", "remote_hdf5"])
def test_providerless_cache_requires_explicit_local_context(tmp_path, mode):
    scenario = SimpleNamespace(frames_dir=tmp_path / "frames", data_mode=mode)
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    assert service.save_cached_stats({"total_mpcs": 6}) is None
    assert service.load_cached_stats() is None


def test_unknown_source_without_identity_cannot_borrow_local_files(tmp_path):
    scenario = _file_scenario(tmp_path)
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    provider = _provider(None, None)
    assert service.save_cached_stats({"total_mpcs": 6}, provider=provider) is None
    assert service._cache_key(provider=SimpleNamespace()) is None


def test_legacy_local_identity_tracks_actual_provider_files(tmp_path):
    scenario = _file_scenario(tmp_path / "configured")
    _write_frame_files(scenario.frames_dir)
    actual = tmp_path / "active" / "frames"
    _write_frame_files(actual)
    provider = _provider(None, None)
    provider.recording_paths = lambda: tuple(actual.iterdir())
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    path = service.save_cached_stats({"total_mpcs": 8}, provider=provider)
    assert path is not None
    (scenario.frames_dir / "frames_manifest.json").write_text("unrelated change")
    assert service.load_cached_stats(provider=provider)["total_mpcs"] == 8
    (actual / "frames_manifest.json").write_text("active recording changed")
    assert service.load_cached_stats(provider=provider) is None
    provider.recording_paths = lambda: tuple(scenario.frames_dir.iterdir())
    assert service.load_cached_stats(provider=provider) is None


def test_unsafe_previous_statistics_schema_is_rejected(tmp_path):
    scenario = _file_scenario(tmp_path)
    _write_frame_files(scenario.frames_dir)
    service = StatisticsCacheService(SimpleNamespace(scenario=scenario))
    path = service._cache_path()
    np.savez(
        path,
        metadata_json=np.array(
            json.dumps(
                {
                    "schema_version": 4,
                    "cache_key": service._cache_key(),
                    "stats": {"total_mpcs": 6},
                }
            )
        ),
    )
    assert service.load_cached_stats() is None
