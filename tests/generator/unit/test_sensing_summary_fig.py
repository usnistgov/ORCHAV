from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shared.sensing.plotting import viz as sensing_summary_fig
from shared.sensing.signal_processing import MicroDopplerResult, doppler_axis_to_velocity_m_s
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload


class _FakeAxes:
    def __init__(self):
        self.plots = []
        self.scatters = []
        self.imshow_kwargs = None
        self.pcolormesh_args = None
        self.pcolormesh_kwargs = None
        self.ylabel = None
        self.transAxes = object()

    def plot(self, x, y, **kwargs):
        self.plots.append((list(x), list(y), kwargs))

    def scatter(self, x, y, **kwargs):
        self.scatters.append((list(x), list(y), kwargs))

    def imshow(self, *_args, **kwargs):
        self.imshow_kwargs = kwargs
        return object()

    def pcolormesh(self, *args, **kwargs):
        self.pcolormesh_args = args
        self.pcolormesh_kwargs = kwargs
        return object()

    def set_xlabel(self, *_args, **_kwargs):
        pass

    def set_ylabel(self, value, *_args, **_kwargs):
        self.ylabel = value

    def set_title(self, *_args, **_kwargs):
        pass

    def text(self, *_args, **_kwargs):
        pass

    def grid(self, *_args, **_kwargs):
        pass

    def legend(self, *_args, **_kwargs):
        pass

    def set_xlim(self, *_args, **_kwargs):
        pass

    def set_ylim(self, *_args, **_kwargs):
        pass


class _FakeFigure:
    def tight_layout(self):
        pass

    def savefig(self, *_args, **_kwargs):
        pass

    def colorbar(self, *_args, **_kwargs):
        pass


def _typed_rd_payload(
    *,
    range_axis_m: np.ndarray,
    doppler_axis_hz: np.ndarray,
    power_linear: np.ndarray,
) -> dict:
    carrier_freq_hz = 28e9
    sensing = build_visualizer_sensing_payload(
        {
            "config_resolved": {
                "range_mode": "bistatic",
                "carrier_freq_hz": carrier_freq_hz,
            },
            "range_axis_m": range_axis_m,
            "doppler_axis_hz": doppler_axis_hz,
            "range_doppler_power_by_pair": power_linear[np.newaxis, ...],
            "detections": [],
        }
    )
    pair_id = sensing["temporal"]["acquisition"]["pair_id"]
    sensing["truth"] = {
        "targets": [
            {
                "target_id": "target-beta",
                "lifecycle": "active",
                "valid": True,
                "pair_observables": [
                    {
                        "pair_id": pair_id,
                        "bistatic_path_length_m": 4.0,
                        "monostatic_equivalent_range_m": 2.0,
                        "monostatic_equivalent_velocity_m_s": -0.5,
                        "velocity_valid": True,
                        "velocity_derivation_source": "analytic_velocity_projection",
                        "velocity_reference_dt_s": None,
                        "undefined_reason": None,
                    }
                ],
            },
            {
                "target_id": "target-alpha",
                "lifecycle": "active",
                "valid": True,
                "pair_observables": [
                    {
                        "pair_id": pair_id,
                        "bistatic_path_length_m": 2.0,
                        "monostatic_equivalent_range_m": 1.0,
                        "monostatic_equivalent_velocity_m_s": 0.25,
                        "velocity_valid": True,
                        "velocity_derivation_source": "analytic_velocity_projection",
                        "velocity_reference_dt_s": None,
                        "undefined_reason": None,
                    }
                ],
            },
        ]
    }
    return sensing


def _typed_angular_payload() -> dict:
    """Return one exact P,E,A,D,R product for shared plotting tests."""

    return build_visualizer_sensing_payload(
        {
            "config_resolved": {
                "range_mode": "monostatic",
                "carrier_freq_hz": 28e9,
            },
            "range_axis_m": np.linspace(0.0, 7.0, 8, dtype=np.float64),
            "doppler_axis_hz": np.linspace(-500.0, 250.0, 4, dtype=np.float64),
            "elevation_direction_cosine": np.array([-0.5, 0.5], dtype=np.float64),
            "azimuth_direction_cosine": np.linspace(-0.75, 0.75, 5, dtype=np.float64),
            "elevation_broadside_angle_deg": np.array([-32.0, 31.0], dtype=np.float64),
            "azimuth_broadside_angle_deg": np.array(
                [-48.0, -20.0, 1.0, 25.0, 52.0], dtype=np.float64
            ),
            "range_doppler_angle_by_pair": np.ones((1, 2, 5, 4, 8), dtype=np.float32),
            "detections": [
                {
                    "range_m": 2.0,
                    "velocity_m_s": 1.0,
                    "aoa_az_deg": 10.0,
                }
            ],
        }
    )


def test_create_gt_evolution_figure_uses_explicit_step_axis(monkeypatch, tmp_path: Path):
    ax_range = _FakeAxes()
    ax_speed = _FakeAxes()
    fig = _FakeFigure()

    monkeypatch.setattr(
        sensing_summary_fig.plt,
        "subplots",
        lambda *args, **kwargs: (fig, (ax_range, ax_speed)),
    )
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    gt_history = [
        {
            "step": 127,
            "target_ids": ["walker"],
            "target_ranges": [12.0],
            "target_speeds": [1.2],
            "target_velocity_valid": [True],
            "det_ranges": [],
            "det_velocities": [],
        },
        {
            "step": 255,
            "target_ids": ["walker"],
            "target_ranges": [11.8],
            "target_speeds": [1.2],
            "target_velocity_valid": [True],
            "det_ranges": [11.9],
            "det_velocities": [1.1],
        },
    ]

    sensing_summary_fig.create_gt_evolution_figure(
        gt_history,
        output_path=tmp_path / "gt_evolution.png",
    )

    assert ax_range.plots[0][0] == [127, 255]
    assert ax_speed.plots[0][0] == [127, 255]
    assert ax_range.scatters[0][0] == [255]
    assert ax_speed.scatters[0][0] == [255]


def test_create_gt_evolution_figure_joins_reordered_targets_by_id(
    monkeypatch,
    tmp_path: Path,
):
    ax_range = _FakeAxes()
    ax_speed = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(
        sensing_summary_fig.plt,
        "subplots",
        lambda *args, **kwargs: (fig, (ax_range, ax_speed)),
    )
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    sensing_summary_fig.create_gt_evolution_figure(
        [
            {
                "step": 10,
                "target_ids": ["alpha", "beta"],
                "target_ranges": [1.0, 20.0],
                "target_speeds": [0.1, 2.0],
                "target_velocity_valid": [True, True],
                "det_ranges": [],
                "det_velocities": [],
            },
            {
                "step": 11,
                "target_ids": ["beta", "alpha"],
                "target_ranges": [21.0, 1.5],
                "target_speeds": [2.1, 0.15],
                "target_velocity_valid": [True, True],
                "det_ranges": [],
                "det_velocities": [],
            },
        ],
        output_path=tmp_path / "gt_reordered.png",
    )

    assert ax_range.plots[0][0:2] == ([10, 11], [1.0, 1.5])
    assert ax_range.plots[0][2]["label"] == "alpha"
    assert ax_range.plots[1][0:2] == ([10, 11], [20.0, 21.0])
    assert ax_range.plots[1][2]["label"] == "beta"


def test_tracking_truth_series_joins_ids_and_breaks_lifecycle_gaps() -> None:
    target_order, series = sensing_summary_fig._identity_keyed_tracking_truth_series(
        [
            {
                "step": 0,
                "target_ids": ["alpha", "bravo"],
                "target_ranges": [10.0, 20.0],
                "target_speeds": [1.0, 2.0],
                "target_velocity_valid": [True, True],
            },
            {
                "step": 1,
                "target_ids": ["bravo", "alpha"],
                "target_ranges": [21.0, 11.0],
                "target_speeds": [2.1, 1.1],
                "target_velocity_valid": [True, True],
            },
            {
                "step": 2,
                "target_ids": ["alpha"],
                "target_ranges": [12.0],
                "target_speeds": [None],
                "target_velocity_valid": [False],
            },
            {
                "step": 3,
                "target_ids": ["bravo", "alpha"],
                "target_ranges": [23.0, 13.0],
                "target_speeds": [2.3, 1.3],
                "target_velocity_valid": [True, True],
            },
        ]
    )

    assert target_order == ["alpha", "bravo"]
    assert series["alpha"]["frames"] == [0.0, 1.0, 2.0, 3.0]
    assert series["alpha"]["ranges"] == [10.0, 11.0, 12.0, 13.0]
    assert series["alpha"]["speeds"][:2] == [1.0, 1.1]
    assert np.isnan(series["alpha"]["speeds"][2])
    assert series["alpha"]["speeds"][3] == 1.3
    assert series["bravo"]["ranges"][:2] == [20.0, 21.0]
    assert np.isnan(series["bravo"]["ranges"][2])
    assert series["bravo"]["ranges"][3] == 23.0


def test_create_gt_evolution_figure_does_not_bridge_absent_target_lifecycle(
    monkeypatch,
    tmp_path: Path,
) -> None:
    ax_range = _FakeAxes()
    ax_speed = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(
        sensing_summary_fig.plt,
        "subplots",
        lambda *args, **kwargs: (fig, (ax_range, ax_speed)),
    )
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    sensing_summary_fig.create_gt_evolution_figure(
        [
            {
                "step": 0,
                "target_ids": ["walker"],
                "target_ranges": [10.0],
                "target_speeds": [1.0],
                "target_velocity_valid": [True],
                "det_ranges": [],
                "det_velocities": [],
            },
            {
                "step": 1,
                "target_ids": [],
                "target_ranges": [],
                "target_speeds": [],
                "target_velocity_valid": [],
                "det_ranges": [],
                "det_velocities": [],
            },
            {
                "step": 2,
                "target_ids": ["walker"],
                "target_ranges": [12.0],
                "target_speeds": [1.2],
                "target_velocity_valid": [True],
                "det_ranges": [],
                "det_velocities": [],
            },
        ],
        output_path=tmp_path / "gt_lifecycle_gap.png",
    )

    assert ax_range.plots[0][0] == [0.0, 1.0, 2.0]
    assert np.isnan(ax_range.plots[0][1][1])
    assert np.isnan(ax_speed.plots[0][1][1])


def test_create_gt_evolution_figure_renders_unavailable_velocity_as_gap(
    monkeypatch,
    tmp_path: Path,
):
    ax_range = _FakeAxes()
    ax_speed = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(
        sensing_summary_fig.plt,
        "subplots",
        lambda *args, **kwargs: (fig, (ax_range, ax_speed)),
    )
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    sensing_summary_fig.create_gt_evolution_figure(
        [
            {
                "step": 0,
                "target_ids": ["alpha"],
                "target_ranges": [1.0],
                "target_speeds": [None],
                "target_velocity_valid": [False],
                "det_ranges": [],
                "det_velocities": [],
            },
            {
                "step": 1,
                "target_ids": ["alpha"],
                "target_ranges": [1.5],
                "target_speeds": [0.5],
                "target_velocity_valid": [True],
                "det_ranges": [],
                "det_velocities": [],
            },
        ],
        output_path=tmp_path / "gt_velocity_gap.png",
    )

    assert np.isnan(ax_speed.plots[0][1][0])
    assert ax_speed.plots[0][1][1] == pytest.approx(0.5)


def test_create_micro_doppler_figure_uses_configurable_dynamic_range(monkeypatch, tmp_path: Path):
    ax = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(sensing_summary_fig.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    result = MicroDopplerResult(
        spectrogram=np.array([[1.0, 10.0], [5.0, 20.0]], dtype=np.float64),
        time_axis_s=np.array([0.0, 1.0], dtype=np.float64),
        doppler_axis_hz=np.array([-100.0, 100.0], dtype=np.float64),
        range_bin=5,
        range_m=1.5,
    )
    sensing_summary_fig.create_micro_doppler_figure(
        result,
        output_path=tmp_path / "md.png",
        dynamic_range_db=75.0,
    )

    power_db = 10.0 * np.log10(np.maximum(result.spectrogram, 1e-30))
    assert ax.imshow_kwargs is not None
    assert ax.imshow_kwargs["vmin"] == pytest.approx(power_db.max() - 75.0)


def test_create_range_doppler_figure_uses_monostatic_equivalent_velocity_label(
    monkeypatch,
    tmp_path: Path,
):
    ax = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(sensing_summary_fig.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)

    sensing_summary_fig.create_range_doppler_figure(
        sensing_data=_typed_rd_payload(
            range_axis_m=np.linspace(0.0, 7.0, 8, dtype=np.float64),
            doppler_axis_hz=np.linspace(-500.0, 250.0, 4, dtype=np.float64),
            power_linear=np.ones((4, 8), dtype=np.float32),
        ),
        output_path=tmp_path / "rd.png",
    )

    assert ax.ylabel == "Monostatic-equivalent Velocity (m/s)"


def test_create_range_doppler_figure_preserves_nonuniform_authored_axes_and_truth_ids(
    monkeypatch,
    tmp_path: Path,
):
    ax = _FakeAxes()
    fig = _FakeFigure()
    monkeypatch.setattr(sensing_summary_fig.plt, "subplots", lambda *args, **kwargs: (fig, ax))
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda *_args, **_kwargs: None)
    range_axis = np.array([0.25, 1.0, 3.5], dtype=np.float64)
    doppler_axis = np.array([-200.0, -25.0, 300.0], dtype=np.float64)

    sensing_summary_fig.create_range_doppler_figure(
        _typed_rd_payload(
            range_axis_m=range_axis,
            doppler_axis_hz=doppler_axis,
            power_linear=np.arange(1.0, 10.0, dtype=np.float64).reshape(3, 3),
        ),
        output_path=tmp_path / "rd_nonuniform.png",
    )

    assert ax.pcolormesh_args is not None
    np.testing.assert_array_equal(ax.pcolormesh_args[0], range_axis)
    np.testing.assert_allclose(
        ax.pcolormesh_args[1],
        doppler_axis_to_velocity_m_s(doppler_axis, 28e9),
    )
    assert ax.pcolormesh_kwargs["shading"] == "nearest"
    assert ax.scatters[0][2]["label"] == "GT target-beta"
    assert ax.scatters[1][2]["label"] == "GT target-alpha"


def test_create_range_doppler_figure_rejects_ambiguous_typed_products(tmp_path: Path):
    sensing = _typed_rd_payload(
        range_axis_m=np.array([1.0, 2.0], dtype=np.float64),
        doppler_axis_hz=np.array([-10.0, 10.0], dtype=np.float64),
        power_linear=np.ones((2, 2), dtype=np.float32),
    )
    source_key = "range_doppler_power_by_pair"
    duplicate_key = "alternate_rd_power"
    sensing[duplicate_key] = np.array(sensing[source_key], copy=True)
    sensing["products"][duplicate_key] = {
        **sensing["products"][source_key],
        "data_key": duplicate_key,
    }

    with pytest.raises(ValueError, match="Ambiguous sensing products"):
        sensing_summary_fig.create_range_doppler_figure(
            sensing,
            output_path=tmp_path / "ambiguous.png",
        )


def test_create_range_azimuth_figure_writes_file(tmp_path: Path) -> None:
    output_path = sensing_summary_fig.create_range_azimuth_figure(
        _typed_angular_payload(),
        gt_state=None,
        output_path=tmp_path / "range_azimuth.png",
    )

    assert output_path.exists()


def test_create_angular_overview_figure_writes_file(tmp_path: Path) -> None:
    output_path = sensing_summary_fig.create_angular_overview_figure(
        _typed_angular_payload(),
        gt_state=None,
        output_path=tmp_path / "angular_overview.png",
    )

    assert output_path.exists()


def test_create_xy_tracking_figure_writes_file(tmp_path: Path) -> None:
    track_history = {
        1: [
            {"frame": 0, "x_m": 5.0, "y_m": 1.0, "world_x_m": 1.0, "world_y_m": 2.0},
            {"frame": 1, "x_m": 5.5, "y_m": 1.3, "world_x_m": 1.5, "world_y_m": 2.4},
        ],
        2: [
            {"frame": 0, "x_m": 7.0, "y_m": -1.0, "world_x_m": 3.0, "world_y_m": 1.0},
            {"frame": 1, "x_m": 7.4, "y_m": -0.7, "world_x_m": 3.3, "world_y_m": 1.2},
        ],
    }
    raw_detections = [
        {"x_m": 1.0, "y_m": 2.0},
        {"x_m": 1.5, "y_m": 2.4},
        {"x_m": 3.0, "y_m": 1.0},
        {"x_m": 3.3, "y_m": 1.2},
    ]
    gt_tracks = {
        0: [{"frame": 0, "x_m": 0.9, "y_m": 2.0}, {"frame": 1, "x_m": 1.4, "y_m": 2.3}],
        1: [{"frame": 0, "x_m": 3.1, "y_m": 1.1}, {"frame": 1, "x_m": 3.4, "y_m": 1.3}],
    }
    radar_positions = [
        {"frame": 0, "x_m": 0.0, "y_m": -4.8},
        {"frame": 1, "x_m": 0.0, "y_m": -4.8},
    ]

    output_path = sensing_summary_fig.create_xy_tracking_figure(
        track_history,
        output_path=tmp_path / "tracking_xy.png",
        raw_detections=raw_detections,
        gt_tracks=gt_tracks,
        radar_positions=radar_positions,
    )

    assert output_path.exists()


def test_single_stft_column_has_nonzero_display_extent(monkeypatch, tmp_path):
    ax, fig = _FakeAxes(), _FakeFigure()
    monkeypatch.setattr(sensing_summary_fig.plt, "subplots", lambda **kw: (fig, ax))
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda _: None)
    result = MicroDopplerResult(
        spectrogram=np.ones((1, 64)),
        time_axis_s=np.array([0.032]),
        doppler_axis_hz=np.fft.fftshift(np.fft.fftfreq(64, 0.001)),
        range_bin=0,
        range_m=4.5,
    )
    before = result.spectrogram.copy()
    sensing_summary_fig.create_micro_doppler_figure(result, tmp_path / "one_cpi.png")
    extent = ax.imshow_kwargs["extent"]
    assert extent[:2] == pytest.approx([0, 0.064])
    assert extent[3] - extent[2] == pytest.approx(1000)
    np.testing.assert_array_equal(result.spectrogram, before)


@pytest.mark.parametrize("speeds", [[1.8, 1.81], [-1.8, -1.81], [-0.2, 0.2], [0.0, 0.0]])
def test_truth_overview_includes_zero_and_preserves_signed_samples(tmp_path, monkeypatch, speeds):
    captured = []
    close = sensing_summary_fig.plt.close
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda fig: captured.append(fig))
    history = [
        dict(
            step=i,
            target_ids=["cube"],
            target_ranges=[5.0 - i * 0.01],
            target_speeds=[v],
            target_velocity_valid=[True],
            det_ranges=[4.5 - i * 0.01],
            det_velocities=[v],
        )
        for i, v in enumerate(speeds)
    ]
    sensing_summary_fig.create_gt_evolution_figure(history, tmp_path / "truth.png")
    fig = captured[-1]
    assert fig.axes[0].get_ylim()[0] == 0
    low, high = fig.axes[1].get_ylim()
    assert low <= min(0, *speeds) and high >= max(0, *speeds)
    np.testing.assert_array_equal(fig.axes[1].lines[0].get_ydata(), speeds)
    np.testing.assert_array_equal(fig.axes[0].collections[0].get_offsets()[:, 1], [4.5, 4.49])
    close(fig)


def test_rd_acquisition_annotations_are_outside_unchanged_map(tmp_path, monkeypatch):
    captured = []
    close = sensing_summary_fig.plt.close
    monkeypatch.setattr(sensing_summary_fig.plt, "close", lambda fig: captured.append(fig))
    power = np.arange(1, 25, dtype=float).reshape(4, 6)
    sensing_summary_fig.create_range_doppler_array_figure(
        power_linear=power,
        range_axis_m=np.arange(6.0),
        doppler_axis_hz=np.arange(4.0) * 10,
        range_mode="monostatic",
        carrier_freq_hz=28e9,
        output_path=tmp_path / "rd.png",
        detections=[dict(range_m=2, velocity_m_s=0.1)],
        display_config=dict(range_resolution_m=0.3, velocity_resolution_m_s=0.084),
    )
    fig = captured[-1]
    ax = fig.axes[0]
    fig.canvas.draw()
    assert all(t.get_position()[1] > 1 for t in ax.texts)
    assert "Velocity-bin spacing" in ax.texts[0].get_text()
    legend_bounds = ax.get_legend().get_window_extent(fig.canvas.get_renderer())
    assert legend_bounds.y0 >= ax.get_window_extent().y1
    np.testing.assert_array_equal(
        ax.collections[0].get_array().reshape(power.shape), 10 * np.log10(power)
    )
    close(fig)
