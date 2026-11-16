"""Noise-sweep figure metric and denominator contracts."""

from pathlib import Path

import numpy as np
import pytest

from shared.sensing.plotting.noise import (
    _prepare_rows,
    _seed_summary,
    generate_noise_sweep_figures,
)


def _row(**updates):
    row = dict(
        regime="AWGN",
        sample_snr_db=-10,
        seed=1,
        recall=0.5,
        false_reports_per_cpi=0.25,
        range_rmse_m=0.3,
        velocity_rmse_m_s=0.2,
        matched_count=10,
    )
    return row | updates


def test_zero_matches_masks_errors_instead_of_drawing_perfect_accuracy():
    rows = _prepare_rows([_row(matched_count=0, range_rmse_m=0, velocity_rmse_m_s=0)])
    assert rows[0]["range_rmse_m"] is None
    assert rows[0]["velocity_rmse_m_s"] is None
    assert np.isnan(_seed_summary(rows, "range_rmse_m")[0])


def test_descriptive_seed_summary_ignores_missing_without_zero_imputation():
    rows = [_row(recall=0.2), _row(seed=2, recall=0.8), _row(seed=3, recall=None)]
    assert _seed_summary(rows, "recall") == (0.5, 0.2, 0.8, 2)


def test_rejects_duplicate_seed_rows_and_nonfinite_snr():
    with pytest.raises(ValueError, match="one aggregate row"):
        _prepare_rows([_row(), _row()])
    with pytest.raises(ValueError, match="finite"):
        _prepare_rows([_row(sample_snr_db=np.inf)])


def test_empty_sweep_does_not_create_output(tmp_path):
    assert generate_noise_sweep_figures(tmp_path, []) == []
    assert not (tmp_path / "figures").exists()


def test_figures_keep_labels_missing_values_and_common_rd_scale(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    figures = []
    savefig = Figure.savefig

    def capture(self, *args, **kwargs):
        figures.append(self)
        return savefig(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    original_facecolor = plt.rcParams["axes.facecolor"]
    rows = [
        _row(sample_snr_db=None, regime="Clean", tracked_fraction=1, id_switches=0),
        _row(sample_snr_db=-20, matched_count=0, range_rmse_m=0),
        _row(),
        _row(seed=2, recall=0.7),
        _row(regime="Noise-only (reference-power SNR)", recall=None, matched_count=0),
    ]
    examples = [
        dict(
            label=name,
            power=np.ones((3, 4)) * power,
            range_m=np.arange(4),
            velocity_m_s=np.arange(3),
        )
        for name, power in [("Clean, CPI 1", 100), ("Noisy, seed 1, CPI 1", 1)]
    ]
    paths = generate_noise_sweep_figures(tmp_path, rows, rd_examples=examples)
    assert len(paths) == 4
    assert all(path.is_file() and path.stat().st_size > 1000 for path in paths)
    assert plt.rcParams["axes.facecolor"] == original_facecolor
    assert figures[0].axes[1].get_ylabel() == "Raw false reports / evaluated CPI"
    assert "not a CI" in figures[0].texts[-1].get_text()
    range_axis = figures[1].axes[0]
    awgn_line = next(line for line in range_axis.lines if line.get_label() == "AWGN")
    assert np.isnan(awgn_line.get_ydata()[0])
    maps = [axis.collections[0] for axis in figures[-1].axes[:2]]
    assert maps[0].get_clim() == maps[1].get_clim() == (-80, 0)
    assert np.allclose(maps[0].get_array(), 0)
    assert np.allclose(maps[1].get_array(), -20)
    assert not plt.get_fignums()


def test_rd_rejects_wrong_orientation(tmp_path: Path):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="shape"):
        generate_noise_sweep_figures(
            tmp_path,
            [_row()],
            rd_examples=[
                dict(
                    label="Wrong",
                    power=np.ones((4, 3)),
                    range_m=np.arange(4),
                    velocity_m_s=np.arange(3),
                )
            ],
        )


def test_confirmed_metrics_and_per_target_panels(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    figures = []
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        figures.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    paths = generate_noise_sweep_figures(
        tmp_path,
        [
            _row(
                tracked_recall=0.4,
                confirmed_false_reports_per_cpi=0.1,
                per_target={"weak": {"recall": 0, "tracked_recall": 0, "matched_count": 0}},
            )
        ],
    )
    assert len(figures[0].axes) == 4
    assert "Confirmed false reports" in figures[0].axes[-1].get_ylabel()
    assert paths[-1].name == "noise_sweep_per_target.png"
    assert figures[-1].axes[0].lines[0].get_ydata()[0] == 0


def test_rd_display_limits_do_not_change_power_or_add_empty_truth_legend(tmp_path, monkeypatch):
    pytest.importorskip("matplotlib")
    from matplotlib.figure import Figure

    figures = []
    original = Figure.savefig

    def capture(self, *args, **kwargs):
        figures.append(self)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    example = dict(
        label="Noise only",
        power=np.ones((3, 4)),
        range_m=np.arange(4),
        velocity_m_s=np.arange(3),
        truth_range_m=[],
        truth_velocity_m_s=[],
        display_range_m=[1, 2],
        display_velocity_m_s=[0.5, 1.5],
    )
    generate_noise_sweep_figures(tmp_path, [_row()], rd_examples=[example] * 4)
    axes = figures[-1].axes
    assert len(axes) == 7  # 2x3 display grid plus shared color bar
    assert axes[0].get_xlim() == (1.0, 2.0)
    assert axes[0].get_ylim() == (0.5, 1.5)
    assert axes[0].get_legend() is None
    assert axes[0].collections[0].get_array().shape == (3, 4)
    assert np.all(axes[0].collections[0].get_array() == 0)
    assert not axes[4].get_visible() and not axes[5].get_visible()


@pytest.mark.parametrize("limits", [[1, 1], [2, 1], [0, float("nan")], [1]])
def test_rd_rejects_invalid_display_limits(tmp_path, limits):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="display_range_m"):
        generate_noise_sweep_figures(
            tmp_path,
            [_row()],
            rd_examples=[
                dict(
                    label="invalid",
                    power=np.ones((3, 4)),
                    range_m=np.arange(4),
                    velocity_m_s=np.arange(3),
                    display_range_m=limits,
                )
            ],
        )


@pytest.mark.parametrize(
    "overlay",
    [
        {"detected_range_m": [1]},
        {"detected_range_m": [1, 2], "detected_velocity_m_s": [1]},
        {"detected_range_m": [1], "detected_velocity_m_s": [float("nan")]},
    ],
)
def test_rd_rejects_invalid_report_coordinates(tmp_path, overlay):
    pytest.importorskip("matplotlib")
    with pytest.raises(ValueError, match="detected_range_m"):
        generate_noise_sweep_figures(
            tmp_path,
            [_row()],
            rd_examples=[
                dict(
                    label="invalid",
                    power=np.ones((3, 4)),
                    range_m=np.arange(4),
                    velocity_m_s=np.arange(3),
                    **overlay,
                )
            ],
        )
