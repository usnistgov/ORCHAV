"""Local summary framing preserves actor samples and default bounds."""

from types import SimpleNamespace

import numpy as np
import pytest

from generator.figures import generator_summary_fig as figures


def test_scene2d_explicit_bounds_frame_local_view_without_changing_samples(tmp_path, monkeypatch):
    points = [[-20.0, -10.0, 1.0], [30.0, 20.0, 1.0]]
    cache = SimpleNamespace(
        tx_positions=[points],
        rx_positions=[],
        target_positions=[],
        tx_orientations=[[(0, 0, 0)] * 2],
        rx_orientations=[],
        target_orientations=[],
    )
    state = SimpleNamespace(prepare_cached=lambda: cache)
    simulation = SimpleNamespace(num_steps=2, duration=1.0, quality="low", scene_name="test")
    observed = []

    def record_figure(*_args, **_kwargs):
        ax = figures.plt.gca()
        observed.append(
            (ax.get_xlim(), ax.get_ylim(), [line.get_xydata().copy() for line in ax.lines])
        )

    monkeypatch.setattr(figures.plt, "savefig", record_figure)
    for bounds in [None, (-25.0, 35.0, -15.0, 25.0)]:
        figures.create_2d_scene_summary_figures(
            [SimpleNamespace(name="TX")],
            [],
            [],
            simulation,
            state,
            output_path=tmp_path / "view.png",
            explicit_bounds=bounds,
        )
    assert observed[0][0] == (-22.5, 32.5)
    assert observed[0][1] == (-12.0, 22.0)
    assert observed[1][0] == (-25.0, 35.0)
    assert observed[1][1] == (-15.0, 25.0)
    for default_line, local_line in zip(observed[0][2], observed[1][2]):
        np.testing.assert_array_equal(default_line, local_line)
    assert cache.tx_positions[0] == points


@pytest.mark.parametrize("bounds", [(0, 0, 1, 2), (0, 1, 2, 1), (0, np.nan, 1, 2), (0, 1, 2)])
def test_scene2d_rejects_invalid_explicit_bounds_before_creating_figure(bounds):
    before = figures.plt.get_fignums()
    with pytest.raises(ValueError, match="finite increasing"):
        figures.create_2d_scene_summary_figures([], [], [], None, None, explicit_bounds=bounds)
    assert figures.plt.get_fignums() == before


def test_scene2d_optional_target_palette_preserves_default_geometry(tmp_path, monkeypatch):
    points = [[0.0, 0.0, 1.0], [10.0, 5.0, 1.0]]
    cache = SimpleNamespace(
        tx_positions=[],
        rx_positions=[],
        target_positions=[points],
        tx_orientations=[],
        rx_orientations=[],
        target_orientations=[[(0, 0, 0)] * 2],
    )
    state = SimpleNamespace(prepare_cached=lambda: cache)
    simulation = SimpleNamespace(num_steps=2, duration=1.0, quality="low", scene_name="test")
    observed = []

    def record(*_args, **_kwargs):
        ax = figures.plt.gca()
        observed.append(
            [(line.get_color(), line.get_xydata().copy(), line.get_label()) for line in ax.lines]
        )

    monkeypatch.setattr(figures.plt, "savefig", record)
    for palette in [None, {"Drone": "#8956bd"}]:
        figures.create_2d_scene_summary_figures(
            [],
            [],
            [SimpleNamespace(config=SimpleNamespace(name="Drone"))],
            simulation,
            state,
            output_path=tmp_path / "view.png",
            target_colors=palette,
        )
    assert all(color == "g" for color, _, _ in observed[0])
    assert all(color == "#8956bd" for color, _, _ in observed[1])
    for before, after in zip(observed[0], observed[1]):
        np.testing.assert_array_equal(before[1], after[1])
    assert observed[0][-1][2] == "Target Trajectory"
    assert observed[1][-1][2] == "Drone"


def test_scene2d_invalid_target_palette_rejected_before_figure():
    before = figures.plt.get_fignums()
    with pytest.raises(ValueError, match="target_colors"):
        figures.create_2d_scene_summary_figures(
            [], [], [], None, None, target_colors={"Drone": "invalid-color"}
        )
    assert figures.plt.get_fignums() == before


@pytest.mark.parametrize("limits", [[0, 0, 1, 2], [0, 1, 2, 1], [0, np.inf, 1, 2], [0, 1, 2]])
def test_yaml_summary_rejects_invalid_scene2d_limits(limits):
    from shared.scenarios.model import VisualizationConfigModel

    with pytest.raises(ValueError):
        VisualizationConfigModel(scene2d_limits=limits)


@pytest.mark.parametrize("limits", [None, [-225.0, 175.0, -225.0, 125.0]])
def test_yaml_summary_dispatch_preserves_display_bounds(tmp_path, monkeypatch, limits):
    from generator.figures.summary import generator as dispatch
    from shared.scenarios.model import GeneratorSummaryModel

    model = GeneratorSummaryModel(
        enabled=True,
        create=["scene2d"],
        visualization={} if limits is None else {"scene2d_limits": limits},
    )
    scenario = SimpleNamespace(generator_summary=model.model_dump())
    simulation = SimpleNamespace(num_steps=2, duration=1)
    captured = []
    monkeypatch.setattr(dispatch, "ActorStateManager", lambda **_kwargs: object())
    monkeypatch.setattr(dispatch, "get_scene_geometry", lambda **_kwargs: None)

    def record(*_args, **kwargs):
        captured.append(kwargs["explicit_bounds"])
        return kwargs["output_path"]

    monkeypatch.setattr(dispatch, "create_2d_scene_summary_figures", record)
    paths = dispatch.maybe_generate_generator_summary(
        [], [], [], simulation, scenario, output_root=tmp_path, strict=True
    )
    assert captured == [limits]
    assert paths == [tmp_path / "topology/scene_2d.png"]


@pytest.mark.parametrize("dimension", [2, 3])
def test_receiver_palette_preserves_samples_and_names_legend(tmp_path, monkeypatch, dimension):
    points = [[[0.0, 0.0, 1.0], [10.0, 5.0, 1.0]], [[0.0, 0.0, 2.0], [7.0, 8.0, 2.0]]]
    cache = SimpleNamespace(
        tx_positions=[[[0, 0, 1]] * 2],
        rx_positions=points,
        target_positions=[],
        tx_orientations=[[(0, 0, 0)] * 2],
        rx_orientations=[[(0, 0, 0)] * 2] * 2,
        target_orientations=[],
    )
    state = SimpleNamespace(prepare_cached=lambda: cache)
    simulation = SimpleNamespace(num_steps=2, duration=1.0, quality="low", scene_name="test")
    captured = []

    def record(*_args, **_kwargs):
        ax = figures.plt.gca()
        captured.append(
            (
                [
                    (line.get_color(), line.get_data(), line.get_label(), line.get_zorder())
                    for line in ax.lines
                ],
                ax.get_legend_handles_labels()[1],
            )
        )

    monkeypatch.setattr(figures.plt, "savefig", record)
    draw = getattr(figures, f"create_{dimension}d_scene_summary_figures")
    for mode in ("role", "individual"):
        draw(
            [SimpleNamespace(name="TX")],
            [SimpleNamespace(name="Car"), SimpleNamespace(name="Drone")],
            [],
            simulation,
            state,
            output_path=tmp_path / "view.png",
            receiver_color_mode=mode,
        )
    old_lines, old_legend = captured[0]
    new_lines, new_legend = captured[1]
    assert "RX Trajectory" in old_legend
    assert "Car trajectory" in new_legend and "Drone trajectory" in new_legend
    routes = [line for line in new_lines if line[2].endswith("trajectory")]
    assert len(routes) == 2 and routes[0][0] != routes[1][0]
    for before, after in zip(old_lines, new_lines):
        np.testing.assert_array_equal(before[1], after[1])
    if dimension == 2:
        tx = next(line for line in new_lines if line[2] == "TX Stationary")
        assert tx[3] > max(line[3] for line in new_lines if line is not tx)


@pytest.mark.parametrize("mode", ["rainbow", "", None])
def test_summary_rejects_invalid_receiver_color_mode_before_drawing(mode):
    from shared.scenarios.model import VisualizationConfigModel

    with pytest.raises(ValueError):
        VisualizationConfigModel(receiver_color_mode=mode)
    before = figures.plt.get_fignums()
    with pytest.raises(ValueError, match="receiver_color_mode"):
        figures.create_2d_scene_summary_figures([], [], [], None, None, receiver_color_mode=mode)
    assert before == figures.plt.get_fignums()


def test_receiver_inset_reuses_exact_prepared_paths(tmp_path, monkeypatch):
    paths = [[[0.0, 0.0, 1.0], [2.0, 3.0, 1.0]], [[1.0, 0.0, 1.0], [3.0, 3.0, 1.0]]]
    cache = SimpleNamespace(
        tx_positions=[],
        rx_positions=paths,
        target_positions=[],
        tx_orientations=[],
        rx_orientations=[[(0, 0, 0)] * 2] * 2,
        target_orientations=[],
    )
    state = SimpleNamespace(prepare_cached=lambda: cache)
    simulation = SimpleNamespace(num_steps=2, duration=1.0, quality="low", scene_name="test")
    captured = []

    def record(*args, **kwargs):
        inset = figures.plt.gca().child_axes[0]
        captured.extend(line.get_xydata().copy() for line in inset.lines)

    monkeypatch.setattr(figures.plt, "savefig", record)
    figures.create_2d_scene_summary_figures(
        [],
        [SimpleNamespace(name="RX3"), SimpleNamespace(name="RX5")],
        [],
        simulation,
        state,
        output_path=tmp_path / "inset.pdf",
        receiver_color_mode="individual",
        receiver_inset_names=["RX3", "RX5"],
    )
    for source, drawn in zip(paths, captured, strict=True):
        np.testing.assert_array_equal(np.asarray(source)[:, :2], drawn)


def test_unknown_receiver_inset_fails_before_figure_allocation():
    before = figures.plt.get_fignums()
    with pytest.raises(ValueError, match="unknown receiver"):
        figures.create_2d_scene_summary_figures(
            [], [], [], None, None, receiver_inset_names=["missing"]
        )
    assert figures.plt.get_fignums() == before
