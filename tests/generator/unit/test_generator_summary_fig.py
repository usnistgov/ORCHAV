from types import SimpleNamespace

import mitsuba as mi
import numpy as np

if not hasattr(mi, "Point3f"):
    for _variant in ("llvm_ad_mono_polarized", "llvm_ad_rgb", "scalar_rgb"):
        if hasattr(mi, "variants") and _variant not in mi.variants():
            continue
        try:
            mi.set_variant(_variant)
            break
        except Exception:
            continue

from generator.core.scenario_actors.state import ActorStateCache
from generator.figures import generator_summary_fig
from generator.figures.generator_summary_fig import (
    _create_device_angular_velocity_figure,
    _summary_actor_label,
)
from generator.figures.motion import (
    collect_orientation_data_from_actor_state_manager,
    collect_velocity_data_from_actor_state_manager,
    orientations_to_angular_velocities,
    point_to_xyz,
    positions_to_tuples,
    positions_to_velocities,
)


def test_summary_actor_label_defaults_to_role_index():
    config = SimpleNamespace(name="MainTransmitter")

    assert _summary_actor_label("tx", 0, config) == "TX1"
    assert _summary_actor_label("rx", 3, config) == "RX4"


def test_summary_actor_label_can_use_yaml_name():
    config = SimpleNamespace(name="RX_Static")

    assert _summary_actor_label("rx", 0, config, "name") == "RX_Static"


def test_summary_actor_label_name_mode_falls_back_to_role_index():
    config = SimpleNamespace(name="")

    assert _summary_actor_label("tx", 1, config, "name") == "TX2"


def test_motion_helpers_convert_point_like_positions():
    positions = [
        SimpleNamespace(x=1.0, y=2.0, z=3.0),
        (4.0, 5.0, 6.0),
    ]

    assert point_to_xyz(positions[0]) == (1.0, 2.0, 3.0)
    assert positions_to_tuples(positions) == [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)]


def test_motion_helpers_velocity_uses_duration():
    positions = [
        SimpleNamespace(x=0.0, y=0.0, z=0.0),
        SimpleNamespace(x=2.0, y=0.0, z=0.0),
        SimpleNamespace(x=4.0, y=2.0, z=0.0),
    ]

    velocities = positions_to_velocities(positions, duration_s=2.0)

    assert velocities.tolist() == [[2.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0]]


def test_motion_helpers_orientation_differences():
    orientations = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [2.0, 3.0, 5.0]])

    angular_velocities = orientations_to_angular_velocities(orientations)

    assert angular_velocities.tolist() == [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 1.0, 2.0]]


def test_angular_summary_labels_per_step_euler_changes(monkeypatch, tmp_path):
    captured = {}
    original_figure = generator_summary_fig.plt.figure

    def capture_figure(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        captured["figure"] = figure
        return figure

    monkeypatch.setattr(generator_summary_fig.plt, "figure", capture_figure)
    output = _create_device_angular_velocity_figure(
        device_data=[
            {
                "name": "RX_Main",
                "angular_velocities": np.array(
                    [[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=np.float64
                ),
            }
        ],
        device_type="RX",
        device_name="Receivers",
        base_color="blue",
        time_steps=np.arange(2),
        simulation_config=SimpleNamespace(scene_name="unit_scene.xml", num_steps=2, duration=1.0),
        path_policy=None,
        scenario_context=None,
        show_markers=False,
        force_dir=tmp_path,
        force_ext=".png",
    )

    figure = captured["figure"]
    assert output == tmp_path / "rx_angular_velocity_evolution.png"
    assert output.is_file()
    assert figure._suptitle.get_text().startswith("Receivers Euler-Angle Change per Output Step")
    assert [axis.get_ylabel() for axis in figure.axes] == [
        "Euler-Angle Change (deg/output step)",
        "Yaw Change (deg/output step)",
        "Pitch Change (deg/output step)",
        "Roll Change (deg/output step)",
    ]
    assert all("deg/s" not in axis.get_ylabel() for axis in figure.axes)


def test_motion_helpers_collectors_preserve_actor_names():
    tx_positions = [[SimpleNamespace(x=0.0, y=0.0, z=0.0), SimpleNamespace(x=1.0, y=0.0, z=0.0)]]
    rx_positions = [[SimpleNamespace(x=0.0, y=1.0, z=0.0)]]
    target_positions = [[SimpleNamespace(x=0.0, y=0.0, z=1.0)]]
    tx_orientations = [np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])]
    rx_orientations = [np.array([[0.0, 0.0, 0.0]])]
    target_orientations = [np.array([[0.0, 0.0, 0.0]])]
    actor_state_manager = SimpleNamespace(
        tx_configs=[SimpleNamespace(name="TX_Main")],
        rx_configs=[SimpleNamespace(name="RX_Main")],
        target_managers=[SimpleNamespace(config=SimpleNamespace(name="Target_Main"))],
        prepare_cached=lambda: ActorStateCache(
            tx_positions=tx_positions,
            rx_positions=rx_positions,
            target_positions=target_positions,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
            target_orientations=target_orientations,
        ),
    )
    simulation_config = SimpleNamespace(duration=1.0)

    velocity_data = collect_velocity_data_from_actor_state_manager(
        actor_state_manager, simulation_config
    )
    orientation_data = collect_orientation_data_from_actor_state_manager(actor_state_manager)

    assert velocity_data["tx"][0]["name"] == "TX_Main"
    assert velocity_data["tx"][0]["velocities"].tolist() == [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    assert orientation_data["tx"][0]["name"] == "TX_Main"
    assert orientation_data["tx"][0]["angular_velocities"].tolist() == [
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ]
