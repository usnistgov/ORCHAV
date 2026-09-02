"""Focused Qt tests for the complete mobility editor."""

from __future__ import annotations

from uuid import UUID

import pytest
from pydantic import ValidationError
from PySide6.QtWidgets import QAbstractSpinBox, QFileDialog

from shared.scenarios.actors import (
    MAX_RANDOM_SEED,
    ActorRole,
    CircularMobilitySpec,
    ConstantSpeedTraversalSpec,
    Figure8MobilitySpec,
    GaussMarkovMobilitySpec,
    GridScanMobilitySpec,
    GroupMemberMobilitySpec,
    GroupOffsetSpec,
    LinearMobilitySpec,
    ManhattanGridMobilitySpec,
    MeshSequenceMobilitySpec,
    NetworkRouteMobilitySpec,
    OscillatingMobilitySpec,
    PendulumMobilitySpec,
    RandomSamplingMobilitySpec,
    RandomWaypointMobilitySpec,
    SampledMobilitySpec,
    SpiralMobilitySpec,
    StationaryMobilitySpec,
    SurveyMobilitySpec,
    WaypointMobilitySpec,
)
from visualizer.src.authoring.mobility_editor import MobilityEditor
from visualizer.src.authoring.mobility_models import (
    MOBILITY_MODELS,
    MobilityKind,
    mobility_kind,
)

GROUP_ID = UUID("66f83594-f041-410f-8fa6-ad06d28bf80b")
CONSTANT_SPEED = ConstantSpeedTraversalSpec(
    speed_mps=2.5,
    after_end="ping_pong",
)

MOBILITIES = (
    StationaryMobilitySpec(position_m=(1.25, -2.5, 3.75)),
    LinearMobilitySpec(
        start_m=(-1.0, 2.0, 3.0),
        end_m=(4.0, -5.0, 6.0),
        traversal=CONSTANT_SPEED,
    ),
    WaypointMobilitySpec(
        points_m=((0.0, 0.0, 1.0), (2.0, 3.0, 4.0), (-5.0, 6.0, 7.0)),
        interpolation="catmull_rom",
    ),
    CircularMobilitySpec(
        center_m=(4.0, 5.0, 6.0),
        radius_m=7.5,
        start_angle_deg=-135.0,
        clockwise=True,
        turns=2.5,
        traversal=CONSTANT_SPEED,
    ),
    SurveyMobilitySpec(
        origin_m=(1.0, 2.0, 3.0),
        width_m=20.0,
        height_m=10.0,
        row_spacing_m=2.0,
        heading_deg=35.0,
        traversal=CONSTANT_SPEED,
    ),
    GridScanMobilitySpec(
        x_bounds_m=(-2.0, 8.0),
        y_bounds_m=(1.0, 9.0),
        z_bounds_m=(3.0, 7.0),
        x_steps=4,
        y_steps=3,
        z_steps=2,
        traversal_pattern="raster",
        start_corner="top_right",
        interpolation="catmull_rom",
        traversal=CONSTANT_SPEED,
    ),
    OscillatingMobilitySpec(
        center_m=(1.0, 2.0, 3.0),
        axis=(1.0, -2.0, 0.5),
        amplitude_m=4.0,
        frequency_hz=1.25,
        phase_deg=45.0,
    ),
    PendulumMobilitySpec(
        pivot_m=(1.0, 2.0, 9.0),
        length_m=6.0,
        max_angle_deg=40.0,
        frequency_hz=0.75,
        plane="yz",
        phase_deg=-30.0,
    ),
    Figure8MobilitySpec(
        center_m=(1.0, 2.0, 3.0),
        size_m=8.0,
        plane="xz",
        turns=3.0,
        traversal=CONSTANT_SPEED,
    ),
    SpiralMobilitySpec(
        center_m=(1.0, 2.0, 3.0),
        radius_m=5.0,
        start_altitude_m=3.0,
        end_altitude_m=30.0,
        turns=2.0,
        start_angle_deg=90.0,
        clockwise=True,
        traversal=CONSTANT_SPEED,
    ),
    RandomSamplingMobilitySpec(
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-5.0, 5.0),
        z_bounds_m=(1.0, 4.0),
        initial_position_m=(0.0, 0.0, 2.0),
        seed=42,
        sampling="poisson_disk",
        min_distance_m=0.5,
    ),
    GaussMarkovMobilitySpec(
        initial_position_m=(1.0, 2.0, 3.0),
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-20.0, 20.0),
        z_bounds_m=(3.0, 3.0),
        alpha=0.75,
        mean_speed_mps=3.0,
        mean_direction_deg=25.0,
        speed_std_mps=0.5,
        direction_std_deg=4.0,
        seed=123,
    ),
    RandomWaypointMobilitySpec(
        initial_position_m=(1.0, 2.0, 3.0),
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-20.0, 20.0),
        z_bounds_m=(3.0, 8.0),
        speed_range_mps=(1.0, 4.0),
        pause_range_s=(0.5, 2.0),
        seed=456,
    ),
    ManhattanGridMobilitySpec(
        origin_xy_m=(10.0, -5.0),
        block_size_m=25.0,
        grid_width=6,
        grid_height=7,
        altitude_m=4.0,
        turn_probability=0.25,
        speed_range_mps=(2.0, 6.0),
        pause_range_s=(0.25, 1.0),
        seed=789,
    ),
    NetworkRouteMobilitySpec(
        travel_mode="car",
        route="shortest_path",
        altitude_m=2.0,
        seed=99,
        graph_path="maps/network.graphml",
        start_node="A",
        end_node="B",
        traversal=CONSTANT_SPEED,
    ),
    MeshSequenceMobilitySpec(
        positions_path="motion/positions.npy",
        position_key="actor_positions",
        interpolation="step",
        traversal=CONSTANT_SPEED,
    ),
    GroupMemberMobilitySpec(
        group=str(GROUP_ID),
        offset_m=GroupOffsetSpec(right=1.0, forward=-2.0, up=3.0),
    ),
)


@pytest.mark.parametrize("mobility", MOBILITIES, ids=lambda value: value.type)
def test_all_canonical_models_round_trip_without_type_signal(qapp, mobility) -> None:
    editor = MobilityEditor()
    editor.set_group_choices(((GROUP_ID, "Convoy"),))
    emitted: list[str] = []
    editor.mobility_type_changed.connect(emitted.append)

    editor.set_mobility(mobility)

    kind = mobility_kind(mobility)
    assert editor.type_combo.currentData() == kind.value
    assert editor.page_stack.currentWidget() is editor._pages[kind.value]
    assert editor.mobility() == mobility
    assert emitted == []
    editor.close()


def test_registry_drives_all_type_labels_and_pages(qapp) -> None:
    editor = MobilityEditor()

    assert editor.type_combo.count() == 17
    assert [
        (
            editor.type_combo.itemData(index),
            editor.type_combo.itemText(index),
        )
        for index in range(editor.type_combo.count())
    ] == [(kind.value, descriptor.label) for kind, descriptor in MOBILITY_MODELS.items()]
    assert set(editor._pages) == {kind.value for kind in MobilityKind}
    editor.close()


def test_sampled_mobility_is_preserved_read_only_then_editor_recovers(qapp) -> None:
    editor = MobilityEditor()
    sampled = SampledMobilitySpec(
        positions_m=(
            (0.0, 1.0, 2.0),
            (3.0, 4.0, 5.0),
            (6.0, 7.0, 8.0),
        )
    )
    type_changes: list[str] = []
    apply_requests: list[bool] = []
    editor.mobility_type_changed.connect(type_changes.append)
    editor.apply_requested.connect(lambda: apply_requests.append(True))

    editor.set_mobility(sampled, preserve_type_signal=True)

    assert editor.mobility() is sampled
    assert editor.type_combo.currentData() == "sampled"
    assert editor.type_combo.currentText() == "Sampled (read-only)"
    assert not editor.type_combo.model().item(editor.type_combo.currentIndex()).isEnabled()
    assert editor.page_stack.currentWidget() is editor.read_only_page
    assert editor.read_only_mobility_label.text() == (
        "3 exact timeline positions. This mobility is preserved unchanged "
        "and is not editable here."
    )
    assert not editor.type_combo.isEnabled()
    assert not editor.page_stack.isEnabled()
    assert not editor.apply_button.isEnabled()
    editor.apply_button.click()
    assert type_changes == []
    assert apply_requests == []

    stationary = StationaryMobilitySpec(position_m=(9.0, 10.0, 11.0))
    editor.set_mobility(stationary)

    assert editor.type_combo.count() == len(MOBILITY_MODELS)
    assert editor.type_combo.currentData() == "stationary"
    assert editor.page_stack.currentWidget() is editor._pages["stationary"]
    assert editor.mobility() == stationary
    assert editor.type_combo.isEnabled()
    assert editor.page_stack.isEnabled()
    assert editor.apply_button.isEnabled()
    assert type_changes == []
    assert apply_requests == []
    editor.close()


def test_type_signal_controls_pages_without_constructing_a_model(qapp) -> None:
    editor = MobilityEditor()
    emitted: list[str] = []
    editor.mobility_type_changed.connect(emitted.append)

    editor.set_mobility(
        LinearMobilitySpec(start_m=(1.0, 2.0, 3.0), end_m=(4.0, 5.0, 6.0)),
        preserve_type_signal=True,
    )

    assert emitted == ["linear"]
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("stationary"))
    assert emitted == ["linear", "stationary"]
    editor.close()


def test_waypoint_table_supports_reorder_add_remove_draw_and_apply(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(
        WaypointMobilitySpec(points_m=((1.0, 0.0, 0.0), (2.0, 0.0, 0.0), (3.0, 0.0, 0.0)))
    )
    draw_requests: list[bool] = []
    apply_requests: list[bool] = []
    editor.draw_waypoints_requested.connect(lambda: draw_requests.append(True))
    editor.apply_requested.connect(lambda: apply_requests.append(True))

    editor.waypoint_table.selectRow(1)
    editor.waypoint_up_button.click()
    assert editor.mobility().points_m == (
        (2.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
    )
    editor.waypoint_down_button.click()
    editor.waypoint_add_button.click()
    assert len(editor.mobility().points_m) == 4
    editor.waypoint_remove_button.click()
    assert len(editor.mobility().points_m) == 3

    editor.draw_waypoints_button.click()
    editor.apply_button.click()
    assert draw_requests == [True]
    assert apply_requests == [True]
    assert editor.waypoint_table.cellWidget(0, 0).objectName() == "waypointPoint0XSpin"
    editor.close()


def test_traversal_fields_live_in_initially_collapsed_advanced_sections(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(MOBILITIES[1])

    assert not editor.linear_advanced_section.expanded
    assert editor.linear_advanced_section.content.isHidden()
    assert editor.linear_traversal_type_combo.currentData() == "constant_speed"
    assert editor.linear_traversal_speed_spin.value() == 2.5
    assert editor.linear_traversal_after_end_combo.currentData() == "ping_pong"

    editor.linear_advanced_section.set_expanded(True)
    assert editor.linear_advanced_section.expanded
    assert not editor.linear_advanced_section.content.isHidden()
    editor.close()


def test_group_choices_store_uuid_and_restore_unavailable_reference(qapp) -> None:
    editor = MobilityEditor()
    editor.set_group_choices(((GROUP_ID, "Convoy"),))
    editor.set_mobility(MOBILITIES[-1])

    assert editor.group_member_group_combo.currentText() == "Convoy"
    assert editor.group_member_group_combo.currentData() == GROUP_ID
    assert editor.mobility() == MOBILITIES[-1]

    editor.set_group_choices((), preserve_selection=False)
    editor.set_mobility(MOBILITIES[-1])
    assert "Unavailable group" in editor.group_member_group_combo.currentText()
    assert editor.mobility().group == str(GROUP_ID)
    editor.close()


def test_resource_browse_buttons_populate_editable_paths(qapp, monkeypatch) -> None:
    editor = MobilityEditor()
    calls: list[tuple[object, ...]] = []
    selections = iter(
        (
            ("C:/inputs/network.graphml", "Network graphs"),
            ("C:/inputs/positions.npy", "Position sequences"),
        )
    )

    def choose_file(*args, **_kwargs):
        calls.append(args)
        return next(selections)

    monkeypatch.setattr(QFileDialog, "getOpenFileName", choose_file)

    editor.network_route_browse_button.click()
    editor.mesh_sequence_browse_button.click()

    assert editor.network_route_graph_path_edit.text() == "C:/inputs/network.graphml"
    assert editor.mesh_sequence_positions_path_edit.text() == "C:/inputs/positions.npy"
    assert calls[0][3] == "Network graphs (*.graphml *.xml *.json);;All files (*)"
    editor.close()


def test_seed_controls_cover_the_complete_shared_schema_range(qapp) -> None:
    editor = MobilityEditor()
    seed_spins = (
        editor.random_sampling_seed_spin,
        editor.gauss_markov_seed_spin,
        editor.random_waypoint_seed_spin,
        editor.manhattan_grid_seed_spin,
        editor.network_route_seed_spin,
    )

    for spin in seed_spins:
        assert spin.minimum() == 0
        assert spin.maximum() == MAX_RANDOM_SEED
        spin.setValue(MAX_RANDOM_SEED)
        assert spin.value() == MAX_RANDOM_SEED

    editor.close()


def test_network_random_walk_suppresses_nodes_and_requires_a_default_seed(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(
        NetworkRouteMobilitySpec(
            route="shortest_path",
            seed=None,
            graph_path="maps/network.graphml",
            start_node="A",
            end_node="B",
        )
    )

    assert not editor.network_route_seed_enabled_check.isChecked()
    editor.network_route_route_combo.setCurrentIndex(
        editor.network_route_route_combo.findData("random_walk")
    )

    assert not editor.network_route_start_node_edit.isEnabled()
    assert not editor.network_route_end_node_edit.isEnabled()
    assert editor.network_route_start_node_edit.text() == "A"
    assert editor.network_route_end_node_edit.text() == "B"
    assert editor.network_route_seed_enabled_check.isChecked()
    assert not editor.network_route_seed_enabled_check.isEnabled()
    random_walk = editor.mobility()
    assert random_walk.route == "random_walk"
    assert random_walk.seed == 0
    assert random_walk.start_node is None
    assert random_walk.end_node is None

    editor.network_route_route_combo.setCurrentIndex(
        editor.network_route_route_combo.findData("shortest_path")
    )
    assert editor.network_route_start_node_edit.isEnabled()
    assert editor.network_route_end_node_edit.isEnabled()
    assert editor.network_route_start_node_edit.text() == "A"
    assert editor.network_route_end_node_edit.text() == "B"
    assert not editor.network_route_seed_enabled_check.isChecked()
    shortest_path = editor.mobility()
    assert shortest_path.route == "shortest_path"
    assert shortest_path.seed is None
    assert shortest_path.start_node == "A"
    assert shortest_path.end_node == "B"
    editor.close()


def test_network_route_mode_round_trip_preserves_an_explicit_shortest_path_seed(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(
        NetworkRouteMobilitySpec(
            route="shortest_path",
            seed=37,
            graph_path="maps/network.graphml",
            start_node="A",
            end_node="B",
        )
    )

    editor.network_route_route_combo.setCurrentIndex(
        editor.network_route_route_combo.findData("random_walk")
    )
    assert editor.mobility().seed == 37
    editor.network_route_route_combo.setCurrentIndex(
        editor.network_route_route_combo.findData("shortest_path")
    )

    restored = editor.mobility()
    assert restored.route == "shortest_path"
    assert restored.seed == 37
    editor.close()


def test_invalid_fields_raise_only_when_model_is_requested(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(
        RandomSamplingMobilitySpec(
            x_bounds_m=(0.0, 1.0),
            y_bounds_m=(0.0, 1.0),
            z_bounds_m=(0.0, 1.0),
            seed=1,
        )
    )

    editor.random_sampling_x_bounds_spins[0].setValue(10.0)
    editor.random_sampling_x_bounds_spins[1].setValue(-10.0)

    with pytest.raises(ValidationError):
        editor.mobility()
    editor.close()


def test_random_sampling_initial_observation_can_be_enabled_or_omitted(qapp) -> None:
    editor = MobilityEditor()
    unanchored = RandomSamplingMobilitySpec(
        x_bounds_m=(-10.0, 10.0),
        y_bounds_m=(-5.0, 5.0),
        z_bounds_m=(2.0, 2.0),
        seed=9,
    )

    editor.set_mobility(unanchored)
    assert not editor.random_sampling_initial_position_enabled_check.isChecked()
    assert not editor.random_sampling_initial_position_editor.isEnabled()
    assert editor.mobility() == unanchored

    editor.random_sampling_initial_position_enabled_check.setChecked(True)
    for spin, value in zip(
        editor.random_sampling_initial_position_spins,
        (4.0, -3.0, 2.0),
    ):
        spin.setValue(value)

    assert editor.mobility().initial_position_m == (4.0, -3.0, 2.0)
    editor.close()


def test_random_sampling_modes_explain_spacing_and_use_a_visible_poisson_default(qapp) -> None:
    editor = MobilityEditor()
    editor.type_combo.setCurrentIndex(editor.type_combo.findData("random_sampling"))

    assert "independent spatial observation" in editor.random_sampling_explanation_label.text()
    assert "allows clusters" in editor.random_sampling_explanation_label.text()

    editor.random_sampling_sampling_combo.setCurrentIndex(
        editor.random_sampling_sampling_combo.findData("poisson_disk")
    )

    assert editor.random_sampling_advanced_section.expanded
    assert editor.random_sampling_min_distance_spin.isEnabled()
    assert editor.random_sampling_min_distance_spin.value() == pytest.approx(1.0)
    assert "rejects candidates" in editor.random_sampling_explanation_label.text()
    assert editor.mobility().min_distance_m == pytest.approx(1.0)
    editor.close()


def test_gauss_markov_controls_explain_memory_heading_and_noise(qapp) -> None:
    editor = MobilityEditor()

    explanation = editor.gauss_markov_explanation_label.text()
    assert "Memory α=0" in explanation
    assert "α near 1" in explanation
    assert "0° is +X" in explanation
    assert "Motion memory" in editor.gauss_markov_alpha_spin.toolTip()
    assert "Standard deviation" in editor.gauss_markov_direction_std_spin.toolTip()
    editor.close()


def test_numeric_controls_are_adaptive_and_editor_state_is_respected(qapp) -> None:
    editor = MobilityEditor()

    assert editor.objectName() == "mobilityEditor"
    assert editor.type_combo.objectName() == "mobilityTypeCombo"
    numeric_controls = editor.findChildren(QAbstractSpinBox)
    assert numeric_controls
    for control in numeric_controls:
        assert control.stepType() == QAbstractSpinBox.StepType.AdaptiveDecimalStepType
        assert control.buttonSymbols() == QAbstractSpinBox.ButtonSymbols.UpDownArrows

    editor.set_average_speed(12.3456)
    assert editor.average_speed_label.text() == "Average speed: 12.346 m/s (computed)"
    editor.set_average_speed(None)
    assert editor.average_speed_label.text() == "Average speed: —"

    editor.set_editing_enabled(False)
    assert not editor.type_combo.isEnabled()
    assert not editor.page_stack.isEnabled()
    assert not editor.apply_button.isEnabled()
    assert not editor.draw_waypoints_button.isEnabled()
    editor.set_editing_enabled(True)
    assert editor.type_combo.isEnabled()
    assert editor.page_stack.isEnabled()
    assert editor.apply_button.isEnabled()
    editor.close()


def test_waypoint_drawing_reserves_editor_until_session_finishes(qapp) -> None:
    editor = MobilityEditor()
    editor.set_mobility(WaypointMobilitySpec(points_m=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0))))

    editor.set_waypoint_drawing_active(True)
    assert editor.draw_waypoints_button.text() == "Drawing…"
    assert not editor.type_combo.isEnabled()
    assert not editor.waypoint_table.isEnabled()
    assert not editor.draw_waypoints_button.isEnabled()
    assert not editor.apply_button.isEnabled()

    editor.set_waypoint_drawing_active(False)
    assert editor.draw_waypoints_button.text() == "Draw Waypoints"
    assert editor.type_combo.isEnabled()
    assert editor.waypoint_table.isEnabled()
    assert editor.apply_button.isEnabled()
    editor.close()


def test_target_only_mesh_sequence_role_is_surfaced_without_hiding_it(qapp) -> None:
    editor = MobilityEditor()
    mesh_index = editor.type_combo.findData(MobilityKind.MESH_SEQUENCE.value)

    editor.set_actor_role(ActorRole.TX)
    assert not editor.type_combo.model().item(mesh_index).isEnabled()
    editor.set_actor_role(ActorRole.TARGET)
    assert editor.type_combo.model().item(mesh_index).isEnabled()
    assert editor.type_combo.itemData(mesh_index) == MobilityKind.MESH_SEQUENCE.value
    editor.close()
