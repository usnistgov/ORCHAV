from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from visualizer.src.scene.trajectory_payloads import (
    TrajectoryNaming,
    build_trajectory_payloads,
    sanitize_trajectory_name,
)


def _naming() -> TrajectoryNaming:
    return TrajectoryNaming(
        tx_lines="trajectory_tx_lines",
        tx_points="trajectory_tx_points",
        rx_lines="trajectory_rx_lines",
        rx_points="trajectory_rx_points",
        target_lines_prefix="trajectory_target_lines_",
        target_points_prefix="trajectory_target_points_",
    )


def test_node_trajectory_payload_uses_individual_rx_color_offset() -> None:
    viz = SimpleNamespace(
        node_coloring_mode="individual",
        individual_node_colors=[
            [1.0, 0.0, 0.0],
            [0.8, 0.2, 0.2],
            [0.0, 0.0, 1.0],
            [0.2, 0.6, 1.0],
        ],
        tx_markers=[object(), object()],
    )
    trajectory_data = {
        "rx_positions": {1: [(2, 1.0, 0.0, 0.0), (0, 0.0, 0.0, 0.0), (1, 0.5, 0.0, 0.0)]}
    }

    batch = build_trajectory_payloads(
        kind="rx",
        trajectory_data=trajectory_data,
        visualizer=viz,
        naming=_naming(),
    )

    assert batch.stale_names == ()
    payload = batch.payloads[0]
    assert payload.lines_name == "trajectory_rx_lines"
    np.testing.assert_allclose(payload.point_payload.points[:, 0], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(payload.point_payload.colors, [[0.2, 0.6, 1.0]] * 3)
    assert payload.line_payload is not None
    np.testing.assert_allclose(payload.line_payload.colors, [[0.2, 0.6, 1.0]] * 2)


def test_target_trajectory_payload_uses_material_color_and_reports_stale_names() -> None:
    viz = SimpleNamespace(
        node_coloring_mode="individual",
        target_entries=[{"target_name": "Target A/1", "color": [0.3, 0.4, 0.5]}],
    )
    trajectory_data = {
        "target_positions": {
            "Target A/1": [(0, 0.0, 0.0, 0.0), (1, 1.0, 0.0, 0.0)],
            "Target:B": [(0, 0.0, 1.0, 0.0)],
        }
    }
    existing = (
        "trajectory_target_lines_Target_A_1",
        "trajectory_target_points_Target_A_1",
        "trajectory_target_lines_stale",
    )

    batch = build_trajectory_payloads(
        kind="target",
        trajectory_data=trajectory_data,
        visualizer=viz,
        naming=_naming(),
        existing_names=existing,
    )

    names = {(payload.lines_name, payload.points_name) for payload in batch.payloads}
    assert (
        "trajectory_target_lines_Target_A_1",
        "trajectory_target_points_Target_A_1",
    ) in names
    assert (
        "trajectory_target_lines_Target_B",
        "trajectory_target_points_Target_B",
    ) in names
    assert batch.stale_names == ("trajectory_target_lines_stale",)
    first = batch.payloads[0]
    assert first.line_payload is not None
    np.testing.assert_allclose(first.line_payload.colors, [[0.3, 0.4, 0.5]])


def test_empty_trajectory_payload_reports_kind_names_for_removal() -> None:
    batch = build_trajectory_payloads(
        kind="tx",
        trajectory_data={"tx_positions": {}},
        visualizer=SimpleNamespace(),
        naming=_naming(),
    )

    assert batch.payloads == ()
    assert batch.stale_names == ("trajectory_tx_lines", "trajectory_tx_points")


def test_sanitize_trajectory_name_replaces_unsafe_characters() -> None:
    assert sanitize_trajectory_name("Target A/1:phase\\x") == "Target_A_1_phase_x"
