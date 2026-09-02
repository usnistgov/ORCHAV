"""Exercise public visualizer paths with manifest-backed compact frames."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from shared.frames.contracts import PathMetric
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.types import StandardMPCFrame
from visualizer.src.io.frame_sources import FileSource
from visualizer.src.notebook.pygfx import PygfxNotebookViz
from visualizer.src.panels.data_source.widgets import FrameComparisonDialog
from visualizer.src.pipeline.core import MPCCore


def _compact_frame(
    frame_index: int,
    *,
    tx_count: int = 1,
    rx_count: int = 1,
    paths_per_pair: tuple[int, ...] = (1,),
) -> StandardMPCFrame:
    """Build one compact frame with a physical bounce on every path."""
    pairs = np.asarray(
        [
            (min(pair_index, tx_count - 1), min(pair_index, rx_count - 1))
            for pair_index in range(len(paths_per_pair))
        ],
        dtype=np.int32,
    )
    vertices_by_pair = []
    interactions_by_pair = []
    path_lengths_by_pair = []
    for pair_index, path_count in enumerate(paths_per_pair):
        vertices = np.empty((path_count, 1, 3), dtype=np.float32)
        for path_index in range(path_count):
            vertices[path_index, 0] = [5.0 + path_index, pair_index + 1.0, 1.0]
        vertices_by_pair.append(vertices)
        interactions_by_pair.append(np.ones((path_count, 1), dtype=np.uint8))
        path_lengths_by_pair.append(np.ones((path_count,), dtype=np.int64))

    metrics_by_pair = {
        metric: [
            np.arange(path_count, dtype=np.float32) + metric_index + 1.0
            for path_count in paths_per_pair
        ]
        for metric_index, metric in enumerate(PathMetric)
    }
    tx_positions = np.column_stack(
        (
            np.arange(tx_count, dtype=np.float64),
            np.zeros(tx_count, dtype=np.float64),
            np.ones(tx_count, dtype=np.float64),
        )
    )
    rx_positions = np.column_stack(
        (
            10.0 + np.arange(rx_count, dtype=np.float64),
            np.zeros(rx_count, dtype=np.float64),
            np.ones(rx_count, dtype=np.float64),
        )
    )
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_index,
        tx_rx_pairs=pairs,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        vertices_by_pair=vertices_by_pair,
        interactions_by_pair=interactions_by_pair,
        path_lengths_by_pair=path_lengths_by_pair,
        metrics_by_pair=metrics_by_pair,
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        provenance={"provider": "compact-public-path-test"},
    )


@pytest.fixture
def compact_frame_source(tmp_path: Path) -> Iterator[FileSource]:
    """Publish and open one real compact frame set for notebook tests."""
    writer = FrameSetWriter.for_scenario(tmp_path, compression=None)
    writer.append(_compact_frame(0))
    manifest = writer.finalize(provenance={"producer": "notebook-test"})
    assert manifest is not None

    source = FileSource(tmp_path, "frames", "h5")
    source.open()
    try:
        yield source
    finally:
        source.close()


def test_notebook_frame_builder_uses_projection_and_preserves_marker_overrides(
    compact_frame_source: FileSource,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = compact_frame_source.provider
    assert provider is not None
    projection_calls: list[int] = []
    original_projection_load = provider.load_frame_projection

    def record_projection(step: int, request: Any) -> Any:
        projection_calls.append(step)
        return original_projection_load(step, request)

    def reject_complete_load(step: int) -> StandardMPCFrame:
        raise AssertionError(f"Notebook requested complete frame {step}")

    monkeypatch.setattr(provider, "load_frame_projection", record_projection)
    monkeypatch.setattr(provider, "load_frame", reject_complete_load)

    visualizer = object.__new__(PygfxNotebookViz)
    visualizer._frame_source = compact_frame_source
    visualizer._mpc_core = MPCCore(visualizer=None)
    tx_override = [[50.0, 1.0, 2.0]]
    rx_override = [[60.0, 3.0, 4.0]]

    payload, view_model = visualizer._build_frame_data(
        frame=0,
        color_mode="reflection_order",
        selected_tx="all",
        selected_rx="all",
        mpc_layer_enabled=True,
        show_mpc_paths=True,
        show_mpc_bounce_points=True,
        tx_positions=tx_override,
        rx_positions=rx_override,
    )

    assert projection_calls == [0]
    assert payload is not None
    assert view_model is not None
    np.testing.assert_allclose(payload["tx_positions"], tx_override)
    np.testing.assert_allclose(payload["rx_positions"], rx_override)
    np.testing.assert_allclose(view_model.tx_positions, tx_override)
    np.testing.assert_allclose(view_model.rx_positions, rx_override)
    np.testing.assert_allclose(payload["canonical_data"].points[0], [0.0, 0.0, 1.0])
    np.testing.assert_allclose(payload["canonical_data"].points[-1], [10.0, 0.0, 1.0])


def test_frame_comparison_dialog_uses_compact_frame_counts(qtbot: Any) -> None:
    first = _compact_frame(4)
    second = _compact_frame(
        5,
        tx_count=2,
        rx_count=3,
        paths_per_pair=(1, 2),
    )
    dialog = FrameComparisonDialog(None)
    qtbot.addWidget(dialog)

    dialog._display_frame_info(first, dialog.frame1_info, first.frame_index)
    dialog._display_frame_info(second, dialog.frame2_info, second.frame_index)
    dialog._display_differences(first, second)

    assert dialog.frame1_info.toPlainText().splitlines() == [
        "Frame 4",
        "MPC Count: 1",
        "TX Count: 1",
        "RX Count: 1",
    ]
    assert dialog.frame2_info.toPlainText().splitlines() == [
        "Frame 5",
        "MPC Count: 3",
        "TX Count: 2",
        "RX Count: 3",
    ]
    assert dialog.diff_info.toPlainText().splitlines() == [
        "MPC Count: 1 -> 3 (delta 2)",
        "TX Count: 1 -> 2 (delta 1)",
        "RX Count: 1 -> 3 (delta 2)",
    ]
