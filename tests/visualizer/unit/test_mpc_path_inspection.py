"""Tests for the renderer-neutral selected-MPC path snapshot."""

from __future__ import annotations

import numpy as np
import pytest

from visualizer.src.renderers.mpc_path_inspection import MpcPathInspectionSnapshot


def test_snapshot_freezes_selected_path_and_precomputes_arc_length() -> None:
    points = np.asarray(
        ((0.0, 0.0, 0.0), (3.0, 0.0, 0.0), (3.0, 4.0, 0.0)),
        dtype=np.float64,
    )
    snapshot = MpcPathInspectionSnapshot(
        frame_token=("scenario-a", 7),
        canonical_path_id=42,
        points=points,
        bounce_interaction_types=np.asarray((1,), dtype=np.uint8),
        bounce_colors=np.asarray(((0.2, 0.4, 0.8),), dtype=np.float64),
    )
    points[:] = -99.0

    assert snapshot.canonical_path_id == 42
    assert snapshot.points.dtype == np.float32
    assert snapshot.points.flags.writeable is False
    assert snapshot.segment_lengths.flags.writeable is False
    assert snapshot.cumulative_lengths.flags.writeable is False
    np.testing.assert_allclose(snapshot.points[0], (0.0, 0.0, 0.0))
    np.testing.assert_allclose(snapshot.segment_lengths, (3.0, 4.0))
    np.testing.assert_allclose(snapshot.cumulative_lengths, (0.0, 3.0, 7.0))
    assert snapshot.total_length == pytest.approx(7.0)
    assert snapshot.bounce_labels == ("1",)


def test_snapshot_preserves_explicit_selected_bounce_labels() -> None:
    snapshot = MpcPathInspectionSnapshot(
        frame_token=1,
        canonical_path_id=0,
        points=np.asarray(
            ((0, 0, 0), (1, 0, 0), (2, 1, 0), (3, 1, 0)),
            dtype=np.float32,
        ),
        bounce_labels=("first", "second"),
    )

    assert snapshot.bounce_labels == ("first", "second")
    assert snapshot.bounce_interaction_types is None
    assert snapshot.bounce_colors is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"canonical_path_id": -1}, "canonical_path_id"),
        ({"points": np.zeros((1, 3), dtype=np.float32)}, "points"),
        (
            {"bounce_interaction_types": np.asarray((1, 2), dtype=np.int32)},
            "bounce_interaction_types",
        ),
        (
            {"bounce_colors": np.ones((2, 3), dtype=np.float32)},
            "bounce_colors",
        ),
        ({"bounce_labels": ("1", "2")}, "bounce_labels"),
    ],
)
def test_snapshot_rejects_misaligned_selected_path_data(kwargs, message) -> None:
    values = {
        "frame_token": "frame",
        "canonical_path_id": 0,
        "points": np.asarray(((0, 0, 0), (1, 0, 0), (2, 0, 0)), dtype=np.float32),
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=message):
        MpcPathInspectionSnapshot(**values)
