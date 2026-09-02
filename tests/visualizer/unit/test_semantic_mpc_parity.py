"""Semantic MPC parity tests independent of any renderer backend."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.frames import FrameReadRequest
from shared.frames.adapters import project_standard_mpc_frame
from shared.frames.schema import validate_standard_mpc_frame
from tests.visualizer.fixtures.semantic_mpc import (
    BASELINE_SEMANTIC_FRAME,
    assert_canonical_matches_semantics,
    assert_render_packet_matches_semantics,
    assert_renderer_neutral_packets_equal,
    build_semantic_oracle,
    build_standard_mpc_frame,
    semantic_frame_sequence,
)
from visualizer.src.io.packed_frame_payload import projection_to_visual_frame
from visualizer.src.pipeline.core import MPCCore
from visualizer.src.state import MpcVisibility

_ALL_TEST_INTERACTIONS = (0, 1, 2, 4, 8, 37, 99)


def _visual_payload(frame):
    """Project one canonical frame onto the renderer-facing payload seam."""

    projection = project_standard_mpc_frame(frame, FrameReadRequest.full())
    return projection_to_visual_frame(projection)


def _build_packet(core: MPCCore, variant, step: int):
    """Run one semantic frame through the current renderer-neutral pipeline."""
    frame = build_standard_mpc_frame(variant, frame_idx=step)
    view_model = core.create_view_model(
        step=step,
        raw_frame=_visual_payload(frame),
        color_mode="mpc_type",
        mpc_allowed_orders=(0, 1, 2, 3),
        mpc_allowed_types=_ALL_TEST_INTERACTIONS,
        mpc_visibility=MpcVisibility(enabled=True, paths=True, bounce_points=True),
    )
    assert view_model is not None
    return view_model.to_render_packet()


def test_semantic_standard_frame_is_valid_and_keeps_non_cartesian_pair_order():
    """The reusable fixture itself must satisfy the shared frame contract."""
    frame = build_standard_mpc_frame("baseline")

    assert (
        validate_standard_mpc_frame(
            frame,
            raise_on_error=False,
        )
        == []
    )
    np.testing.assert_array_equal(
        frame.tx_rx_pairs,
        np.asarray(((1, 0), (0, 1), (1, 1), (0, 0)), dtype=np.int32),
    )
    np.testing.assert_array_equal(np.diff(frame.pair_path_offsets), [2, 1, 0, 1])


def test_canonical_semantics_cover_order_geometry_metrics_and_materials():
    """Canonicalization preserves pair/path order and all scientific columns."""
    frame = build_standard_mpc_frame("baseline")

    canonical = _visual_payload(frame)["canonical_data"]

    assert_canonical_matches_semantics(canonical, BASELINE_SEMANTIC_FRAME)
    np.testing.assert_array_equal(
        canonical.path_tx,
        np.asarray([1, 1, 0, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        canonical.path_rx,
        np.asarray([0, 0, 1, 0], dtype=np.int16),
    )
    np.testing.assert_array_equal(
        canonical.path_orders,
        np.asarray([0, 1, 3, 2], dtype=np.uint8),
    )

    # Legitimate zero metrics remain measurements, while the NaN delay uses
    # the explicit geometric fallback and records that provenance.
    assert canonical.path_delays[0] == pytest.approx(0.0)
    assert canonical.path_losses[0] == pytest.approx(0.0)
    assert not canonical.path_delay_is_estimated[0]
    assert not canonical.path_loss_is_estimated[0]
    assert canonical.path_delay_is_estimated[2]
    assert np.isfinite(canonical.path_delays[2])

    # Unavailable angles remain NaN instead of becoming a valid zero angle.
    path_starts = canonical.path_start_indices
    assert path_starts is not None
    assert canonical.aoa_az is not None
    assert canonical.aoa_az[path_starts[0]] == pytest.approx(0.0)
    assert np.isnan(canonical.aoa_az[path_starts[2]])


def test_renderer_neutral_packet_preserves_unknown_types_and_full_geometry():
    """The backend boundary receives exact polylines and interaction identity."""
    core = MPCCore(logger=MagicMock())

    packet = _build_packet(core, BASELINE_SEMANTIC_FRAME, step=0)

    assert_render_packet_matches_semantics(packet, BASELINE_SEMANTIC_FRAME)
    oracle = build_semantic_oracle(BASELINE_SEMANTIC_FRAME)
    assert packet.mpc_points.shape == (14, 3)
    assert packet.mpc_lines.shape == (10, 2)
    assert packet.mpc_bounce_points.shape == (6, 3)
    assert set(packet.mpc_line_itypes.tolist()) == {0, 1, 2, 4, 8, 37, 99}

    unknown_segments = oracle.segment_interactions == 37
    virtual_segments = oracle.segment_interactions == 99
    np.testing.assert_allclose(
        packet.mpc_colors[unknown_segments],
        np.asarray([[0.5, 0.5, 0.5]], dtype=packet.mpc_colors.dtype),
    )
    np.testing.assert_allclose(
        packet.mpc_colors[virtual_segments],
        np.asarray(
            [[1.0, 0.6, 0.2], [1.0, 0.6, 0.2]],
            dtype=packet.mpc_colors.dtype,
        ),
    )
    assert "Unknown (Type 37)" in packet.stats_text
    assert "Virtual" in packet.stats_text


def test_populated_changed_empty_populated_packets_do_not_retain_stale_mpcs():
    """Frame transitions clear geometry and restore the original semantics."""
    core = MPCCore(logger=MagicMock())

    packets = tuple(
        _build_packet(core, frame_spec, step)
        for step, frame_spec in enumerate(semantic_frame_sequence())
    )
    first, changed, empty, restored = packets

    assert first.mpc_lines.shape == (10, 2)
    assert changed.mpc_lines.shape == (5, 2)
    assert empty.mpc_points.shape == (0, 3)
    assert empty.mpc_lines.shape == (0, 2)
    assert empty.mpc_colors.shape == (0, 3)
    assert empty.mpc_bounce_points.shape == (0, 3)
    assert empty.mpc_bounce_itypes is None
    assert empty.path_mask.shape == (0,)
    assert empty.segment_mask.shape == (0,)

    assert_renderer_neutral_packets_equal(first, restored)
    assert restored.mpc_points is not first.mpc_points
    assert restored.mpc_lines is not first.mpc_lines
