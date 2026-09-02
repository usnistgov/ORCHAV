"""Semantic tests for compact projection-to-visualizer canonicalization."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock

import numpy as np
import pytest

from shared.frames import (
    PATH_METRIC_VALIDITY_BITS,
    FrameReadRequest,
)
from shared.frames.adapters import project_standard_mpc_frame
from tests.visualizer.fixtures.semantic_mpc import (
    assert_canonical_matches_semantics,
    assert_render_packet_matches_semantics,
    build_standard_mpc_frame,
)
from visualizer.src.io.packed_frame_payload import projection_to_visual_frame
from visualizer.src.metrics.mpc_canon import (
    build_filter_mask,
    colorize_segments,
    ensure_luts,
)
from visualizer.src.metrics.mpc_path_catalog import MpcPathCatalog
from visualizer.src.metrics.mpc_stats import MPCStatsComputer
from visualizer.src.metrics.packed_canon import canonical_from_projection
from visualizer.src.pipeline.core import MPCCore
from visualizer.src.services.beamforming_service import BeamformingService
from visualizer.src.state import MpcVisibility

_ALL_INTERACTIONS = (0, 1, 2, 4, 8, 37, 99)


def _projection(variant: str = "baseline"):
    frame = build_standard_mpc_frame(variant, frame_idx=3)
    return frame, project_standard_mpc_frame(frame, FrameReadRequest.full())


@pytest.mark.parametrize("points_dtype", [np.float32, np.float64])
def test_projected_canonical_matches_semantic_oracle(points_dtype) -> None:
    _frame, projection = _projection()

    compact = canonical_from_projection(projection, points_dtype=points_dtype)

    assert_canonical_matches_semantics(compact)
    assert compact.points.dtype == np.dtype(points_dtype)
    assert compact.points.flags.c_contiguous
    assert compact.lines.flags.c_contiguous
    assert set(compact.itype.tolist()) >= {0, 1, 2, 4, 8, 37, 99}


def test_projection_validity_bits_drive_geometric_fallbacks() -> None:
    _frame, projection = _projection()
    canonical = canonical_from_projection(projection)

    np.testing.assert_array_equal(
        canonical.path_delay_is_estimated,
        np.asarray([False, False, True, False]),
    )
    np.testing.assert_array_equal(
        canonical.path_loss_is_estimated,
        np.asarray([False, False, False, False]),
    )
    assert canonical.path_delays[0] == pytest.approx(0.0)
    assert np.isfinite(canonical.path_delays[2])
    assert canonical.path_losses[0] == pytest.approx(0.0)
    assert canonical.aoa_az is not None
    assert np.isnan(canonical.aoa_az[canonical.path_start_indices[2]])


def test_projection_canonical_does_not_mutate_metric_sources_when_filling_fallbacks() -> None:
    frame, projection = _projection()
    original_delays = frame.delays_ns.copy()
    original_losses = frame.path_loss_db.copy()

    canonical_from_projection(projection)

    np.testing.assert_array_equal(frame.delays_ns, original_delays)
    np.testing.assert_array_equal(frame.path_loss_db, original_losses)


def test_projection_canonical_reuses_fully_valid_float32_path_metrics() -> None:
    frame, _projection_value = _projection()
    all_valid = np.full(
        frame.metric_valid_bits.shape,
        sum(PATH_METRIC_VALIDITY_BITS.values()),
        dtype=np.uint8,
    )
    replacements = {"metric_valid_bits": all_valid}
    for field_name in (
        "delays_ns",
        "path_loss_db",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
    ):
        values = getattr(frame, field_name)
        replacements[field_name] = np.nan_to_num(values, nan=0.0).astype(
            np.float32,
            copy=False,
        )
    fully_valid = replace(frame, **replacements)
    projection = project_standard_mpc_frame(fully_valid, FrameReadRequest.full())

    canonical = canonical_from_projection(projection)

    assert canonical.path_delays is fully_valid.delays_ns
    assert canonical.path_losses is fully_valid.path_loss_db


def test_projection_canonical_supports_catalog_filter_color_and_stats_consumers() -> None:
    _frame, projection = _projection()
    compact = canonical_from_projection(projection)

    compact_catalog = MpcPathCatalog(compact)
    np.testing.assert_array_equal(compact_catalog.path_ids, np.arange(4, dtype=np.int32))
    np.testing.assert_array_equal(compact_catalog.tx_ids, [1, 1, 0, 0])
    np.testing.assert_array_equal(compact_catalog.rx_ids, [0, 0, 1, 0])
    assert np.all(compact_catalog.geometric_lengths_m > 0.0)

    filter_kwargs = {
        "selected_tx": "all",
        "selected_rx": "all",
        "allowed_orders": (1, 2, 3),
        "allowed_types": _ALL_INTERACTIONS,
        "delay_min_ns": 5.0,
        "delay_max_ns": 30.0,
    }
    compact_mask = build_filter_mask(compact, **filter_kwargs)
    assert all(mask.dtype == np.dtype(np.bool_) for mask in compact_mask)
    assert compact_mask[0].shape == (14,)
    assert compact_mask[1].shape == (10,)

    segment_mask = np.ones((len(compact.lines),), dtype=np.bool_)
    order_palette = np.zeros((8, 3), dtype=np.float32)
    type_palette = np.zeros((100, 3), dtype=np.float32)
    viridis = ensure_luts()
    colors = colorize_segments(
        compact,
        segment_mask,
        "mpc_type",
        order_palette,
        type_palette,
        viridis,
    )
    assert colors.shape == (10, 3)
    assert np.all(np.isfinite(colors))

    computer = MPCStatsComputer()
    compact_stats = computer.compute_frame_stats(compact)
    assert compact_stats.total_paths == 4
    assert compact_stats.orders_hist == {0: 1, 1: 1, 2: 1, 3: 1}


def test_projected_canonical_svd_paths_preserve_non_cartesian_pairs() -> None:
    _frame, projection = _projection()
    compact = canonical_from_projection(projection)
    service = BeamformingService(MagicMock())

    compact_paths = service.extract_mpc_paths(compact)
    compact_selected = service.extract_mpc_paths(compact, selected_pair=(1, 0))

    assert compact_paths is not None
    assert compact_selected is not None
    assert set(compact_paths) == {(1, 0), (0, 1), (0, 0)}
    assert set(compact_selected) == {(1, 0)}
    assert len(compact_selected[(1, 0)]) == 2
    assert compact_selected[(1, 0)][1][0].shape == (3, 3)
    assert np.shares_memory(compact_selected[(1, 0)][1][0], compact.points)


def test_compact_payload_drives_semantic_renderer_packet_without_padded_arrays() -> None:
    _frame, projection = _projection()
    compact_payload = projection_to_visual_frame(projection)

    compact_core = MPCCore(logger=MagicMock())
    kwargs = {
        "step": 3,
        "color_mode": "mpc_type",
        "mpc_allowed_orders": (0, 1, 2, 3),
        "mpc_allowed_types": _ALL_INTERACTIONS,
        "mpc_visibility": MpcVisibility(
            enabled=True,
            paths=True,
            bounce_points=True,
        ),
    }
    compact_view = compact_core.create_view_model(
        raw_frame=compact_payload,
        **kwargs,
    )
    assert compact_view is not None
    assert_render_packet_matches_semantics(compact_view.to_render_packet())
    assert "all_padded_vertices" not in compact_payload
    assert "all_padded_interactions" not in compact_payload
    assert "all_path_lengths" not in compact_payload


def test_empty_projection_clears_every_renderer_aligned_array() -> None:
    _frame, projection = _projection("empty")

    canonical = canonical_from_projection(projection)
    payload = projection_to_visual_frame(projection)

    assert canonical.points.shape == (0, 3)
    assert canonical.lines.shape == (0, 2)
    assert canonical.material_ids is None
    assert canonical.segment_material_ids.shape == (0,)
    assert payload["canonical_data"] is not None
    assert payload["num_tx"] == 2
    assert payload["num_rx"] == 2
