"""Regression tests for the complete compact MPC frame contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, fields, replace

import numpy as np
import pytest

from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.schema import (
    FRAME_SCHEMA_VERSION,
    count_frame_mpcs,
    is_valid_standard_mpc_frame,
    summarize_frame,
    validate_standard_mpc_frame,
)
from shared.frames.types import FRAME_FORMAT_VERSION, StandardMPCFrame


def _minimal_frame() -> StandardMPCFrame:
    """Return one device pair with no propagation paths."""

    return standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.5]], dtype=np.float64),
        rx_positions=np.asarray([[10.0, 0.0, 1.5]], dtype=np.float64),
        vertices_by_pair=[np.empty((0, 2, 3), dtype=np.float32)],
        interactions_by_pair=[np.empty((0, 2), dtype=np.uint8)],
        path_lengths_by_pair=[np.empty((0,), dtype=np.int64)],
    )


def _single_path_frame() -> StandardMPCFrame:
    """Return one line-of-sight path with complete metric data."""

    vertices = np.asarray(
        [[[0.0, 0.0, 1.5], [10.0, 0.0, 1.5]]],
        dtype=np.float32,
    )
    return standard_mpc_frame_from_pair_data(
        frame_index=3,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.5]], dtype=np.float64),
        rx_positions=np.asarray([[10.0, 0.0, 1.5]], dtype=np.float64),
        vertices_by_pair=[vertices],
        interactions_by_pair=[np.ones((1, 2), dtype=np.uint8)],
        path_lengths_by_pair=[np.asarray([2], dtype=np.int64)],
        metrics_by_pair={
            "delay_ns": [np.asarray([25.0], dtype=np.float32)],
            "path_loss_db": [np.asarray([72.0], dtype=np.float32)],
        },
        provenance={"producer": "schema-regression"},
    )


def test_frame_contract_version_is_v2() -> None:
    """Pin the public in-memory contract and its schema alias."""

    assert FRAME_FORMAT_VERSION == 2
    assert FRAME_SCHEMA_VERSION == FRAME_FORMAT_VERSION


def test_complete_frame_field_surface_is_stable() -> None:
    """Pin the fields needed by producers, providers, and consumers."""

    expected = {
        "frame_index",
        "tx_rx_pairs",
        "pair_path_offsets",
        "bounce_offsets",
        "tx_positions",
        "rx_positions",
        "tx_orientations",
        "rx_orientations",
        "tx_names",
        "rx_names",
        "bounce_xyz_m",
        "interactions",
        "material_ids",
        "material_names",
        "material_itu_types",
        "delays_ns",
        "path_loss_db",
        "aoa_az_deg",
        "aoa_el_deg",
        "aod_az_deg",
        "aod_el_deg",
        "metric_valid_bits",
        "target_positions_m",
        "targets_metadata",
        "version",
        "timestamp_s",
        "sensing",
        "beamforming",
        "provenance",
        "recomputed_from_stored_positions",
    }
    assert {field.name for field in fields(StandardMPCFrame)} == expected


def test_frame_is_structurally_immutable() -> None:
    """Prevent consumers from replacing canonical arrays accidentally."""

    frame = _minimal_frame()
    with pytest.raises(FrozenInstanceError):
        frame.frame_index = 2  # type: ignore[misc]


def test_validation_accepts_only_complete_frame_instances() -> None:
    """Reject mappings at the complete-frame package boundary."""

    frame = _minimal_frame()
    assert validate_standard_mpc_frame(frame, raise_on_error=False) == []
    assert is_valid_standard_mpc_frame(frame)

    candidate = {"frame_index": 0}
    assert validate_standard_mpc_frame(candidate, raise_on_error=False) == [
        "frame must be a complete StandardMPCFrame instance"
    ]
    assert not is_valid_standard_mpc_frame(candidate)
    with pytest.raises(ValueError, match="complete StandardMPCFrame"):
        validate_standard_mpc_frame(candidate)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"frame_index": -1}, "frame_index"),
        ({"version": 1}, "Unsupported StandardMPCFrame version"),
        ({"timestamp_s": np.inf}, "timestamp_s"),
        ({"tx_rx_pairs": np.asarray([[0, 0, 0]], dtype=np.int32)}, "tx_rx_pairs"),
        ({"tx_rx_pairs": np.asarray([[0, 0]], dtype=np.int64)}, "tx_rx_pairs"),
        ({"pair_path_offsets": np.asarray([0], dtype=np.int32)}, "pair_path_offsets"),
        ({"bounce_offsets": np.asarray([1], dtype=np.int64)}, "bounce_offsets"),
        (
            {"tx_positions": np.asarray([[0.0, 0.0, 1.5]], dtype=np.float32)},
            "tx_positions",
        ),
        (
            {"rx_positions": np.asarray([[10.0, 0.0]], dtype=np.float64)},
            "rx_positions",
        ),
    ],
)
def test_direct_constructor_rejects_invalid_core_arrays(
    changes: dict[str, object],
    message: str,
) -> None:
    """Keep exact shape, dtype, offset, and scalar requirements enforced."""

    with pytest.raises(ValueError, match=message):
        replace(_minimal_frame(), **changes)


def test_pair_references_must_resolve_to_devices() -> None:
    """Reject topology that refers to a missing transmitter or receiver."""

    frame = _minimal_frame()
    with pytest.raises(ValueError, match="unknown transmitter"):
        replace(frame, tx_rx_pairs=np.asarray([[1, 0]], dtype=np.int32))
    with pytest.raises(ValueError, match="unknown receiver"):
        replace(frame, tx_rx_pairs=np.asarray([[0, 1]], dtype=np.int32))


def test_offsets_must_describe_pair_path_and_bounce_axes() -> None:
    """Require every compact axis to terminate at its resident data length."""

    frame = _single_path_frame()
    with pytest.raises(ValueError, match="pair count plus one"):
        replace(frame, pair_path_offsets=np.asarray([0], dtype=np.int64))
    with pytest.raises(ValueError, match="path count plus one"):
        replace(frame, bounce_offsets=np.asarray([0], dtype=np.int64))
    with pytest.raises(ValueError, match="bounce_xyz_m length"):
        replace(frame, bounce_xyz_m=np.empty((1, 3), dtype=np.float32))


def test_material_catalog_and_ids_are_consistent() -> None:
    """Require the reserved empty material and valid catalog references."""

    frame = _single_path_frame()
    with pytest.raises(ValueError, match="row zero"):
        replace(frame, material_names=("wall",))
    with pytest.raises(ValueError, match="unknown material"):
        replace(frame, material_ids=np.asarray([1, 1], dtype=np.uint16))


def test_metric_arrays_align_with_paths_and_validity_bits() -> None:
    """Require one finite-or-NaN metric value and validity state per path."""

    frame = _single_path_frame()
    with pytest.raises(ValueError, match="delay_ns length"):
        replace(frame, delays_ns=np.empty((0,), dtype=np.float32))
    with pytest.raises(ValueError, match="Invalid delay_ns entries must contain NaN"):
        replace(
            frame,
            delays_ns=np.asarray([25.0], dtype=np.float32),
            metric_valid_bits=np.zeros((1,), dtype=np.uint8),
        )


def test_target_metadata_position_cannot_override_canonical_target_state() -> None:
    """Keep target_positions_m as the single position authority."""

    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.empty((0, 2), dtype=np.int32),
        tx_positions=np.empty((0, 3), dtype=np.float64),
        rx_positions=np.empty((0, 3), dtype=np.float64),
        vertices_by_pair=[],
        interactions_by_pair=[],
        target_positions_m=np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        targets_metadata=({"name": "target", "current_position": [1.0, 2.0, 3.0]},),
    )
    assert is_valid_standard_mpc_frame(frame)

    with pytest.raises(ValueError, match="current_position must match"):
        replace(
            frame,
            targets_metadata=({"name": "target", "current_position": [9.0, 9.0, 9.0]},),
        )


def test_empty_topology_frame_is_valid() -> None:
    """Allow a frame containing no devices, pairs, paths, or targets."""

    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.empty((0, 2), dtype=np.int32),
        tx_positions=np.empty((0, 3), dtype=np.float64),
        rx_positions=np.empty((0, 3), dtype=np.float64),
        vertices_by_pair=[],
        interactions_by_pair=[],
    )
    assert is_valid_standard_mpc_frame(frame)
    assert frame.num_tx == frame.num_rx == frame.num_pairs == frame.num_paths == 0


def test_schema_helpers_report_compact_frame_content() -> None:
    """Keep counting and human-readable summaries on the canonical arrays."""

    frame = _single_path_frame()
    assert count_frame_mpcs(frame) == 1
    summary = summarize_frame(frame)
    assert "TX: 1, RX: 1, Targets: 0" in summary
    assert "Pairs: 1" in summary
    assert "Total MPCs: 1" in summary
    assert "delays_ns" in summary
    assert "path_loss_db" in summary
