"""Tests for the public per-pair canonical-frame normalizer."""

import numpy as np
import pytest

from shared.frames import (
    PATH_METRIC_VALIDITY_BITS,
    PathMetric,
    StandardMPCFrame,
    standard_mpc_frame_from_pair_data,
)


def test_padded_pair_data_is_compacted_once_with_materials_and_metrics() -> None:
    frame = standard_mpc_frame_from_pair_data(
        frame_index=4,
        tx_rx_pairs=[[0, 0], [0, 1]],
        tx_positions=[[0.0, 0.0, 1.0]],
        rx_positions=[[1.0, 0.0, 1.0], [2.0, 0.0, 1.0]],
        vertices_by_pair=[
            np.asarray(
                [
                    [[np.nan, np.nan, np.nan]] * 3,
                    [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0], [np.nan] * 3],
                ],
                dtype=np.float64,
            ),
            np.empty((0, 0, 3), dtype=np.float32),
        ],
        interactions_by_pair=[
            np.asarray([[-1, -1, -1], [1, 2, -1]], dtype=np.int16),
            np.empty((0, 0), dtype=np.int16),
        ],
        path_lengths_by_pair=[
            np.asarray([0, 2], dtype=np.int32),
            np.empty((0,), dtype=np.int32),
        ],
        material_names_by_pair=[
            np.asarray([["", "", ""], ["brick", "glass", ""]], dtype=object),
            np.empty((0, 0), dtype=object),
        ],
        material_itu_types_by_pair=[
            np.asarray([["", "", ""], ["itu_brick", "itu_glass", ""]], dtype=object),
            np.empty((0, 0), dtype=object),
        ],
        metrics_by_pair={
            PathMetric.DELAY_NS: [np.asarray([0.0, 7.5]), np.empty((0,))],
            "path_loss_db": [np.asarray([np.nan, 82.0]), np.empty((0,))],
        },
        target_positions_m=[3.0, 4.0, 0.5],
        targets_metadata=[{"name": "vehicle"}],
    )

    assert isinstance(frame, StandardMPCFrame)
    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 2, 2])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 0, 2])
    np.testing.assert_allclose(
        frame.bounce_xyz_m,
        [[1.0, 1.0, 1.0], [1.5, 1.0, 1.0]],
    )
    assert frame.bounce_xyz_m.dtype == np.float32
    assert frame.interactions.dtype == np.uint8
    assert frame.material_ids.dtype == np.uint16
    assert frame.material_names == ("", "brick", "glass")
    assert frame.material_itu_types == ("", "itu_brick", "itu_glass")
    np.testing.assert_array_equal(frame.material_ids, [1, 2])
    np.testing.assert_array_equal(
        frame.metric_is_valid(PathMetric.DELAY_NS),
        [True, True],
    )
    np.testing.assert_array_equal(
        frame.metric_is_valid(PathMetric.PATH_LOSS_DB),
        [False, True],
    )
    assert frame.num_targets == 1


def test_ragged_pair_data_infers_lengths_and_defaults_optional_axes() -> None:
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int64),
        tx_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        rx_positions=np.asarray([[2.0, 0.0, 0.0]], dtype=np.float32),
        vertices_by_pair=[
            [
                np.empty((0, 3), dtype=np.float32),
                np.asarray([[1.0, 1.0, 0.0]], dtype=np.float32),
            ]
        ],
        interactions_by_pair=[
            [
                np.empty((0,), dtype=np.uint8),
                np.asarray([99], dtype=np.uint8),
            ]
        ],
    )

    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 2])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 0, 1])
    np.testing.assert_array_equal(frame.material_ids, [0])
    assert frame.material_names == ("",)
    assert frame.tx_names == ("tx_0",)
    assert frame.rx_names == ("rx_0",)
    assert not np.any(frame.metric_valid_bits)
    for metric in PathMetric:
        assert np.all(np.isnan(frame.path_metrics[metric]))


def test_rectangular_ragged_data_without_lengths_uses_full_depth() -> None:
    frame = standard_mpc_frame_from_pair_data(
        frame_index=1,
        tx_rx_pairs=[[0, 0]],
        tx_positions=[[0.0, 0.0, 0.0]],
        rx_positions=[[1.0, 0.0, 0.0]],
        vertices_by_pair=[np.zeros((2, 2, 3), dtype=np.float32)],
        interactions_by_pair=[np.ones((2, 2), dtype=np.uint8)],
    )

    np.testing.assert_array_equal(frame.bounce_offsets, [0, 2, 4])


def test_ragged_lengths_must_agree_with_the_path_arrays() -> None:
    with pytest.raises(ValueError, match="disagrees with the ragged paths"):
        standard_mpc_frame_from_pair_data(
            frame_index=0,
            tx_rx_pairs=[[0, 0]],
            tx_positions=[[0.0, 0.0, 0.0]],
            rx_positions=[[1.0, 0.0, 0.0]],
            vertices_by_pair=[[[[0.0, 0.0, 0.0]]]],
            interactions_by_pair=[[[1]]],
            path_lengths_by_pair=[[0]],
        )


def test_direct_constructor_reuses_canonical_numpy_buffers() -> None:
    pairs = np.asarray([[0, 0]], dtype=np.int32)
    pair_offsets = np.asarray([0, 1], dtype=np.int64)
    bounce_offsets = np.asarray([0, 0], dtype=np.int64)
    tx_positions = np.zeros((1, 3), dtype=np.float64)
    rx_positions = np.ones((1, 3), dtype=np.float64)
    orientations = np.zeros((1, 3), dtype=np.float64)
    bounces = np.empty((0, 3), dtype=np.float32)
    interactions = np.empty((0,), dtype=np.uint8)
    materials = np.empty((0,), dtype=np.uint16)
    metric = np.asarray([np.nan], dtype=np.float32)
    validity = np.asarray([0], dtype=np.uint8)
    targets = np.empty((0, 3), dtype=np.float64)

    frame = StandardMPCFrame(
        frame_index=0,
        tx_rx_pairs=pairs,
        pair_path_offsets=pair_offsets,
        bounce_offsets=bounce_offsets,
        tx_positions=tx_positions,
        rx_positions=rx_positions,
        tx_orientations=orientations,
        rx_orientations=orientations,
        tx_names=("tx",),
        rx_names=("rx",),
        bounce_xyz_m=bounces,
        interactions=interactions,
        material_ids=materials,
        material_names=("",),
        material_itu_types=("",),
        delays_ns=metric,
        path_loss_db=metric,
        aoa_az_deg=metric,
        aoa_el_deg=metric,
        aod_az_deg=metric,
        aod_el_deg=metric,
        metric_valid_bits=validity,
        target_positions_m=targets,
        targets_metadata=(),
    )

    assert frame.tx_rx_pairs is pairs
    assert frame.pair_path_offsets is pair_offsets
    assert frame.bounce_xyz_m is bounces
    assert frame.delays_ns is metric
    assert frame.metric_valid_bits is validity


def test_metric_validity_bits_use_the_stable_metric_contract() -> None:
    assert PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS] == 1
    assert PATH_METRIC_VALIDITY_BITS[PathMetric.AOD_EL_DEG] == 32
