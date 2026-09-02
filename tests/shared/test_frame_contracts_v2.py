"""Tests for the complete canonical frame and selective projections."""

from collections.abc import Mapping
from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from shared.frames import (
    FRAME_FORMAT_VERSION,
    MPC_FRAME_MANIFEST_VERSION,
    MPC_HDF5_LAYOUT,
    MPC_HDF5_SCHEMA_VERSION,
    PACKED_MPC_FRAME_VERSION,
    FrameComponent,
    FrameProjection,
    FrameReadRequest,
    PathMetric,
    ProjectedMPCFrame,
    StandardMPCFrame,
)
from shared.frames.contracts import PATH_METRIC_VALIDITY_BITS


def _minimal_standard_frame() -> StandardMPCFrame:
    valid_delay = PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS]
    unavailable = np.full((3,), np.nan, dtype=np.float32)
    return StandardMPCFrame(
        frame_index=7,
        tx_rx_pairs=np.asarray([[0, 0], [0, 1], [0, 2]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, 1, 1, 3], dtype=np.int64),
        # The first path is LoS; the next two have one and two bounces.
        bounce_offsets=np.asarray([0, 0, 1, 3], dtype=np.int64),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray(
            [[1.0, 0.0, 1.0], [2.0, 0.0, 1.0], [3.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((3, 3), dtype=np.float64),
        tx_names=("tx",),
        rx_names=("rx-0", "rx-1", "rx-2"),
        bounce_xyz_m=np.asarray(
            [[1.5, 1.0, 1.0], [2.0, 1.0, 1.0], [2.5, 1.0, 1.0]],
            dtype=np.float32,
        ),
        interactions=np.asarray([1, 2, 99], dtype=np.uint8),
        material_ids=np.asarray([1, 1, 0], dtype=np.uint16),
        material_names=("", "concrete"),
        material_itu_types=("", "itu_concrete"),
        delays_ns=np.asarray([0.0, 3.0, np.nan], dtype=np.float32),
        path_loss_db=unavailable.copy(),
        aoa_az_deg=unavailable.copy(),
        aoa_el_deg=unavailable.copy(),
        aod_az_deg=unavailable.copy(),
        aod_el_deg=unavailable.copy(),
        metric_valid_bits=np.asarray([valid_delay, valid_delay, 0], dtype=np.uint8),
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
    )


def _project_geometry_and_delay(frame: StandardMPCFrame) -> ProjectedMPCFrame:
    return ProjectedMPCFrame(
        frame_index=frame.frame_index,
        tx_rx_pairs=frame.tx_rx_pairs,
        pair_path_offsets=frame.pair_path_offsets,
        bounce_offsets=frame.bounce_offsets,
        tx_positions=frame.tx_positions,
        rx_positions=frame.rx_positions,
        tx_orientations=frame.tx_orientations,
        rx_orientations=frame.rx_orientations,
        tx_names=frame.tx_names,
        rx_names=frame.rx_names,
        bounce_xyz_m=frame.bounce_xyz_m,
        interactions=frame.interactions,
        material_ids=frame.material_ids,
        material_names=frame.material_names,
        material_itu_types=frame.material_itu_types,
        delays_ns=frame.delays_ns,
        metric_valid_bits=frame.metric_valid_bits,
    )


def test_versions_are_independent() -> None:
    assert FRAME_FORMAT_VERSION == 2
    assert MPC_HDF5_SCHEMA_VERSION == 2
    assert MPC_HDF5_LAYOUT == "packed_ragged_v2"
    assert MPC_FRAME_MANIFEST_VERSION == 2
    # This identifies the unchanged physical compact-array vocabulary in HDF5.
    assert PACKED_MPC_FRAME_VERSION == 1


def test_standard_frame_is_frozen_complete_and_not_a_mapping() -> None:
    frame = _minimal_standard_frame()

    assert not isinstance(frame, Mapping)
    assert frame.num_tx == 1
    assert frame.num_rx == 3
    assert frame.num_pairs == 3
    assert frame.num_paths == 3
    assert frame.num_bounces == 3
    assert frame.num_targets == 0
    with pytest.raises(FrozenInstanceError):
        frame.frame_index = 8  # type: ignore[misc]


def test_metric_request_is_immutable_and_adds_dependencies() -> None:
    request = FrameReadRequest(metrics=frozenset({PathMetric.DELAY_NS, "path_loss_db"}))

    assert request.metrics == frozenset({PathMetric.DELAY_NS, PathMetric.PATH_LOSS_DB})
    assert request.components == frozenset(
        {FrameComponent.PATH_METRICS, FrameComponent.PATH_TOPOLOGY}
    )
    with pytest.raises(FrozenInstanceError):
        request.metrics = frozenset()  # type: ignore[misc]


def test_geometry_request_adds_topology_and_device_endpoints() -> None:
    request = FrameReadRequest(components=frozenset({FrameComponent.PATH_GEOMETRY}))

    assert request.components == frozenset(
        {
            FrameComponent.DEVICES,
            FrameComponent.PATH_TOPOLOGY,
            FrameComponent.PATH_BOUNCE_TOPOLOGY,
            FrameComponent.PATH_GEOMETRY,
        }
    )


def test_path_metrics_component_without_selection_means_all_metrics() -> None:
    request = FrameReadRequest(components=frozenset({FrameComponent.PATH_METRICS}))

    assert request.metrics == frozenset(PathMetric)


def test_sensing_selection_and_union_preserve_all_products_semantics() -> None:
    selected = FrameReadRequest(sensing_products=frozenset({"range_doppler"}))
    all_products = FrameReadRequest(components=frozenset({FrameComponent.SENSING}))

    assert selected.includes_component(FrameComponent.SENSING)
    assert not selected.all_sensing_products
    assert all_products.all_sensing_products
    assert selected.union(all_products).all_sensing_products


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"components": frozenset({"not-a-component"})}, "Unknown frame component"),
        ({"metrics": frozenset({"not-a-metric"})}, "Unknown path metric"),
        ({"sensing_products": frozenset({""})}, "non-empty strings"),
    ],
)
def test_request_rejects_unknown_or_empty_selectors(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        FrameReadRequest(**kwargs)


def test_standard_frame_preserves_empty_pairs_los_and_unknown_interactions() -> None:
    frame = _minimal_standard_frame()

    np.testing.assert_array_equal(np.diff(frame.pair_path_offsets), np.asarray([1, 0, 2]))
    np.testing.assert_array_equal(np.diff(frame.bounce_offsets), np.asarray([0, 1, 2]))
    np.testing.assert_array_equal(
        frame.metric_is_valid(PathMetric.DELAY_NS),
        np.asarray([True, True, False]),
    )
    assert frame.interactions[-1] == 99


@pytest.mark.parametrize(
    "changes, message",
    [
        (
            {"pair_path_offsets": np.asarray([1, 2, 2, 3], dtype=np.int64)},
            "must start at zero",
        ),
        (
            {"pair_path_offsets": np.asarray([0, 3, 2, 3], dtype=np.int64)},
            "monotonically non-decreasing",
        ),
        (
            {"bounce_offsets": np.asarray([0, 1, 2], dtype=np.int64)},
            "path count plus one",
        ),
        (
            {"delays_ns": np.asarray([1.0], dtype=np.float32)},
            "does not match the path count",
        ),
        (
            {"delays_ns": np.asarray([0.0, 3.0, 8.0], dtype=np.float32)},
            "Invalid delay_ns entries must contain NaN",
        ),
        (
            {"interactions": np.asarray([1, 2, 99], dtype=np.int16)},
            "dtype uint8",
        ),
        (
            {"bounce_xyz_m": np.zeros((3, 3), dtype=np.float64)},
            "dtype float32",
        ),
    ],
)
def test_standard_frame_rejects_malformed_compact_data(changes, message) -> None:
    source = _minimal_standard_frame()
    values = {
        data_field.name: getattr(source, data_field.name)
        for data_field in fields(source)
        if data_field.name != "version"
    }
    values.update(changes)

    with pytest.raises(ValueError, match=message):
        StandardMPCFrame(**values)


def test_projection_inventory_and_satisfaction_are_explicit() -> None:
    frame = _project_geometry_and_delay(_minimal_standard_frame())
    request = FrameReadRequest(
        components=frozenset(
            {
                FrameComponent.DEVICES,
                FrameComponent.PATH_GEOMETRY,
                FrameComponent.PATH_INTERACTIONS,
                FrameComponent.PATH_MATERIALS,
            }
        ),
        metrics=frozenset({PathMetric.DELAY_NS}),
    )
    projection = FrameProjection.from_request(frame, request)

    assert projection.satisfies(FrameReadRequest(metrics=frozenset({PathMetric.DELAY_NS})))
    assert not projection.satisfies(FrameReadRequest(metrics=frozenset({PathMetric.PATH_LOSS_DB})))
    assert not projection.satisfies(
        FrameReadRequest(components=frozenset({FrameComponent.TARGETS}))
    )


def test_metric_only_projection_does_not_require_geometry_or_devices() -> None:
    delay_bit = PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS]
    frame = ProjectedMPCFrame(
        frame_index=3,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        pair_path_offsets=np.asarray([0, 1], dtype=np.int64),
        delays_ns=np.asarray([2.5], dtype=np.float32),
        metric_valid_bits=np.asarray([delay_bit], dtype=np.uint8),
    )
    projection = FrameProjection.from_request(
        frame,
        FrameReadRequest(metrics=frozenset({PathMetric.DELAY_NS})),
    )

    assert projection.frame.bounce_xyz_m is None
    assert projection.frame.tx_positions is None


def test_device_projection_requires_stable_names() -> None:
    frame = ProjectedMPCFrame(
        frame_index=4,
        tx_positions=np.zeros((1, 3), dtype=np.float64),
        rx_positions=np.ones((1, 3), dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
    )

    with pytest.raises(ValueError, match="requires frame.tx_names"):
        FrameProjection.from_request(
            frame,
            FrameReadRequest(components=frozenset({FrameComponent.DEVICES})),
        )


def test_projection_rejects_loaded_component_without_required_arrays() -> None:
    with pytest.raises(ValueError, match="requires frame"):
        FrameProjection.from_request(
            ProjectedMPCFrame(frame_index=0),
            FrameReadRequest(components=frozenset({FrameComponent.PATH_GEOMETRY})),
        )


def test_projection_requires_its_distinct_partial_type() -> None:
    with pytest.raises(TypeError, match="ProjectedMPCFrame"):
        FrameProjection(frame=_minimal_standard_frame())  # type: ignore[arg-type]
