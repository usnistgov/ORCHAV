"""Provider fallback and in-memory canonical-frame projection tests."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import shared.frames.providers as providers_module
from shared.frames.adapters import project_standard_mpc_frame
from shared.frames.base import FormatHandler
from shared.frames.contracts import (
    PATH_METRIC_VALIDITY_BITS,
    FrameComponent,
    FrameReadRequest,
    PathMetric,
)
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed import FrameProjection, ProjectedMPCFrame
from shared.frames.provider_base import DataProvider
from shared.frames.providers import FileProvider, Hdf5Provider
from shared.frames.types import StandardMPCFrame


def _standard_frame() -> StandardMPCFrame:
    return standard_mpc_frame_from_pair_data(
        frame_index=7,
        timestamp_s=1.25,
        tx_rx_pairs=np.asarray([[1, 0], [0, 1]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0], [10.0, 0.0, 2.0]], dtype=np.float64),
        rx_positions=np.asarray([[0.0, 10.0, 1.0], [10.0, 10.0, 2.0]], dtype=np.float64),
        tx_orientations=np.zeros((2, 3), dtype=np.float64),
        rx_orientations=np.ones((2, 3), dtype=np.float64),
        tx_names=("alpha", "beta"),
        rx_names=("gamma", "delta"),
        vertices_by_pair=[
            np.asarray(
                [
                    [[np.nan, np.nan, np.nan]],
                    [[4.0, 5.0, 6.0]],
                ],
                dtype=np.float32,
            ),
            np.empty((0, 0, 3), dtype=np.float32),
        ],
        interactions_by_pair=[
            np.asarray([[-1], [37]], dtype=np.int32),
            np.empty((0, 0), dtype=np.int32),
        ],
        path_lengths_by_pair=[
            np.asarray([0, 1], dtype=np.int64),
            np.empty((0,), dtype=np.int64),
        ],
        material_names_by_pair=[
            np.asarray([[""], ["custom-metal"]], dtype=object),
            np.empty((0, 0), dtype=object),
        ],
        material_itu_types_by_pair=[
            np.asarray([[""], ["itu_metal"]], dtype=object),
            np.empty((0, 0), dtype=object),
        ],
        metrics_by_pair={
            "delays_ns": [
                np.asarray([0.0, 12.5], dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            ],
            "path_loss_db": [
                np.asarray([np.nan, 0.0], dtype=np.float32),
                np.empty((0,), dtype=np.float32),
            ],
        },
        target_positions_m=np.asarray([[8.0, 9.0, 1.5]], dtype=np.float64),
        targets_metadata=({"name": "walker", "speed_mps": 1.2},),
        beamforming={"method": "mvdr", "enabled": True},
        sensing={
            "config": {"carrier_hz": 60e9},
            "rd_map": np.arange(6, dtype=np.float32).reshape(2, 3),
            "cir": np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64),
        },
        provenance={"provider": "unit-test", "frame_idx": 7},
        recomputed_from_stored_positions=True,
    )


def test_standard_frame_is_already_compact_and_keeps_valid_zero_metrics() -> None:
    frame = _standard_frame()

    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 2, 2])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 0, 1])
    np.testing.assert_array_equal(frame.interactions, np.asarray([37], dtype=np.uint8))
    np.testing.assert_allclose(frame.bounce_xyz_m, [[4.0, 5.0, 6.0]])
    assert frame.material_names == ("", "custom-metal")
    assert frame.material_itu_types == ("", "itu_metal")
    np.testing.assert_array_equal(frame.material_ids, [1])

    delay_bit = PATH_METRIC_VALIDITY_BITS[PathMetric.DELAY_NS]
    loss_bit = PATH_METRIC_VALIDITY_BITS[PathMetric.PATH_LOSS_DB]
    assert int(frame.metric_valid_bits[0]) & delay_bit
    assert int(frame.metric_valid_bits[0]) & loss_bit == 0
    assert int(frame.metric_valid_bits[1]) & delay_bit
    assert int(frame.metric_valid_bits[1]) & loss_bit
    assert frame.delays_ns[0] == 0.0
    assert frame.path_loss_db[1] == 0.0
    assert np.isnan(frame.path_loss_db[0])


def test_projection_has_exact_metric_inventory_and_borrows_arrays() -> None:
    frame = _standard_frame()
    request = FrameReadRequest.for_metrics([PathMetric.DELAY_NS])

    projection = project_standard_mpc_frame(frame, request)

    assert projection.loaded_components == frozenset(
        {FrameComponent.PATH_TOPOLOGY, FrameComponent.PATH_METRICS}
    )
    assert projection.loaded_path_metrics == frozenset({PathMetric.DELAY_NS})
    assert projection.frame.tx_rx_pairs is frame.tx_rx_pairs
    assert projection.frame.pair_path_offsets is frame.pair_path_offsets
    assert projection.frame.delays_ns is frame.delays_ns
    assert projection.frame.path_loss_db is None
    assert projection.frame.bounce_xyz_m is None
    assert projection.frame.tx_positions is None
    assert projection.satisfies(request)


def test_projection_selects_sensing_products_and_keeps_metadata() -> None:
    frame = _standard_frame()
    request = FrameReadRequest(
        components=frozenset({FrameComponent.SENSING}),
        sensing_products=frozenset({"rd_map", "missing"}),
    )

    projection = project_standard_mpc_frame(frame, request)

    assert projection.loaded_sensing_products == frozenset({"rd_map", "missing"})
    assert projection.all_sensing_products_loaded is False
    assert projection.frame.sensing is not None
    assert projection.frame.sensing["config"] == {"carrier_hz": 60e9}
    np.testing.assert_array_equal(
        projection.frame.sensing["rd_map"],
        np.arange(6, dtype=np.float32).reshape(2, 3),
    )
    assert "cir" not in projection.frame.sensing

    full_sensing = project_standard_mpc_frame(
        frame,
        FrameReadRequest(components=frozenset({FrameComponent.SENSING})),
    )
    assert full_sensing.all_sensing_products_loaded is True
    assert full_sensing.loaded_sensing_products == frozenset({"rd_map", "cir"})


def test_projection_records_requested_sensing_products_as_absent() -> None:
    frame = replace(_standard_frame(), sensing=None)
    request = FrameReadRequest(sensing_products=frozenset({"detections"}))

    projection = project_standard_mpc_frame(frame, request)

    assert projection.frame.sensing is None
    assert projection.loaded_sensing_products == frozenset({"detections"})
    assert projection.all_sensing_products_loaded is False
    assert projection.satisfies(request)


def test_projection_keeps_target_and_provenance_components_separate() -> None:
    frame = _standard_frame()

    targets = project_standard_mpc_frame(
        frame,
        FrameReadRequest(components=frozenset({FrameComponent.TARGETS})),
    )
    provenance = project_standard_mpc_frame(
        frame,
        FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE})),
    )

    assert targets.frame.target_positions_m is frame.target_positions_m
    assert targets.frame.targets_metadata is frame.targets_metadata
    assert targets.frame.provenance is None
    assert provenance.frame.target_positions_m is None
    assert provenance.frame.provenance == frame.provenance
    assert provenance.frame.timestamp_s == 1.25


class _MemoryProvider(DataProvider):
    def __init__(self, frame: StandardMPCFrame) -> None:
        self.frame = frame
        self.load_count = 0

    def list_frames(self) -> list[int]:
        return [7]

    def has_frame(self, step: int) -> bool:
        return step == 7

    def load_frame(self, step: int) -> StandardMPCFrame:
        assert step == 7
        self.load_count += 1
        return self.frame


def test_non_file_provider_uses_complete_frame_projection_fallback() -> None:
    provider = _MemoryProvider(_standard_frame())
    request = FrameReadRequest.for_metrics([PathMetric.PATH_LOSS_DB])

    projection = provider.load_frame_projection(7, request)
    batch = list(provider.iter_frame_projections([7], request))

    assert provider.load_count == 2
    assert projection.frame.frame_index == 7
    assert projection.frame.path_loss_db is not None
    assert projection.frame.delays_ns is None
    assert [item.frame.frame_index for item in batch] == [7]


def test_device_projection_requires_stable_device_names() -> None:
    frame = _standard_frame()
    partial = ProjectedMPCFrame(
        frame_index=frame.frame_index,
        tx_positions=frame.tx_positions,
        rx_positions=frame.rx_positions,
        tx_orientations=frame.tx_orientations,
        rx_orientations=frame.rx_orientations,
    )

    with pytest.raises(ValueError, match=r"devices.*tx_names"):
        FrameProjection(
            frame=partial,
            loaded_components=frozenset({FrameComponent.DEVICES}),
        )


def test_loaded_optional_component_can_be_absent() -> None:
    projection = FrameProjection(
        frame=ProjectedMPCFrame(frame_index=3, provenance=None),
        loaded_components=frozenset({FrameComponent.PROVENANCE}),
    )

    assert projection.frame.provenance is None


class _FallbackHandler(FormatHandler):
    def __init__(self, source: Path, frame: StandardMPCFrame) -> None:
        super().__init__(source)
        self.frame = frame

    def can_handle(self) -> bool:
        return True

    def list_frames(self) -> list[int]:
        return [7]

    def has_frame(self, step: int) -> bool:
        return step == 7

    def load_frame(self, step: int) -> StandardMPCFrame:
        assert step == 7
        return self.frame

    @property
    def generation_id(self) -> str | None:
        return "generation-123"

    @property
    def frame_set_id(self) -> str | None:
        return "frame-set-456"


def test_file_provider_fallback_exposes_frame_set_identity(tmp_path: Path) -> None:
    provider = FileProvider(tmp_path, _FallbackHandler(tmp_path, _standard_frame()))

    projection = provider.load_frame_projection(
        7,
        FrameReadRequest(components=frozenset({FrameComponent.TARGETS})),
    )

    assert projection.frame.targets_metadata is not None
    assert projection.frame.targets_metadata[0]["name"] == "walker"
    assert provider.info.generation_id == "generation-123"
    assert provider.info.frame_set_id == "frame-set-456"


class _SelectiveHandler(FormatHandler):
    def __init__(self, source: Path) -> None:
        super().__init__(source)
        self.requests: list[tuple[int, FrameReadRequest]] = []

    def can_handle(self) -> bool:
        return True

    def list_frames(self) -> list[int]:
        return [3]

    def has_frame(self, step: int) -> bool:
        return step == 3

    def load_frame(self, step: int) -> StandardMPCFrame:
        raise AssertionError("A selective projection must not load a complete frame")

    def load_frame_projection(
        self,
        step: int,
        request: FrameReadRequest,
    ) -> FrameProjection:
        self.requests.append((step, request))
        return FrameProjection.from_request(
            ProjectedMPCFrame(frame_index=step),
            request,
        )


def test_hdf5_provider_delegates_projection_to_its_handler(
    tmp_path: Path,
    monkeypatch,
) -> None:
    handler = _SelectiveHandler(tmp_path)
    monkeypatch.setattr(
        providers_module,
        "HDF5FormatHandler",
        lambda source, *, frames_subdir: handler,
    )
    provider = Hdf5Provider(tmp_path)
    request = FrameReadRequest()

    projection = provider.load_frame_projection(3, request)
    batch = list(provider.iter_frame_projections([3], request))

    assert projection.frame.frame_index == 3
    assert [item.frame.frame_index for item in batch] == [3]
    assert handler.requests == [(3, request), (3, request)]
