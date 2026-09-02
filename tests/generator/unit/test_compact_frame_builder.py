from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from generator.io.frames.builder import process_cached_frame_data, process_frame_data
from generator.io.frames.conversion import standard_mpc_frame_from_raw


class _ArrayValue:
    def __init__(self, value: Any) -> None:
        self._value = np.asarray(value)

    def numpy(self) -> np.ndarray:
        return self._value


class _Vector3:
    def __init__(self, x: float, y: float, z: float) -> None:
        self.x = x
        self.y = y
        self.z = z

    def __array__(self, dtype: Any = None, copy: bool | None = None) -> np.ndarray:
        del copy
        return np.asarray([self.x, self.y, self.z], dtype=dtype)


class _MustNotExpand:
    def __str__(self) -> str:
        raise AssertionError("filtered material metadata was expanded")


def _device(name: str, xyz: tuple[float, float, float]) -> SimpleNamespace:
    return SimpleNamespace(
        name=name,
        position=_Vector3(*xyz),
        orientation=_Vector3(0.0, 0.0, 0.0),
    )


def _paths() -> SimpleNamespace:
    path_loss_db = np.asarray([10.0, 30.0, 20.0], dtype=np.float32)
    amplitudes = np.power(10.0, -path_loss_db / 20.0).astype(np.complex64)
    vertices = np.zeros((1, 1, 1, 3, 3), dtype=np.float64)
    vertices[0, 0, 0, :, 0] = [1.0, 2.0, 3.0]
    interactions = np.ones((1, 1, 1, 3), dtype=np.int32)
    angles = np.zeros((1, 1, 3), dtype=np.float32)
    return SimpleNamespace(
        valid=_ArrayValue(np.ones((1, 1, 3), dtype=bool)),
        vertices=_ArrayValue(vertices),
        interactions=_ArrayValue(interactions),
        tau=_ArrayValue(np.asarray([[[1e-9, 2e-9, 3e-9]]], dtype=np.float64)),
        a=_ArrayValue(amplitudes.reshape((1, 1, 3))),
        theta_r=_ArrayValue(angles),
        phi_r=_ArrayValue(angles),
        theta_t=_ArrayValue(angles),
        phi_t=_ArrayValue(angles),
    )


def test_sionna_filter_selects_before_geometry_and_material_expansion() -> None:
    frame = process_frame_data(
        4,
        [_device("tx", (0.0, 0.0, 0.0))],
        [_device("rx", (10.0, 0.0, 0.0))],
        _paths(),
        [],
        [],
        SimpleNamespace(export_path_metrics=True),
        material_mapping={
            (0, 0): {
                0: [{"name": "wall", "itu_type": "itu_concrete"}],
                1: [{"name": _MustNotExpand(), "itu_type": "unused"}],
                2: [{"name": "glass", "itu_type": "itu_glass"}],
            }
        },
        path_filter_config={"max_paths_per_pair": 2, "log_filtering_stats": False},
    )

    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 2])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 1, 2])
    np.testing.assert_allclose(frame.bounce_xyz_m[:, 0], [1.0, 3.0])
    np.testing.assert_allclose(frame.path_loss_db, [10.0, 20.0], rtol=1e-5)
    np.testing.assert_allclose(frame.delays_ns, [1.0, 3.0], rtol=1e-5)
    np.testing.assert_array_equal(frame.metric_valid_bits, [0x3F, 0x3F])
    assert frame.material_names == ("", "wall", "glass")
    assert frame.material_itu_types == ("", "itu_concrete", "itu_glass")
    assert frame.bounce_xyz_m.dtype == np.dtype(np.float32)
    assert frame.tx_names == ("tx",)
    assert frame.rx_names == ("rx",)


def test_cached_frame_is_complete_compact_zero_path_frame() -> None:
    frame = process_cached_frame_data(
        7,
        [_device("tx", (0.0, 0.0, 0.0))],
        [_device("rx", (10.0, 0.0, 0.0))],
        [],
        [],
    )

    assert frame.frame_index == 7
    assert frame.num_pairs == 1
    assert frame.num_paths == 0
    assert frame.num_bounces == 0
    assert frame.tx_names == ("tx",)
    assert frame.rx_names == ("rx",)
    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 0])
    np.testing.assert_array_equal(frame.bounce_offsets, [0])
    assert frame.material_names == ("",)


def test_multi_device_pairs_preserve_rx_major_path_and_bounce_order() -> None:
    valid = np.zeros((2, 2, 2), dtype=bool)
    valid[0, 0, 0] = True
    valid[1, 0, 0] = True
    valid[1, 1, 1] = True
    vertices = np.zeros((2, 2, 2, 2, 3), dtype=np.float64)
    vertices[:, 1, 0, 0, 0] = [10.0, 11.0]
    vertices[0, 1, 1, 1, 0] = 20.0
    interactions = np.zeros((2, 2, 2, 2), dtype=np.int32)
    interactions[:, 1, 0, 0] = [1, 2]
    interactions[0, 1, 1, 1] = 4
    paths = SimpleNamespace(
        valid=_ArrayValue(valid),
        vertices=_ArrayValue(vertices),
        interactions=_ArrayValue(interactions),
    )

    frame = process_frame_data(
        0,
        [_device("tx0", (0.0, 0.0, 0.0)), _device("tx1", (1.0, 0.0, 0.0))],
        [_device("rx0", (0.0, 1.0, 0.0)), _device("rx1", (1.0, 1.0, 0.0))],
        paths,
        [],
        [],
        SimpleNamespace(export_path_metrics=False),
    )

    np.testing.assert_array_equal(frame.tx_rx_pairs, [[0, 0], [1, 0], [0, 1], [1, 1]])
    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 1, 1, 2, 3])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 0, 2, 3])
    np.testing.assert_array_equal(frame.interactions, [1, 2, 4])
    np.testing.assert_allclose(frame.bounce_xyz_m[:, 0], [10.0, 11.0, 20.0])
    assert np.all(np.isnan(frame.delays_ns))
    np.testing.assert_array_equal(frame.metric_valid_bits, [0, 0, 0])


def test_single_device_collapsed_sionna_axes_keep_origin_bounces() -> None:
    vertices = np.zeros((2, 2, 3), dtype=np.float64)
    vertices[:, 0, :] = [[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]]
    paths = SimpleNamespace(
        valid=_ArrayValue([True, False]),
        vertices=_ArrayValue(vertices),
        interactions=_ArrayValue([[1, 0], [2, 0]]),
    )

    frame = process_frame_data(
        0,
        [_device("tx", (0.0, 0.0, 0.0))],
        [_device("rx", (1.0, 0.0, 0.0))],
        paths,
        [],
        [],
        SimpleNamespace(export_path_metrics=False),
    )

    np.testing.assert_array_equal(frame.pair_path_offsets, [0, 1])
    np.testing.assert_array_equal(frame.bounce_offsets, [0, 2])
    np.testing.assert_array_equal(frame.interactions, [1, 2])
    np.testing.assert_allclose(frame.bounce_xyz_m, [[0.0, 0.0, 0.0], [2.0, 3.0, 4.0]])


def test_multi_device_input_rejects_collapsed_pair_axes() -> None:
    paths = SimpleNamespace(
        valid=_ArrayValue([True]),
        vertices=_ArrayValue(np.zeros((1, 1, 3), dtype=np.float32)),
        interactions=_ArrayValue(np.ones((1, 1), dtype=np.int32)),
    )

    with pytest.raises(ValueError, match="Unexpected valid shape"):
        process_frame_data(
            0,
            [_device("tx0", (0.0, 0.0, 0.0)), _device("tx1", (1.0, 0.0, 0.0))],
            [_device("rx", (0.0, 1.0, 0.0))],
            paths,
            [],
            [],
            SimpleNamespace(export_path_metrics=False),
        )


def test_raw_conversion_binds_snapshots_and_provenance_before_validation() -> None:
    raw = {
        "tx_list": [_device("tx", (9.0, 9.0, 9.0))],
        "rx_list": [_device("rx", (8.0, 8.0, 8.0))],
        "paths": _paths(),
        "target_objects": [],
        "target_managers": [],
        "tx_positions_snapshot": np.asarray([[1.0, 2.0, 3.0]], dtype=np.float64),
        "rx_positions_snapshot": np.asarray([[4.0, 5.0, 6.0]], dtype=np.float64),
    }

    frame = standard_mpc_frame_from_raw(
        raw,
        11,
        source_provider="generator_file",
        simulation_config=SimpleNamespace(export_path_metrics=False),
        timestamp=12.5,
    )

    np.testing.assert_allclose(frame.tx_positions, [[1.0, 2.0, 3.0]])
    np.testing.assert_allclose(frame.rx_positions, [[4.0, 5.0, 6.0]])
    assert frame.frame_index == 11
    assert frame.timestamp_s == 12.5
    assert frame.provenance == {
        "provider": "generator_file",
        "frame_idx": 11,
        "source_rt_frame_idx": 11,
        "timestamp": 12.5,
    }


def test_topology_only_conversion_has_no_raytracing_source() -> None:
    raw = {
        "tx_list": [_device("tx", (1.0, 2.0, 3.0))],
        "rx_list": [_device("rx", (4.0, 5.0, 6.0))],
        "target_objects": [],
        "target_managers": [],
        "_coherent_cached_frame": True,
    }

    frame = standard_mpc_frame_from_raw(
        raw,
        0,
        source_provider="generator_file",
    )

    assert frame.provenance == {"provider": "generator_file", "frame_idx": 0}
