"""Tests for generator runtime utility helpers."""

from __future__ import annotations

import numpy as np
import pytest

from generator.core.utils import (
    angle_difference_deg,
    point_to_tuple,
    to_numpy,
    to_scalar,
    wrap_angle_deg,
    wrap_angles_deg,
)


class _TensorLike:
    def __init__(self, value: object) -> None:
        self._value = value

    def numpy(self) -> object:
        return self._value


class _PointLike:
    def __init__(self, x: object, y: object, z: object) -> None:
        self.x = x
        self.y = y
        self.z = z


def test_wrap_angle_deg_uses_half_open_minus_180_to_180_range() -> None:
    assert wrap_angle_deg(180.0) == pytest.approx(-180.0)
    assert wrap_angle_deg(270.0) == pytest.approx(-90.0)
    assert wrap_angle_deg(-270.0) == pytest.approx(90.0)
    assert wrap_angle_deg(540.0) == pytest.approx(-180.0)


def test_wrap_angles_deg_preserves_empty_arrays() -> None:
    empty = np.asarray([], dtype=np.float64)

    result = wrap_angles_deg(empty)

    assert result is empty


def test_wrap_angles_deg_vectorizes_angle_normalization() -> None:
    angles = np.asarray([0.0, 180.0, 270.0, -270.0, 450.0], dtype=np.float64)

    result = wrap_angles_deg(angles)

    np.testing.assert_allclose(result, [0.0, -180.0, -90.0, 90.0, 90.0])


def test_angle_difference_deg_returns_shortest_signed_delta() -> None:
    assert angle_difference_deg(10.0, 350.0) == pytest.approx(-20.0)
    assert angle_difference_deg(350.0, 10.0) == pytest.approx(20.0)
    assert angle_difference_deg(0.0, 180.0) == pytest.approx(-180.0)


def test_to_numpy_handles_tensor_like_and_nested_sequences() -> None:
    result = to_numpy([_TensorLike(np.asarray([1.0, 2.0])), (3.0, 4.0)])

    np.testing.assert_allclose(result, [1.0, 2.0, 3.0, 4.0])


def test_to_numpy_can_flatten_multidimensional_values() -> None:
    result = to_numpy(np.asarray([[1, 2], [3, 4]], dtype=np.int64), flatten=True)

    np.testing.assert_array_equal(result, [1, 2, 3, 4])


def test_to_scalar_extracts_first_value_from_tensor_like_array() -> None:
    value = _TensorLike(np.asarray([42.8], dtype=np.float64))

    assert to_scalar(value, dtype=int) == 42


def test_to_scalar_keeps_historical_zero_fallback_for_unreadable_values() -> None:
    assert to_scalar(object()) == pytest.approx(0.0)


def test_point_to_tuple_handles_supported_point_representations() -> None:
    assert point_to_tuple([1.0, 2.0, 3.0]) == (1.0, 2.0, 3.0)
    assert point_to_tuple((4.0, 5.0, 6.0)) == (4.0, 5.0, 6.0)
    assert point_to_tuple(np.asarray([1.0, 2.0, 3.0])) == (1.0, 2.0, 3.0)
    assert point_to_tuple(_PointLike(7.0, 8.0, 9.0)) == (7.0, 8.0, 9.0)
    assert point_to_tuple(_TensorLike(np.asarray([10.0, 20.0, 30.0]))) == (
        10.0,
        20.0,
        30.0,
    )


def test_point_to_tuple_raises_configured_exception_for_invalid_values() -> None:
    with pytest.raises(TypeError):
        point_to_tuple([1.0, 2.0], error_type=TypeError)
    with pytest.raises(TypeError):
        point_to_tuple([1.0, 2.0, 3.0, 4.0], error_type=TypeError)
    with pytest.raises(TypeError):
        point_to_tuple(None, error_type=TypeError)
    with pytest.raises(TypeError):
        point_to_tuple([object(), 2.0, 3.0], error_type=TypeError)
    with pytest.raises(TypeError):
        point_to_tuple(
            _TensorLike(np.asarray([object(), 2.0, 3.0], dtype=object)), error_type=TypeError
        )


def test_point_to_tuple_can_use_separate_attribute_converter() -> None:
    def safe_converter(value: object) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    result = point_to_tuple(
        _PointLike("bad", 2.0, 3.0),
        attribute_converter=safe_converter,
    )

    assert result == (0.0, 2.0, 3.0)
