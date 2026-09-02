import logging
import sys
import types

import numpy as np
import pytest


# Create a more complete mock mitsuba module
class Point3f:
    def __init__(self, x, y, z):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)


def variant():
    return "cuda_ad"


# Create a mock mitsuba module
mi_mod = types.ModuleType("mitsuba")
mi_mod.Point3f = Point3f
mi_mod.variant = variant

# Force replace mitsuba in sys.modules
sys.modules["mitsuba"] = mi_mod

# Also mock drjit to avoid CUDA dependencies
if "drjit" not in sys.modules:
    drjit_mod = types.ModuleType("drjit")
    drjit_cuda = types.ModuleType("drjit.cuda")
    drjit_cuda_ad = types.ModuleType("drjit.cuda.ad")

    class Float:
        def __init__(self, value):
            self._value = float(value)

        def __float__(self):
            return self._value

        def __str__(self):
            return str(self._value)

    drjit_cuda_ad.Float = Float
    drjit_cuda.ad = drjit_cuda_ad
    drjit_mod.cuda = drjit_cuda
    sys.modules["drjit"] = drjit_mod
    sys.modules["drjit.cuda"] = drjit_cuda
    sys.modules["drjit.cuda.ad"] = drjit_cuda_ad

# Mock sionna to avoid heavy dependencies
if "sionna" not in sys.modules:
    sionna_mod = types.ModuleType("sionna")
    sionna_rt = types.ModuleType("sionna.rt")

    # Mock the classes that might be imported
    class MockClass:
        def __init__(self, *args, **kwargs):
            pass

    sionna_rt.load_scene = MockClass
    sionna_rt.Scene = MockClass
    sionna_rt.Transmitter = MockClass
    sionna_rt.Receiver = MockClass
    sionna_rt.PlanarArray = MockClass
    sionna_rt.PathSolver = MockClass
    sionna_rt.Camera = MockClass
    sionna_rt.RadioMaterial = MockClass

    sionna_mod.rt = sionna_rt
    sys.modules["sionna"] = sionna_mod
    sys.modules["sionna.rt"] = sionna_rt

from generator.core.mobility import LinearMobility, MobilityPattern
from generator.core.mobility.base import _numeric_pair, _numeric_triple


class _TensorScalar:
    def __init__(self, value):
        self._value = float(value)

    def numpy(self):
        return np.array([self._value])


class _TensorPoint:
    def __init__(self, x, y, z):
        self.x = _TensorScalar(x)
        self.y = _TensorScalar(y)
        self.z = _TensorScalar(z)


class _TensorPointMobility(MobilityPattern):
    def get_positions(self, start_pos, scene_steps, scene_duration):
        del start_pos, scene_duration
        return [_TensorPoint(i, i + 1, i + 2) for i in range(scene_steps)]


def test_linear_positions_basic():
    lm = LinearMobility(start_pos=(0.0, 0.0, 0.0), end_pos=(2.0, 0.0, 0.0))
    lm.prepare(3, 1.0, start_pos=(0, 0, 0))
    pos = lm.prepared_positions()
    assert len(pos) == 3
    assert pos[0] == (0.0, 0.0, 0.0)
    assert pos[1] == (1.0, 0.0, 0.0)
    assert pos[2] == (2.0, 0.0, 0.0)


def test_linear_target_speed_returned():
    lm = LinearMobility(start_pos=(0.0, 0.0, 0.0), end_pos=(10.0, 0.0, 0.0), target_speed_mps=5.0)
    spd = lm.get_speed(start_pos=(0, 0, 0), scene_steps=5, scene_duration=2.0)
    assert spd == 5.0


def test_get_position_warns_when_step_is_clamped(caplog):
    lm = LinearMobility(start_pos=(0.0, 0.0, 0.0), end_pos=(2.0, 0.0, 0.0))
    lm.prepare(3, 1.0, start_pos=(0, 0, 0))

    orchav_logger = logging.getLogger("orchav")
    previous_propagate = orchav_logger.propagate
    orchav_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="orchav.generator.core.mobility.base"):
            pos = lm.get_position(99)
    finally:
        orchav_logger.propagate = previous_propagate

    assert pos == (2.0, 0.0, 0.0)
    assert "outside prepared range [0, 2]" in caplog.text


def test_get_position_returns_cached_position_tuple():
    lm = LinearMobility(start_pos=(0.0, 0.0, 0.0), end_pos=(2.0, 0.0, 0.0))
    lm.prepare(3, 1.0, start_pos=(0, 0, 0))

    assert lm.get_position(1) is lm.prepared_positions()[1]


def test_numeric_helpers_reject_scalar_strings():
    assert _numeric_pair(["1", 2], "bounds") == (1.0, 2.0)
    assert _numeric_triple([1, "2", 3.0], "axis") == (1.0, 2.0, 3.0)

    with pytest.raises(ValueError, match="must not be a string"):
        _numeric_pair("12", "bounds")
    with pytest.raises(ValueError, match="must not be a string"):
        _numeric_triple("123", "axis")


def test_prepare_converts_tensor_like_point_coordinates():
    mobility = _TensorPointMobility()
    mobility.prepare(2, 1.0)

    positions = mobility.prepared_positions()

    assert positions[0] == (0.0, 1.0, 2.0)
    assert positions[1] == (1.0, 2.0, 3.0)


def test_prepare_rejects_zero_steps():
    lm = LinearMobility(start_pos=(0.0, 0.0, 0.0), end_pos=(1.0, 0.0, 0.0))

    with pytest.raises(ValueError, match="requires at least one scene step"):
        lm.prepare(0, 1.0, start_pos=(0.0, 0.0, 0.0))
