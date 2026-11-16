"""Persistent plot models retain data, identities and inspection behavior."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from shared.sensing.signal_processing import RangeDopplerResult, RangeProfileResult
from shared.sensing.stage_inspection import (
    DETECTION_STAGE_KINDS,
    CellSelection,
    DetectionLink,
    DetectionStageBundle,
    DetectionStageEvent,
    DetectionStageProduct,
    InspectionDisposition,
    LineageNode,
    LineageTrace,
    ProductTimeSpan,
)
from visualizer.src.sensing import prepared_results as module
from visualizer.src.sensing.prepared_angle import PreparedAngle
from visualizer.src.sensing.prepared_results import load_prepared, save_prepared
from visualizer.src.sensing.processing_inspection import ProcessingInspection
from visualizer.src.sensing.recording_cache import RecordingCache


@pytest.fixture
def cache(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    return RecordingCache((manifest,), tmp_path / "display-cache")


def test_common_maps_share_array_identity_and_preserve_metadata(cache):
    ranges = np.arange(4, dtype=np.float64)
    doppler = np.arange(3, dtype=np.float64)
    power = np.arange(12, dtype=np.float32).reshape(3, 4)
    rp = RangeProfileResult(power[0], ranges, None, metadata={"variant": "primary"})
    rd = RangeDopplerResult(power, ranges, doppler, None, metadata={"variant": "primary"})
    sensing = {"power": power, "range_axis_m": ranges, "detections": [{"range_bin": 3}]}
    value = {"sensing": sensing, "rp": rp, "rd": rd, "evaluation": {"total_tp": np.int64(2)}}
    assert save_prepared(cache, "common", (127,), value)
    result = load_prepared(RecordingCache(cache.paths, cache.root), "common", (127,))
    assert isinstance(result["rp"], RangeProfileResult)
    assert isinstance(result["rd"], RangeDopplerResult)
    assert result["evaluation"] == {"total_tp": 2}
    assert result["rp"].range_axis_m is result["rd"].range_axis_m
    assert result["rd"].power_linear is result["sensing"]["power"]
    assert result["rp"].metadata == rp.metadata
    assert result["sensing"]["detections"] == sensing["detections"]
    np.testing.assert_array_equal(result["rd"].power_linear, power)
    assert not result["rd"].power_linear.flags.writeable


def test_full_immutable_array_views_share_storage_and_retain_shape_order(cache):
    data = np.arange(24, dtype=np.float32).tobytes()
    cube = np.frombuffer(data, dtype=np.float32).reshape(2, 3, 4)
    flat_map = cube.reshape(6, 4)
    fortran = np.ndarray((4, 6), dtype=np.float32, buffer=data, order="F")
    value = {"cube": cube, "map": flat_map, "repeated_map": flat_map, "fortran": fortran}
    assert save_prepared(cache, "views", (), value)
    _, arrays = cache.load_arrays("views", (module.PREPARATION_VERSION,))
    assert len(arrays) == 1
    restored = load_prepared(cache, "views", ())
    for name, expected in value.items():
        np.testing.assert_array_equal(restored[name], expected)
        assert restored[name].shape == expected.shape
        assert not restored[name].flags.writeable
    assert restored["map"] is restored["repeated_map"]
    assert np.shares_memory(restored["cube"], restored["map"])
    assert np.shares_memory(restored["cube"], restored["fortran"])


def test_partial_offsets_and_mutable_buffer_views_are_not_coalesced(cache):
    source = np.frombuffer(np.arange(12, dtype=np.float32).tobytes(), dtype=np.float32)
    mutable = np.arange(12, dtype=np.float32)
    value = {
        "left": source[:6],
        "right": source[6:],
        "mutable": mutable,
        "mutable_view": mutable.reshape(3, 4),
    }
    assert save_prepared(cache, "views", (), value)
    _, arrays = cache.load_arrays("views", (module.PREPARATION_VERSION,))
    assert len(arrays) == 4
    restored = load_prepared(cache, "views", ())
    for name, expected in value.items():
        np.testing.assert_array_equal(restored[name], expected)


def test_angle_and_filtering_roundtrip(cache):
    ranges = np.arange(4.0)
    doppler = np.arange(3.0)
    angle = PreparedAngle(
        "angle-product",
        "primary",
        (2, 2, 3, 4),
        "monostatic_range",
        ranges,
        doppler,
        np.array([-30.0, 30.0]),
        np.ones((2, 4), dtype=np.float32),
        np.ones((2, 3), dtype=np.float32),
        np.zeros((3, 3)),
        np.ones((3, 3)),
        np.array([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32),
    )
    filtering = ProcessingInspection(
        "clutter",
        ranges=ranges,
        doppler=doppler,
        before=np.ones((3, 4)),
        after=np.zeros((3, 4)),
        identity=("rd-primary", "source-hash"),
        scale_scope=("run", "recipe"),
    )
    assert save_prepared(cache, "angle", (), angle)
    assert save_prepared(cache, "filtering", (), filtering)
    loaded_angle = load_prepared(cache, "angle", ())
    loaded_filtering = load_prepared(cache, "filtering", ())
    assert isinstance(loaded_angle, PreparedAngle)
    assert loaded_angle.shape == angle.shape
    np.testing.assert_array_equal(loaded_angle.spectrum_power, angle.spectrum_power)
    assert isinstance(loaded_filtering, ProcessingInspection)
    assert loaded_filtering.identity == filtering.identity
    np.testing.assert_array_equal(loaded_filtering.before, filtering.before)
    np.testing.assert_array_equal(loaded_filtering.after, filtering.after)


def test_detector_models_preserve_enums_lineage_and_selection_links(cache):
    event = DetectionStageEvent("clock", "acquisition", "run", "recipe", "pair", 1.0)
    time_span = ProductTimeSpan("clock", 0.0, 1.0, 1.0, "window_end")
    node = LineageNode("parent", "range_doppler", (), True, True)
    lineage = LineageTrace("input", (node,), ())
    link = DetectionLink("acquisition", "run", "pair", 0, 1, 2)
    product = DetectionStageProduct(
        "range_doppler_detection_input_power",
        "power_linear",
        InspectionDisposition.AVAILABLE,
        None,
        "input_by_pair",
        "input",
        "detector_input",
        np.arange(12.0).reshape(1, 3, 4),
        lineage,
    )
    bundle = DetectionStageBundle(
        InspectionDisposition.AVAILABLE,
        None,
        event,
        (
            product,
            *(
                DetectionStageProduct(
                    kind, "power_linear", InspectionDisposition.UNAVAILABLE, "Not recorded"
                )
                for kind in DETECTION_STAGE_KINDS
                if kind != product.kind
            ),
        ),
        time_span,
        np.arange(3.0),
        np.arange(4.0),
        (link,),
    )
    assert save_prepared(cache, "detector", (), bundle)
    loaded = load_prepared(cache, "detector", ())
    assert loaded.available
    assert loaded.event == event
    assert loaded.time_span == time_span
    assert loaded.products[0].available
    assert loaded.trace_lineage(product.kind) == lineage
    assert loaded.detection_links[0].stable_key == link.stable_key
    assert loaded.product(product.kind).values[0, 1, 2] == 6
    assert loaded.inspect(CellSelection("pair", 1, 2)).detection_input_power == 6


def test_containers_and_tag_looking_dict_keys_are_unambiguous(cache):
    value = {
        "type": "array",
        "value": "0",
        "tuple": (1, "two"),
        "set": {1, 2},
        "frozen": frozenset(("primary",)),
        3: [True, None, float("inf")],
    }
    assert save_prepared(cache, "metadata", (), value)
    assert load_prepared(cache, "metadata", ()) == value


def test_unknown_python_objects_are_not_serialized(cache):
    @dataclass
    class Unregistered:
        value: int

    assert not save_prepared(cache, "unknown", (), Unregistered(1))
    assert not save_prepared(cache, "unknown", (), object())
    assert not list(cache.root.glob("*.npz"))


@pytest.mark.parametrize(
    "payload",
    [
        {"type": "os.system", "value": "anything"},
        {"type": "RangeProfileResult", "value": {"power_linear": None}},
        {"type": "array", "value": "missing"},
        {"type": "InspectionDisposition", "value": "unknown"},
        {"type": "dict", "value": [[1]]},
        {"type": "tuple", "value": "wrong"},
        ["untagged"],
    ],
)
def test_invalid_prepared_values_fall_back(cache, payload):
    assert cache.save_arrays("broken", {"payload": payload}, {}, (module.PREPARATION_VERSION,))
    assert load_prepared(cache, "broken", ()) is None


def test_preparation_version_and_no_cache_fall_back(cache, monkeypatch):
    assert save_prepared(cache, "value", (), {"result": 1})
    monkeypatch.setattr(module, "PREPARATION_VERSION", module.PREPARATION_VERSION + 1)
    assert load_prepared(cache, "value", ()) is None
    assert load_prepared(None, "value", ()) is None
    assert not save_prepared(None, "value", (), {"result": 1})
