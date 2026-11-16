"""Recorded decisions agree with the running tracker without changing estimates."""

import json
from dataclasses import replace

import numpy as np
import pytest

from shared.sensing.tracker import (
    CartesianKalmanTracker,
    JPDAFTracker,
    KalmanTracker,
    MofNTracker,
    TrackerConfig,
)
from shared.sensing.tracking_inspection import tracking_decisions


@pytest.mark.parametrize(
    "factory", [MofNTracker, KalmanTracker, CartesianKalmanTracker, JPDAFTracker]
)
def test_optional_decisions_preserve_outputs_and_explain_birth_coast_retirement(factory):
    cfg = TrackerConfig(m=2, max_missed_frames=1)
    quiet, recorded = factory(cfg), factory(replace(cfg, record_diagnostics=True))
    batches = [
        [{"range_m": 4.0, "velocity_m_s": 0.1, "x_m": 4.0, "y_m": 0.0}],
        [
            {"range_m": 3.99, "velocity_m_s": 0.1, "x_m": 3.99, "y_m": 0.0},
            {"range_m": 14.0, "velocity_m_s": 2.0, "x_m": 14.0, "y_m": 0.0},
        ],
        [],
        [],
    ]
    for index, detections in enumerate(batches):
        left, lm = quiet.filter_detections(detections, update_time_s=index * 0.1)
        right, rm = recorded.filter_detections(detections, update_time_s=index * 0.1)
        assert left == right
        assert "decisions" not in lm
        record = rm.pop("decisions")
        assert lm == rm
        assert record["update_time_s"] == index * 0.1
        json.dumps(record, allow_nan=False)
        if index == 0:
            assert record["birth_track_ids"] == [1]
            assert record["predictions"] == []
        elif index == 1:
            assert record["birth_track_ids"] == [2]
            assert [c["admissible"] for c in record["candidates"]] == [True, False]
            assert record["assignments"][0] == {"detection_index": 0, "track_id": 1}
            if factory is not MofNTracker:
                assert record["predictions"][0]["covariance"] is not None
        elif index == 2:
            assert all(s["status"] == "coasting" for s in record["tracks"])
            assert not any(s["emitted"] for s in record["tracks"])
        else:
            assert record["retired_track_ids"] == [1, 2]
            assert record["tracks"] == []


def test_legacy_and_unsupported_decision_records():
    assert tracking_decisions({}) == (None, {})
    with pytest.raises(ValueError, match="Unsupported"):
        tracking_decisions(
            {"detection_filter_metadata": {"tracker": {"decisions": {"schema_version": 99}}}}
        )


def test_jpda_records_probabilities_separately_from_reporting_assignment():
    tracker = JPDAFTracker(TrackerConfig(record_diagnostics=True))
    tracker.filter_detections([{"range_m": 4.0, "velocity_m_s": 0.2}], update_time_s=0.0)
    _, meta = tracker.filter_detections(
        [
            {"range_m": 3.98, "velocity_m_s": 0.2},
            {"range_m": 4.02, "velocity_m_s": 0.2},
        ],
        update_time_s=0.1,
    )
    record = meta["decisions"]
    assert len(record["assignments"]) == 1
    assert all(c["probability"] > 0 for c in record["candidates"])
    np.testing.assert_allclose(
        sum(c["probability"] for c in record["candidates"]) + record["missed_probabilities"]["1"], 1
    )


def test_legacy_positional_configuration_arguments_keep_their_meaning():
    from dataclasses import asdict, fields

    from generator.core.configuration.models import SensingConfig
    from shared.sensing.postprocessor import PostProcessingConfig

    for original, option in [
        (TrackerConfig(include_z=True), "record_diagnostics"),
        (PostProcessingConfig(tracker_type="kalman"), "include_tracking_diagnostics"),
        (SensingConfig(output_detections=False), "persist_tracking_diagnostics"),
    ]:
        legacy_arguments = [getattr(original, f.name) for f in fields(original) if not f.kw_only]
        restored = type(original)(*legacy_arguments)
        assert asdict(restored) == asdict(original)


def test_jpda_controls_preserve_existing_positional_configuration_indices():
    from dataclasses import fields

    from generator.core.configuration.models import SensingConfig
    from shared.sensing.postprocessor import PostProcessingConfig

    controls = {
        "jpda_max_joint_nodes",
        "jpda_birth_confirmation_frames",
        "jpda_birth_separation_chi2",
    }
    for cls, index, expected in [
        (SensingConfig, 40, "track_association_position_gate_m"),
        (PostProcessingConfig, 69, "m_of_n_m"),
    ]:
        positional = [f.name for f in fields(cls) if not f.kw_only]
        assert positional[index] == expected
        assert controls.isdisjoint(positional)
        configured = cls(
            jpda_max_joint_nodes=456,
            jpda_birth_confirmation_frames=4,
            jpda_birth_separation_chi2=0.25,
        )
        assert configured.jpda_max_joint_nodes == 456
        assert configured.jpda_birth_confirmation_frames == 4
        assert configured.jpda_birth_separation_chi2 == 0.25
