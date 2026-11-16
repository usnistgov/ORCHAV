"""Attribution follows saved accepted report IDs, never proximity or track IDs."""

from copy import deepcopy

import numpy as np

from visualizer.src.sensing.fusion_comparison import FusionComparison, comparison_data
from visualizer.src.sensing.fusion_inspection import FusionSelection


class Result(dict):
    times = [0.0]

    def epoch(self, index, mode="fused"):
        return self["histories"][mode][index]


def example():
    return Result(
        sources={"west": {}, "east": {}},
        histories={
            "fused": [
                {
                    "tracks": [
                        dict(
                            track_id=1,
                            confirmed=True,
                            predicted_only=False,
                            position_m=[0.1, 0, 0],
                            sources=[dict(sensor_id="west", detection_id="accepted")],
                        )
                    ]
                }
            ]
        },
        observations={
            "west": [
                {
                    "detections": [
                        dict(detection_id="extra", world_position_m=[0, 0, 0]),
                        dict(detection_id="accepted", world_position_m=[0, 0, 0.3]),
                    ]
                }
            ],
            "east": [
                {"detections": [dict(detection_id="unassociated", world_position_m=[0, 0, 0])]}
            ],
        },
        truth_for_evaluation=[
            dict(time_s=0, available=True, targets=[dict(target_id="Cube", position_m=[0, 0, 0])])
        ],
    )


def test_exact_contributors_and_3d_error_preserve_missing_data():
    reader = example()
    before = deepcopy(reader)
    data = comparison_data(reader, FusionSelection("run", 1, 0))
    assert [r["error_m"] for r in data["rows"]] == [0.3, None, 0.1]
    assert data["rows"][1]["reason"] == "No contribution"
    assert reader == before
    reader["truth_for_evaluation"][0]["available"] = False
    assert all(
        r["error_m"] is None for r in comparison_data(reader, FusionSelection("run", 1, 0))["rows"]
    )


def test_extra_estimates_do_not_claim_a_truth_match_in_multi_target_scene():
    reader = example()
    reader["truth_for_evaluation"][0]["targets"].append(
        dict(target_id="Other", position_m=[10, 0, 0])
    )
    reader["histories"]["fused"][0]["tracks"][0]["position_m"] = [5, 0, 0]
    data = comparison_data(reader, FusionSelection("run", 1, 0))
    assert data["target"] is None
    assert all(r["error_m"] is None for r in data["rows"])


def test_projection_keeps_full_3d_errors_and_clear_removes_old_estimate(qtbot):
    widget = FusionComparison()
    qtbot.addWidget(widget)
    widget.update_result(example(), FusionSelection("run", 1, 0))
    widget.plane.setCurrentText("XZ")
    np.testing.assert_allclose(
        [r["error_m"] for r in widget.data["rows"] if r["error_m"] is not None], [0.3, 0.1]
    )
    widget.update_result(example(), None)
    assert widget.data is None
    assert not widget.charts.plots[0].listDataItems()
