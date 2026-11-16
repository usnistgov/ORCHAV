"""Cartesian summaries preserve measured states, coordinate frames, and gaps."""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from generator.core.pipeline.sensing import SensingPipelineState
from generator.figures.summary import saved_sensing
from generator.figures.summary.sensing import maybe_generate_sensing_summary
from shared.frames.frame_set_writer import FrameSetWriter
from shared.scenarios.model import SensingSummaryModel
from shared.sensing.angular import receiver_local_angles
from shared.sensing.plotting import viz as sensing_viz
from shared.sensing.plotting.cartesian import (
    _position_histories,
    cartesian_tracking_event,
    create_cartesian_tracking_figures,
    create_cartesian_tracking_history_figure,
)
from tests.shared.test_sensing_postprocessor import _make_standard_frame
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload


def _payload(index=0, *, empty=False, raw_only=False):
    detection = {
        "range_m": 3.5,
        "velocity_m_s": -0.8,
        "x_m": -3.0 + index * 0.1,
        "y_m": -1.5 + index * 0.1,
        "z_m": -0.5 + index * 0.1,
        "track_id": 7,
        "track_confirmed": index > 0,
    }
    if not raw_only:
        detection.update(track_x_m=-3.1 + index * 0.1, track_y_m=-1.6, track_z_m=-0.6)
    return build_visualizer_sensing_payload(
        {"detections": [] if empty else [detection]},
        scene_frame_id=100 + index,
        sample_time_s=[1.2, 1.27, 1.43][index],
    )


def _events():
    return [cartesian_tracking_event(_payload(i, empty=i == 1)) for i in range(3)]


def _capture_figures(monkeypatch):
    figures = []
    original = sensing_viz.plt.figure

    def capture(*args, **kwargs):
        figure = original(*args, **kwargs)
        figures.append(figure)
        return figure

    monkeypatch.setattr(sensing_viz.plt, "figure", capture)
    return figures


def test_event_keeps_explicit_states_separate_from_reports_and_preserves_identity():
    payload = _payload(1)
    original = deepcopy(payload)
    event = cartesian_tracking_event(payload)
    assert event["time_s"] == 1.27
    assert event["scene_frame_id"] == 101
    assert event["acquisition_id"] == payload["temporal"]["acquisition"]["acquisition_id"]
    assert event["pair_id"] == payload["temporal"]["acquisition"]["pair_id"]
    assert event["coordinate_frame"] == "rx_array_local"
    assert event["tracks"] == [{"track_id": 7, "position_m": [-3.0, -1.6, -0.6], "confirmed": True}]
    np.testing.assert_allclose(event["detections"], [[-2.9, -1.4, -0.4]])
    assert event["truth"] == []
    assert payload == original


def test_radial_identity_does_not_fabricate_cartesian_states_or_confirmation():
    event = cartesian_tracking_event(_payload(raw_only=True))
    assert event["detections"]
    assert event["tracks"] == []
    payload = _payload()
    del payload["detections"][0]["track_confirmed"]
    assert not cartesian_tracking_event(payload)["tracks"][0]["confirmed"]


def test_complete_empty_event_is_retained_and_warmup_is_omitted():
    payload = _payload(empty=True)
    assert cartesian_tracking_event(payload)["tracks"] == []
    del payload["temporal"]["result"]
    assert cartesian_tracking_event(payload) is None


def _with_rotated_truth():
    payload = _payload()
    local = np.array([4.0, -2.0, -1.0])
    receiver = np.array([-1.0, 3.0, 0.5])
    orientation = np.array([0.7, -0.4, 0.3])  # yaw, pitch, roll
    world = receiver + Rotation.from_euler("ZYX", orientation).apply(local)
    azimuth, elevation = receiver_local_angles(world, receiver, orientation)
    payload["rx_array"] = {"rows": 4, "columns": 4}
    payload["temporal"]["truth"] = {"reference_time_s": 1.2}
    payload["config_resolved"].update(
        eval_association_range_gate_m=1.0, eval_association_velocity_gate_m_s=1.0
    )
    payload["truth"] = {
        "targets": [
            {
                "target_id": "cube",
                "valid": True,
                "lifecycle": "active",
                "position_m": world.tolist(),
                "pair_observables": [
                    {
                        "pair_id": payload["temporal"]["acquisition"]["pair_id"],
                        "angular_reference_frame": "rx_array_local",
                        "receiver_azimuth_deg": azimuth,
                        "receiver_elevation_deg": elevation,
                        "monostatic_equivalent_range_m": float(np.linalg.norm(world - receiver)),
                        "monostatic_equivalent_velocity_m_s": 0.0,
                        "velocity_valid": True,
                    }
                ],
            }
        ]
    }
    payload["pair_axis"] = [
        {
            key: payload["temporal"]["acquisition"][key]
            for key in ("pair_id", "tx_sensor_id", "rx_sensor_id")
        }
    ]
    payload["pair_axis"][0].update(frame_pair_index=0, tx_device_index=0, rx_device_index=0)
    return payload, local, world


def _truth_scene():
    return {
        "_frame_timestamp_s": 1.2,
        "tx_positions": [[-1.0, 3.0, 0.5]],
        "rx_positions": [[-1.0, 3.0, 0.5]],
    }


def test_truth_overlay_uses_full_rotated_receiver_frame_and_translated_height():
    payload, local, world = _with_rotated_truth()
    position = cartesian_tracking_event(payload, scene_context=_truth_scene())["truth"][0][
        "position_m"
    ]
    np.testing.assert_allclose(position, local, atol=1e-12)
    assert not np.allclose(position, world)


@pytest.mark.parametrize("missing", ["frame", "epoch", "pair", "bistatic"])
def test_truth_without_matching_local_provenance_is_omitted(missing):
    payload, _, _ = _with_rotated_truth()
    observable = payload["truth"]["targets"][0]["pair_observables"][0]
    if missing == "frame":
        del observable["angular_reference_frame"]
    elif missing == "epoch":
        payload["temporal"]["truth"]["reference_time_s"] += 0.1
    elif missing == "pair":
        observable["pair_id"] = "other-pair"
    else:
        payload["config_resolved"]["range_mode"] = "bistatic"
    event = cartesian_tracking_event(payload, scene_context=_truth_scene())
    assert event["truth"] == []
    assert event["tracks"]


@pytest.mark.parametrize("missing", ["scene", "epoch", "pair", "separated", "index"])
def test_monostatic_range_alone_does_not_prove_colocation(missing):
    payload, _, _ = _with_rotated_truth()
    scene = _truth_scene()
    if missing == "scene":
        scene = None
    elif missing == "epoch":
        scene["_frame_timestamp_s"] += 0.063
    elif missing == "pair":
        payload["pair_axis"][0]["pair_id"] = "other-pair"
    elif missing == "index":
        payload["pair_axis"][0]["frame_pair_index"] = 2
    else:
        scene["tx_positions"][0][0] += 1
    event = cartesian_tracking_event(payload, scene_context=scene)
    assert event["truth"] == []
    assert event["tracks"]


def test_native_same_epoch_snapshot_is_valid_colocation_evidence():
    payload, local, _ = _with_rotated_truth()
    scene = {
        "scene_time_s": 1.2,
        "tx_positions_snapshot": [[-1.0, 3.0, 0.5]],
        "rx_positions_snapshot": [[-1.0, 3.0, 0.5]],
    }
    event = cartesian_tracking_event(payload, scene_context=scene)
    np.testing.assert_allclose(event["truth"][0]["position_m"], local)


def test_missing_epoch_is_nan_and_time_panels_preserve_exact_times(tmp_path, monkeypatch):
    figures = _capture_figures(monkeypatch)
    events = _events()
    times, tracks, _ = _position_histories(events)
    np.testing.assert_array_equal(times, [1.2, 1.27, 1.43])
    assert np.isnan(tracks[7]["positions"][1]).all()
    create_cartesian_tracking_history_figure(events, tmp_path / "history.png")
    for dimension, axis in enumerate(figures[-1].axes):
        line = axis.lines[0]
        np.testing.assert_array_equal(line.get_xdata(), times)
        np.testing.assert_array_equal(line.get_ydata(), tracks[7]["positions"][:, dimension])
        assert axis.get_ylim()[1] < 0  # Cartesian components are signed, not physical ranges.
        assert axis.get_ylabel() == f"Receiver-local {'XYZ'[dimension]} (m)"
    assert figures[-1].axes[-1].get_xlabel() == "Result reference time (s)"


@pytest.mark.parametrize("invalid", ["pair", "frame", "time", "duplicate"])
def test_incompatible_cartesian_histories_fail_explicitly(invalid):
    events = _events()
    if invalid == "pair":
        events[1]["pair_id"] = "other-pair"
    elif invalid == "frame":
        events[1]["coordinate_frame"] = "world"
    elif invalid == "time":
        events[1]["time_s"] = events[0]["time_s"]
    else:
        events[0]["tracks"] *= 2
    with pytest.raises(ValueError):
        _position_histories(events)


def test_local_spatial_renderers_preserve_track_and_truth_gaps(tmp_path, monkeypatch):
    figures = _capture_figures(monkeypatch)
    events = _events()
    for event in (events[0], events[-1]):
        event["truth"] = [{"target_id": "cube", "position_m": [-3.0, -1.5, -0.5]}]
    paths = create_cartesian_tracking_figures(
        events, tmp_path, figures={"tracking_xy", "tracking_xyz"}
    )
    assert {p.name for p in paths} == {"tracking_xy.png", "tracking_xyz.png"}
    xy, xyz = [fig.axes[0] for fig in figures]
    for line in xy.lines:
        assert len(line.get_xdata()) == 3
        assert np.isnan(line.get_xdata()[1])
    for line in xyz.lines:
        coords = line.get_data_3d()
        assert all(len(component) == 3 and np.isnan(component[1]) for component in coords)
    assert "Receiver-local" in xyz.get_zlabel()
    assert "Reported detections" in xy.get_legend_handles_labels()[1]
    assert "Reported detections" in xyz.get_legend_handles_labels()[1]


def test_xyz_renderer_local_choice_ignores_world_fields_and_default_is_unchanged():
    tracks = {7: [{"x_m": 1.0, "y_m": 2.0, "z_m": 3.0, "world_x_m": 10.0, "world_y_m": 20.0}]}
    local = sensing_viz.create_xyz_tracking_figure(tracks, coordinate_frame="receiver_local")
    world = sensing_viz.create_xyz_tracking_figure(tracks)
    np.testing.assert_array_equal(local.axes[0].collections[0]._offsets3d, [[1], [2], [3]])
    np.testing.assert_array_equal(world.axes[0].collections[0]._offsets3d, [[10], [20], [3]])
    assert world.axes[0].get_xlabel() == "World X (m)"
    sensing_viz.plt.close(local)
    sensing_viz.plt.close(world)


def test_two_dimensional_states_do_not_manufacture_height(tmp_path):
    payload = _payload()
    del payload["detections"][0]["track_z_m"]
    events = [cartesian_tracking_event(payload)]
    create_cartesian_tracking_figures(events, tmp_path, figures={"tracking_xy"})
    for figure in ("tracking_xyz", "tracking_cartesian"):
        with pytest.raises(ValueError, match="three-dimensional"):
            create_cartesian_tracking_figures(events, tmp_path, figures={figure})


def _scenario(figures):
    return SimpleNamespace(
        generator_summary={"enabled": True, "create": ["sensing"], "sensing": {"figures": figures}}
    )


def test_standard_dispatch_skips_radial_ids_without_cartesian_states(tmp_path):
    payload = _payload(raw_only=True)
    output = maybe_generate_sensing_summary(
        payload,
        [],
        gt_state=None,
        scenario_context=_scenario(["tracking_xyz"]),
        cartesian_history=[cartesian_tracking_event(payload)],
        output_root=tmp_path,
        strict=True,
    )
    assert output == []


def test_native_pipeline_collects_without_truth_and_dispatches_same_generic_events(tmp_path):
    state = SensingPipelineState(
        enabled=True,
        frame_dt_s=0.1,
        gt_dt_s=0.1,
        first_step=0,
        collect_summary_cartesian=True,
        helpers=SimpleNamespace(finalize_sensing_v4_publication=lambda _: None),
    )
    payload = _payload()
    state._finalize_published_sensing({"sensing": payload})
    state.last_sensing_data = payload
    assert state.gt_history == []
    assert state.summary_cartesian_history == [cartesian_tracking_event(payload)]
    state.generate_summary(_scenario(["tracking_cartesian"]), output_root=tmp_path, strict=True)
    assert (tmp_path / "sensing" / "tracking_cartesian.png").is_file()
    default = SensingPipelineState(enabled=True, frame_dt_s=0.1, gt_dt_s=0.1, first_step=0)
    assert default.collect_summary_cartesian is False


def test_native_schema_adds_opt_in_figures_without_changing_defaults():
    figures = ["tracking_xy", "tracking_xyz", "tracking_cartesian"]
    assert SensingSummaryModel(figures=figures).figures == figures
    assert not set(figures) & set(SensingSummaryModel().figures)


@pytest.mark.parametrize("explicit", [False, True])
def test_saved_cli_emits_generic_cartesian_figures_from_published_states(tmp_path, explicit):
    source = tmp_path / "recording"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index in range(3):
            payload = _payload(index, empty=index == 1)
            frame = _make_standard_frame(frame_idx=index, sensing=payload)
            writer.append(
                replace(frame, timestamp_s=payload["temporal"]["result"]["reference_time_s"])
            )
        writer.finalize()
    output = tmp_path / "summary"
    args = [str(source), "-o", str(output)]
    if explicit:
        for figure in ("tracking_xy", "tracking_xyz", "tracking_cartesian"):
            args.extend(["--figure", figure])
    saved_sensing.main(args)
    receipt = json.loads((output / "sensing" / "summary_provenance.json").read_text())
    assert {"tracking_xy", "tracking_xyz", "tracking_cartesian"} <= set(receipt["figures"])
    assert receipt["result_count"] == 3
    assert [result["reference_time_s"] for result in receipt["results"]] == [1.2, 1.27, 1.43]
    for name in ("tracking_xy", "tracking_xyz", "tracking_cartesian"):
        assert (output / "sensing" / f"{name}.png").is_file()
