"""Saved summary plots preserve published values, identities, and physical time."""

import hashlib
import json
from contextlib import ExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.figures.summary import saved_sensing
from shared.frames.frame_set_writer import FrameSetWriter
from shared.sensing.plotting import viz as sensing_viz
from tests.shared.test_sensing_postprocessor import _make_standard_frame
from tests.shared.test_sensing_sequential_cfr import (
    PRF,
    bind_results,
    processor_for,
    pulses_for,
    run_pulses,
)
from tests.visualizer.fixtures.sensing_v3 import build_visualizer_sensing_payload


@pytest.fixture
def bound_cartesian_scene(monkeypatch):
    from shared.frames.contracts import FrameComponent

    sensing = build_visualizer_sensing_payload()
    sensing["temporal"]["truth"] = {"reference_time_s": 0.0}
    sensing["temporal"]["display_binding"].update(policy="source_scene_exact", scene_frame_id=0)
    pair = sensing["pair_axis"][0]
    pair.update(tx_device_index=0, rx_device_index=0)
    frame = {
        "_typed_sensing": sensing,
        "_frame_index": 63,
        "_frame_timestamp_s": 0.063,
        "_source_generation_id": "source-generation",
        "_source_frame_set_id": "source-frames",
    }
    scene = SimpleNamespace(
        timestamp_s=0.0,
        tx_names=[sensing["temporal"]["acquisition"]["tx_sensor_id"]],
        rx_names=[sensing["temporal"]["acquisition"]["rx_sensor_id"]],
        tx_positions=np.array([[0.0, 0.0, 1.5]]),
        rx_positions=np.array([[0.0, 0.0, 1.5]]),
    )
    manifest = SimpleNamespace(
        generation_id="source-generation", frame_set_id="source-frames", frame_ids=(0, 63)
    )
    calls = []

    def load(index, request):
        assert request.components == frozenset({FrameComponent.DEVICES, FrameComponent.PROVENANCE})
        calls.append(("read", index))
        return SimpleNamespace(frame=scene)

    provider = SimpleNamespace(load_frame_projection=load, close=lambda: calls.append("close"))

    def open_provider(source):
        calls.append("open")
        return source, manifest, provider

    monkeypatch.setattr(saved_sensing, "_open_provider", open_provider)
    return frame, scene, manifest, calls


def test_saved_cartesian_context_uses_exact_bound_scene_without_signal_reads(bound_cartesian_scene):
    from shared.sensing.plotting.cartesian import _colocated_truth_pair

    frame, scene, _, calls = bound_cartesian_scene
    with ExitStack() as stack:
        contexts = saved_sensing._CartesianSceneContexts(Path("saved"), stack)
        context = contexts.for_frame(frame)
        assert context["scene_time_s"] == 0.0
        assert context["rx_positions_snapshot"] is scene.rx_positions
        assert _colocated_truth_pair(frame["_typed_sensing"], context)
        assert not _colocated_truth_pair(frame["_typed_sensing"], frame)
        assert contexts.for_frame(frame)["scene_time_s"] == 0.0
    assert calls == ["open", ("read", 0), ("read", 0), "close"]
    assert frame["_frame_timestamp_s"] == 0.063


@pytest.mark.parametrize("mismatch", ["epoch", "missing_scene", "policy", "sensor", "source"])
def test_saved_cartesian_context_never_substitutes_nearby_or_unbound_pose(
    bound_cartesian_scene, mismatch
):
    frame, scene, manifest, _ = bound_cartesian_scene
    if mismatch == "epoch":
        scene.timestamp_s = 0.0001
    elif mismatch == "missing_scene":
        manifest.frame_ids = (63,)
    elif mismatch == "policy":
        frame["_typed_sensing"]["temporal"]["display_binding"]["policy"] = "nearest"
    elif mismatch == "sensor":
        scene.rx_names = ["unrelated-rx"]
    else:
        manifest.frame_set_id = "different-recording"
    with ExitStack() as stack:
        contexts = saved_sensing._CartesianSceneContexts(Path("saved"), stack)
        if mismatch == "source":
            with pytest.raises(ValueError, match="different recording"):
                contexts.for_frame(frame)
        else:
            assert contexts.for_frame(frame) is None


def test_saved_cartesian_context_keeps_exact_storage_pose_without_extra_provider(
    bound_cartesian_scene,
):
    frame, _, _, calls = bound_cartesian_scene
    frame["_frame_timestamp_s"] = 0.0
    with ExitStack() as stack:
        assert saved_sensing._CartesianSceneContexts(Path("saved"), stack).for_frame(frame) is frame
    assert calls == []


def test_saved_cartesian_context_reads_packed_device_pose_and_timestamp(tmp_path):
    from shared.sensing.plotting.cartesian import _colocated_truth_pair

    source = tmp_path / "recording"
    sensing = build_visualizer_sensing_payload()
    acquisition = sensing["temporal"]["acquisition"]
    sensing["temporal"]["truth"] = {"reference_time_s": 0.0}
    sensing["temporal"]["display_binding"].update(policy="source_scene_exact", scene_frame_id=0)
    sensing["pair_axis"][0].update(tx_device_index=0, rx_device_index=0)
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index, time, rx_x in ((0, 0.0, 0.0), (63, 0.063, 10.0)):
            scene = replace(
                _make_standard_frame(frame_idx=index),
                timestamp_s=time,
                tx_names=(acquisition["tx_sensor_id"],),
                rx_names=(acquisition["rx_sensor_id"],),
                rx_positions=np.array([[rx_x, 0.0, 0.0]]),
            )
            writer.append(scene)
        writer.finalize()
    manifest_path = next(source.rglob("frames_manifest.json"))
    manifest = json.loads(manifest_path.read_text())
    frame = {
        "_typed_sensing": sensing,
        "_frame_index": 63,
        "_frame_timestamp_s": 0.063,
        "_source_generation_id": manifest["generation_id"],
        "_source_frame_set_id": manifest["frame_set_id"],
    }
    with ExitStack() as stack:
        context = saved_sensing._CartesianSceneContexts(source, stack).for_frame(frame)
    assert context["scene_time_s"] == 0.0
    np.testing.assert_array_equal(context["rx_positions_snapshot"], [[0.0, 0.0, 0.0]])
    assert _colocated_truth_pair(sensing, context)


def _capture_figure(monkeypatch):
    figures = []
    subplots = sensing_viz.plt.subplots

    def capture(*args, **kwargs):
        figure, axes = subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(sensing_viz.plt, "subplots", capture)
    return figures


def test_range_profile_plots_authored_samples_and_saved_estimate(tmp_path, monkeypatch):
    axis = np.array([0.0, 0.4, 2.3, 2.8, 8.0])
    power = np.array([1e-14, 1e-10, 1e-3, 1e-2, 1e-12])
    sensing = build_visualizer_sensing_payload(
        {
            "range_axis_m": axis,
            "range_profile_by_pair": power[np.newaxis, :],
            "detections": [{"range_m": 2.37}, {"range_m": 2.37}, {}],
        }
    )
    figures = _capture_figure(monkeypatch)
    sensing_viz.create_range_profile_figure(sensing, tmp_path / "profile.png", zoom_to_data=True)
    ax = figures[0].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), axis)
    np.testing.assert_allclose(ax.lines[0].get_ydata(), 10 * np.log10(power))
    assert len(ax.lines) == 2  # duplicate ranges share one saved-estimate marker
    np.testing.assert_array_equal(ax.lines[1].get_xdata(), [2.37, 2.37])
    assert ax.get_legend_handles_labels()[1] == ["Reported range: 2.370 m"]
    assert ax.get_xlim()[0] < 2.37 < ax.get_xlim()[1] < 8
    np.testing.assert_array_equal(sensing["range_profile_by_pair"][0], power)
    assert (tmp_path / "profile.png").is_file()


def test_zero_profile_has_no_fabricated_estimate_and_full_extent(tmp_path, monkeypatch):
    sensing = build_visualizer_sensing_payload(
        {"range_axis_m": np.array([1.0, 2.0, 4.0]), "range_profile_by_pair": np.zeros((1, 3))}
    )
    figures = _capture_figure(monkeypatch)
    sensing_viz.create_range_profile_figure(sensing, tmp_path / "empty.png", zoom_to_data=True)
    ax = figures[0].axes[0]
    assert len(ax.lines) == 1
    assert ax.get_xlim() == (1, 4)
    assert not ax.get_legend_handles_labels()[1]


def test_analytical_range_profile_uses_same_renderer_without_acquisition_metadata(
    tmp_path, monkeypatch
):
    axis = np.array([1.0, 1.3, 2.1, 4.7])
    power = np.array([1e-5, 0.3, 1.0, 1e-3])
    figures = _capture_figure(monkeypatch)
    sensing_viz.create_range_profile_plot(
        power,
        axis,
        tmp_path / "analytical.png",
        range_mode="monostatic",
        description="analytical single path",
        range_resolution_m=0.8,
        range_xlim=(0.5, 5.0),
    )
    ax = figures[0].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), axis)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), 10 * np.log10(power))
    assert ax.get_xlim() == (0.5, 5.0)
    assert "analytical single path" in ax.get_title()
    assert len(ax.lines) == 1 and not ax.get_legend_handles_labels()[1]


def _history():
    return [
        {
            "step": frame,
            "reference_time_s": time,
            "target_ids": [],
            "target_ranges": [],
            "target_speeds": [],
            "target_velocity_valid": [],
            "det_ranges": [value],
            "det_velocities": [0.5],
        }
        for frame, time, value in [(63, 1.2, 4.7), (127, 1.264, 4.6)]
    ]


def test_history_and_tracking_use_exact_result_time_without_truth(tmp_path, monkeypatch):
    figures = _capture_figure(monkeypatch)
    history = _history()
    sensing_viz.create_gt_evolution_figure(
        history, tmp_path / "detections.png", use_physical_time=True
    )
    np.testing.assert_array_equal(
        figures[0].axes[0].collections[0].get_offsets()[:, 0], [1.2, 1.264]
    )
    tracks = {
        9: [
            {"frame": row["step"], "range_m": value, "velocity_m_s": 0.5, "confirmed": confirmed}
            for row, value, confirmed in zip(history, [4.71, 4.61], [False, True], strict=True)
        ]
    }
    sensing_viz.create_tracking_figure(
        history, tracks, tmp_path / "tracks.png", use_physical_time=True
    )
    ax = figures[1].axes[0]
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1.2, 1.264])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [4.71, 4.61])
    assert ax.lines[0].get_linestyle() == "--"
    assert "Track 9" in ax.get_legend_handles_labels()[1]
    assert ax.get_xlabel() == "Result reference time (s)"
    assert "Track 9" in figures[1].axes[1].get_legend_handles_labels()[1]


@pytest.mark.parametrize(
    "truth_velocity, detection_velocity, track_velocity, starts_at_zero",
    [
        (1.0, 0.9, 1.1, True),
        (0.0, 0.0, 0.0, True),
        (-0.1, 0.9, 1.1, False),
        (1.0, -0.1, 1.1, False),
        (1.0, 0.9, -0.1, False),
        (-1.0, -0.9, -1.1, False),
        (None, np.nan, np.nan, False),
        (None, 0.9, 1.1, True),
    ],
    ids=[
        "positive",
        "zero",
        "negative-truth",
        "negative-detection",
        "negative-track",
        "all-negative",
        "no-valid-velocity",
        "invalid-truth-ignored",
    ],
)
def test_tracking_zero_reference_includes_all_plotted_sources(
    tmp_path, monkeypatch, truth_velocity, detection_velocity, track_velocity, starts_at_zero
):
    figures = _capture_figure(monkeypatch)
    history = _history()
    for row in history:
        row.update(
            target_ids=["cube"],
            target_ranges=[5.0],
            target_speeds=[truth_velocity],
            target_velocity_valid=[truth_velocity is not None],
            det_velocities=[detection_velocity],
        )
    tracks = {
        9: [
            {"frame": row["step"], "range_m": 4.8, "velocity_m_s": track_velocity}
            for row in history
        ]
    }
    sensing_viz.create_tracking_figure(
        history, tracks, tmp_path / "tracking.png", pruned_track_ids={9}
    )

    range_axis, velocity_axis = figures[0].axes
    assert range_axis.get_ylim()[0] == 0
    assert range_axis.get_ylim()[1] > 5
    low, high = velocity_axis.get_ylim()
    assert np.isfinite([low, high]).all()
    assert low < high
    assert bool(low == 0) is starts_at_zero
    finite_values = np.asarray(
        [v for v in (truth_velocity, detection_velocity, track_velocity) if v is not None]
    )
    finite_values = finite_values[np.isfinite(finite_values)]
    if finite_values.size:
        assert low <= np.min(finite_values) <= np.max(finite_values) < high
        if starts_at_zero:
            assert high >= np.max(finite_values) * (1 + velocity_axis.get_ymargin())
    # Axis choices never modify the recorded report or state values.
    np.testing.assert_array_equal(velocity_axis.lines[-1].get_ydata(), [track_velocity] * 2)


@pytest.mark.parametrize("time", [None, np.nan, 1.0])
def test_physical_history_requires_authored_ordered_time(tmp_path, time):
    history = _history()
    history[-1]["reference_time_s"] = time
    with pytest.raises(ValueError, match="reference_time_s|reference times"):
        sensing_viz.create_gt_evolution_figure(
            history, tmp_path / "bad.png", use_physical_time=True
        )


@pytest.mark.parametrize("fixed_limits", [False, True])
@pytest.mark.parametrize("selected_frame", [None, 7])
def test_saved_summary_cli_reads_packed_warmup_without_changing_observations(
    tmp_path, monkeypatch, fixed_limits, selected_frame
):
    plotted = _capture_figure(monkeypatch)
    processor = processor_for(samples=8)
    results = bind_results(processor, run_pulses(processor, pulses_for("stationary", 9)))
    source = tmp_path / "recording"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for index, sensing in enumerate(results):
            frame = _make_standard_frame(frame_idx=index, sensing=sensing)
            writer.append(replace(frame, timestamp_s=2 + index / PRF))
        writer.finalize()
    before = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    output = tmp_path / "summary"
    saved_sensing.main(
        [
            str(source),
            "-o",
            str(output),
            "--figure",
            "range_profile",
            "--figure",
            "range_doppler",
            "--figure",
            "doppler",
            "--figure",
            "detection_history",
            *(["--frame-index", str(selected_frame)] if selected_frame is not None else []),
            *(
                [
                    "--range-limits",
                    "1",
                    "8",
                    "--velocity-limits",
                    "-2",
                    "2",
                    "--power-limits",
                    "-90",
                    "-10",
                ]
                if fixed_limits
                else []
            ),
        ]
    )
    names = {path.name for path in (output / "sensing").iterdir()}
    assert names == {
        "range_profile.png",
        "range_profile_full.png",
        "range_doppler.png",
        "range_doppler_full.png",
        "doppler_history.png",
        "detection_history.png",
        "summary_provenance.json",
    }
    receipt = json.loads((output / "sensing" / "summary_provenance.json").read_text())
    assert receipt["incomplete_acquisition_count"] == 7
    assert receipt["result_count"] == (2 if selected_frame is None else 1)
    assert receipt["source_files_sha256"] == before
    result_label = "last frame" if selected_frame is None else "selected result"
    for index in (2, 3):
        assert plotted[index].axes[0].get_title() == f"Range–Velocity Map ({result_label})"
    for index in (0, 1):
        assert plotted[index].axes[0].get_title() == "Range response · last acquisition sample"
    if fixed_limits:
        assert receipt["display_limits"] == {
            "range_limits_m": [1.0, 8.0],
            "velocity_limits_m_s": [-2.0, 2.0],
            "power_limits_db": [-90.0, -10.0],
        }
        assert plotted[0].axes[0].get_xlim() == (1, 8)
        assert plotted[1].axes[0].get_xlim() != (1, 8)
        assert plotted[2].axes[0].get_xlim() == (1, 8)
        assert plotted[2].axes[0].get_ylim() == (-2, 2)
        assert plotted[3].axes[0].get_xlim() != (1, 8)
        assert plotted[3].axes[0].get_ylim() != (-2, 2)
        for index in (2, 3):
            assert plotted[index].axes[0].collections[0].get_clim() == (-90, -10)
    else:
        assert "display_limits" not in receipt
    expected_frames = [7, 8] if selected_frame is None else [7]
    assert [r["frame_index"] for r in receipt["results"]] == expected_frames
    np.testing.assert_allclose(
        [r["reference_time_s"] for r in receipt["results"]],
        [2.0035, 2.0045] if selected_frame is None else [2.0035],
    )
    assert before == {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in source.rglob("*")
        if path.is_file()
    }
    with pytest.raises(FileExistsError, match="fresh"):
        saved_sensing.generate_saved_sensing_summary(source, output)
    for frame_index, message in [(6, "no complete"), (99, "not found")]:
        with pytest.raises(ValueError, match=message):
            saved_sensing.generate_saved_sensing_summary(
                source, tmp_path / f"bad{frame_index}", frame_index=frame_index
            )


def test_saved_summary_dispatches_recorded_tracks_and_stops_at_requested_frame(
    tmp_path, monkeypatch
):
    source = tmp_path / "source"
    source.mkdir()
    (source / "frames_manifest.json").write_text("{}")
    records = []
    for row in _history():
        sensing = build_visualizer_sensing_payload(
            {
                "detections": [
                    {
                        "range_m": row["det_ranges"][0],
                        "velocity_m_s": 0.5,
                        "track_id": 9,
                        "track_range_m": 4.6,
                        "track_confirmed": True,
                    }
                ]
            },
            scene_frame_id=row["step"],
            sample_time_s=row["reference_time_s"],
        )
        records.append(
            {
                "_typed_sensing": sensing,
                "_frame_index": row["step"],
                "_source_generation_id": "g",
                "_source_frame_set_id": "f",
            }
        )
    closed = []

    def stream(_source):
        try:
            yield from records
        finally:
            closed.append(True)

    captured = {}

    def render(sensing, history, **kwargs):
        captured.update(sensing=sensing, history=history, **kwargs)
        (kwargs["output_root"] / "sensing").mkdir(parents=True)
        return []

    original = deepcopy(records)
    monkeypatch.setattr(saved_sensing, "iter_frames_sensing", stream)
    monkeypatch.setattr(saved_sensing, "maybe_generate_sensing_summary", render)
    saved_sensing.generate_saved_sensing_summary(
        source, tmp_path / "out", figures=["tracking"], frame_index=63
    )
    assert closed == [True]
    assert len(captured["history"]) == 1
    assert captured["tracking_history"][9][0]["range_m"] == 4.6
    assert captured["tracking_history"][9][0]["confirmed"] is True
    assert records == original
    records[-1]["_typed_sensing"]["temporal"]["result"][
        "processing_recipe_id"
    ] = "another_saved_recipe"
    saved_sensing.generate_saved_sensing_summary(source, tmp_path / "all", figures=["tracking"])
    assert len(captured["tracking_history"][9]) == 2
    assert closed == [True, True]


def _reference_summary_payload(case):
    from tests.shared.test_sensing_evaluation_truth import animated_frame
    from tests.shared.test_sensing_instantaneous_evaluation import (
        interval_payload,
        reference_payload,
    )

    if case in {"root", "legacy_root"}:
        sensing = reference_payload(
            target_velocity=(0, 0, 0),
            tx_velocity=(0.5, 0, 0),
            rx_velocity=(0.5, 0, 0),
        )
        declared = animated_frame()["sensing"]
        target = sensing["truth"]["targets"][0]
        target["target_id"] = "walker"
        target["animated_asset"] = declared["truth"]["targets"][0]["animated_asset"]
        target["pair_observables"][0]["velocity_reference_evidence"]["target_id"] = "walker"
        sensing["target_catalog"] = declared["target_catalog"]
        sensing["config_resolved"]["acquisition"] = {"sample_source": "coherent_one_solve"}
        report_velocity = -0.5
        if case == "legacy_root":
            sensing["truth"].pop("velocity_reference_policy")
            target["pair_observables"][0].pop("velocity_reference_evidence")
            report_velocity = 0.5
    else:
        sensing = interval_payload()
        report_velocity = 100.0
        if case == "legacy":
            sensing["truth"].pop("velocity_reference_policy")
            observable = sensing["truth"]["targets"][0]["pair_observables"][0]
            observable.pop("velocity_reference_evidence")
            report_velocity = observable["monostatic_equivalent_velocity_m_s"]
    sensing["config_resolved"].update(
        carrier_freq_hz=28e9,
        eval_association_range_gate_m=0.6,
        eval_association_velocity_gate_m_s=0.25,
    )
    sensing["temporal"]["result"]["processing_recipe_id"] = "recorded-recipe"
    sensing["detections"] = [{"range_m": 5.0, "velocity_m_s": report_velocity}]
    return sensing


@pytest.mark.parametrize("case", ["root", "interval", "legacy", "legacy_root"])
def test_saved_summary_reference_history_and_metrics_match_declared_policy(
    tmp_path, monkeypatch, case
):
    from shared.sensing.evaluation import INSTANTANEOUS_REFERENCE_PROTOCOL
    from shared.sensing.evaluation_truth import ROOT_RELATIVE_REFERENCE

    sensing = _reference_summary_payload(case)
    records = [
        {
            "_typed_sensing": sensing,
            "_frame_index": 63,
            "_source_generation_id": "source-generation",
            "_source_frame_set_id": "source-frames",
        }
    ]
    before = deepcopy(records)
    source = tmp_path / "source"
    source.mkdir()
    (source / "frames_manifest.json").write_text("{}")
    captured = {}

    def render(_sensing, history, **kwargs):
        captured.update(history=history, **kwargs)
        (kwargs["output_root"] / "sensing").mkdir(parents=True)
        return []

    monkeypatch.setattr(saved_sensing, "iter_frames_sensing", lambda _: (r for r in records))
    monkeypatch.setattr(saved_sensing, "maybe_generate_sensing_summary", render)
    saved_sensing.generate_saved_sensing_summary(
        source, tmp_path / "summary", figures=["truth", "tracking", "detection"]
    )
    history = captured["history"][0]
    association = captured["association_history"][0]
    assert records == before
    assert (association["tp"], association["fp"], association["fn"]) == (1, 0, 0)
    if case in {"legacy", "legacy_root"}:
        expected_speed = 0.5 if case == "legacy_root" else 0.6538604244203522
        assert history["target_speeds"] == [expected_speed]
        assert "evaluation_reference_protocol" not in history
        assert "evaluation_reference_protocol" not in association
        assert association["velocity_rmse"] == 0.0
    else:
        assert history["evaluation_reference_protocol"] == INSTANTANEOUS_REFERENCE_PROTOCOL
        assert association["evaluation_reference_protocol"] == INSTANTANEOUS_REFERENCE_PROTOCOL
        if case == "root":
            assert history["target_speeds"] == [-0.5]
            assert history["recorded_target_speeds"] == [0.5]
            assert history["target_velocity_references"] == [ROOT_RELATIVE_REFERENCE]
            assert association["velocity_rmse"] == 0.0
            assert association["velocity_matched_pair_count"] == 1
        else:
            assert history["target_speeds"] == [None]
            assert history["target_velocity_valid"] == [False]
            assert association["velocity_rmse"] is None
            assert association["velocity_matched_pair_count"] == 0
            assert association["velocity_excluded_matched_pair_count"] == 1
            assert association["velocity_unavailable_opportunity_count"] == 1


@pytest.mark.parametrize("case", ["root", "interval", "legacy", "legacy_root"])
def test_saved_summary_rejects_mismatched_truth_epoch_for_every_reference_policy(
    tmp_path, monkeypatch, case
):
    sensing = _reference_summary_payload(case)
    sensing["temporal"]["result"]["reference_time_s"] += 0.001
    source = tmp_path / "source"
    source.mkdir()
    (source / "frames_manifest.json").write_text("{}")
    frame = {"_typed_sensing": sensing, "_frame_index": 63}
    monkeypatch.setattr(saved_sensing, "iter_frames_sensing", lambda _: (r for r in [frame]))
    with pytest.raises(ValueError, match="Truth and result reference times must match"):
        saved_sensing.generate_saved_sensing_summary(
            source, tmp_path / "summary", figures=["truth", "detection"]
        )
    assert not (tmp_path / "summary").exists()


@pytest.mark.parametrize("compatible", [True, False])
def test_default_saved_summary_compares_verified_replay_settings_and_keeps_recipe_ids(
    tmp_path, monkeypatch, compatible
):
    from shared.extensions.sensing import canonical_processing_recipe_id
    from tests.shared.test_sensing_doppler_summary import _source_bound_replay_payload

    source = tmp_path / "source"
    source.mkdir()
    (source / "frames_manifest.json").write_text("{}")
    payloads = [_source_bound_replay_payload(0.0, 0), _source_bound_replay_payload(0.004, 64)]
    if not compatible:
        config = payloads[1]["config_resolved"]
        recipe = config["processing_recipe"]
        recipe["effective_overrides"]["frequency_window"] = "boxcar"
        recipe_id = canonical_processing_recipe_id(recipe)
        payloads[1]["temporal"]["result"]["processing_recipe_id"] = recipe_id
        for product in payloads[1]["products"].values():
            if product["kind"] == "range_doppler":
                product["lineage"]["processing_recipe_id"] = recipe_id
                product["processing"]["recipe_id"] = recipe_id
    records = [
        {
            "_typed_sensing": payload,
            "_frame_index": index,
            "_source_generation_id": "g",
            "_source_frame_set_id": "f",
        }
        for index, payload in enumerate(payloads)
    ]
    selected = []

    def render(_sensing, _history, **kwargs):
        selected.extend(kwargs["scenario_context"].generator_summary["sensing"]["figures"])
        (kwargs["output_root"] / "sensing").mkdir(parents=True)
        return []

    monkeypatch.setattr(saved_sensing, "maybe_generate_sensing_summary", render)
    monkeypatch.setattr(saved_sensing, "iter_frames_sensing", lambda _source: (r for r in records))
    output = tmp_path / "summary"
    saved_sensing.generate_saved_sensing_summary(source, output)
    assert ("doppler" in selected) is compatible
    receipt = json.loads((output / "sensing/summary_provenance.json").read_text())
    expected_ids = [payload["temporal"]["result"]["processing_recipe_id"] for payload in payloads]
    assert len(set(expected_ids)) == 2
    assert [result["processing_recipe_id"] for result in receipt["results"]] == expected_ids


@pytest.mark.parametrize("has_azimuth, has_elevation", [(True, False), (True, True), (False, True)])
@pytest.mark.parametrize("result_label", ["last frame", "selected result"])
def test_summary_renders_saved_angular_power_without_truth(
    tmp_path, monkeypatch, has_azimuth, has_elevation, result_label
):
    from types import SimpleNamespace

    from generator.figures.summary.sensing import maybe_generate_sensing_summary
    from tests.generator.unit.test_sensing_summary_fig import _typed_angular_payload

    sensing = _typed_angular_payload()
    if not has_azimuth:
        sensing["azimuth_direction_cosine"] = np.array([0.0])
        sensing["azimuth_broadside_angle_deg"] = np.array([0.0])
        sensing["range_doppler_angle_by_pair"] = sensing["range_doppler_angle_by_pair"][:, :, :1]
        for product in sensing["products"].values():
            if product["kind"] == "range_doppler_angle":
                product["shape"][2] = 1
    if not has_elevation:
        sensing["elevation_direction_cosine"] = np.array([0.0])
        sensing["elevation_broadside_angle_deg"] = np.array([0.0])
        sensing["range_doppler_angle_by_pair"] = sensing["range_doppler_angle_by_pair"][:, :1]
        for product in sensing["products"].values():
            if product["kind"] == "range_doppler_angle":
                product["shape"][1] = 1
    context = SimpleNamespace(
        generator_summary={
            "enabled": True,
            "create": ["sensing"],
            "sensing": {"figures": ["angles"]},
            "output": {"dir": tmp_path},
        }
    )
    figures = _capture_figure(monkeypatch)
    paths = maybe_generate_sensing_summary(
        sensing, [], None, context, strict=True, result_label=result_label
    )
    names = {path.name for path in paths}
    expected = {"range_azimuth.png", "doppler_azimuth.png"} if has_azimuth else set()
    if has_elevation:
        expected.add("range_elevation.png")
        if has_azimuth:
            expected.add("azimuth_elevation.png")
    assert names == expected
    assert all(path.stat().st_size > 1000 for path in paths)
    assert all(figure.axes[0].get_title().endswith(f"({result_label})") for figure in figures)


@pytest.mark.parametrize("target_count", [1, 10])
def test_rd_title_clears_the_complete_truth_legend(tmp_path, monkeypatch, target_count):
    figures = _capture_figure(monkeypatch)
    power = np.arange(1, 13, dtype=float).reshape(3, 4)
    before = power.copy()
    sensing_viz.create_range_doppler_array_figure(
        power_linear=power,
        range_axis_m=np.arange(4, dtype=float),
        doppler_axis_hz=np.array([-1.0, 0.0, 1.0]),
        range_mode="monostatic",
        carrier_freq_hz=28e9,
        output_path=tmp_path / "rd.png",
        target_truth=[
            {"target_id": f"target_{i}", "range_m": i / 4, "velocity_m_s": 0.0}
            for i in range(target_count)
        ],
        display_config={"range_resolution_m": 1.0, "velocity_resolution_m_s": 0.1},
    )
    figure = figures[0]
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(figure)
    figure.canvas.draw()
    axis = figure.axes[0]
    renderer = figure.canvas.get_renderer()
    title_box = axis.title.get_window_extent(renderer)
    legend_box = axis.get_legend().get_window_extent(renderer)
    assert title_box.y0 > legend_box.y1
    assert title_box.y1 < figure.bbox.y1
    assert len(axis.get_legend().get_texts()) == target_count
    np.testing.assert_array_equal(power, before)
