"""Derived cadence selects original observations and preserves their scientific values."""

from copy import deepcopy
from dataclasses import replace
from hashlib import sha256

import numpy as np
import pytest

from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from shared.sensing.loader import iter_frames_sensing, load_frames_sensing
from shared.sensing.postprocessor import main
from shared.sensing.window_selection import output_hop_samples, select_saved_windows
from tests.shared.test_sensing_postprocessor import _make_standard_frame
from tests.shared.test_sensing_sequential_cfr import (
    PRF,
    bind_results,
    processor_for,
    pulses_for,
    run_pulses,
)


@pytest.fixture
def saved_windows(tmp_path):
    processor = processor_for(samples=8, noise=0.002, recording_profile="reprocessing")
    results = bind_results(processor, run_pulses(processor, pulses_for("oscillating", 24)))
    source = tmp_path / "source"
    with FrameSetWriter.create_new(source, compression="lzf", chunk_size=4) as writer:
        for index, sensing in enumerate(results):
            frame = _make_standard_frame(frame_idx=index, sensing=sensing)
            writer.append(replace(frame, timestamp_s=2 + index / PRF))
        writer.finalize()
    return source


def test_selection_reads_metadata_only_and_loader_reads_only_selected_rows(
    saved_windows, monkeypatch
):
    original = Hdf5Provider.load_frame_projection
    original_batch = Hdf5Provider.iter_frame_projections
    metadata_rows = []

    def metadata_only(self, steps, request):
        assert request.sensing_products == frozenset({"config_resolved"})
        for projection in original_batch(self, steps, request):
            sensing = projection.frame.sensing or {}
            assert not any(isinstance(value, np.ndarray) for value in sensing.values())
            metadata_rows.append(projection.frame.frame_index)
            yield projection

    monkeypatch.setattr(Hdf5Provider, "iter_frame_projections", metadata_only)
    seen = []

    def observe(self, step, request):
        seen.append(step)
        return original(self, step, request)

    # Batch metadata selection uses its own validated projection path.
    selected = select_saved_windows(saved_windows, 2)
    assert metadata_rows == list(range(24))
    assert selected.frame_ids == tuple(range(7, 24, 2))
    assert selected.recipe["source_hop_samples"] == 1
    assert selected.recipe["selected_count"] == 9
    monkeypatch.setattr(Hdf5Provider, "load_frame_projection", observe)
    frames = list(iter_frames_sensing(saved_windows, frame_ids=selected.frame_ids))
    assert seen == list(selected.frame_ids)
    for index, frame in enumerate(frames):
        selected.bind(frame, index)
        assert frame["_frame_index"] == index
        assert frame["_output_selection"] == selected.recipe


@pytest.mark.parametrize("hop", [0, -1, True, 1.5, 9])
def test_invalid_selection_cadence_is_rejected(saved_windows, hop):
    with pytest.raises((ValueError, TypeError)):
        select_saved_windows(saved_windows, hop)


def test_maps_retention_requires_selected_complete_windows(
    saved_windows, tmp_path, monkeypatch, capsys
):
    output = tmp_path / "not-created"
    monkeypatch.setattr(
        "sys.argv",
        [
            "orchav-sensing",
            str(saved_windows),
            "--output-dir",
            str(output),
            "--streaming",
            "--rebuild-rd",
            "--recording-profile",
            "maps",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "requires --streaming --result-hop-samples --rebuild-rd" in capsys.readouterr().err
    assert not output.exists()


def test_streaming_selection_preserves_values_memberships_and_local_bindings(
    saved_windows, tmp_path, monkeypatch
):
    originals = load_frames_sensing(saved_windows)
    baseline = {
        path.name: sha256(path.read_bytes()).hexdigest() for path in saved_windows.iterdir()
    }
    output = tmp_path / "derived"
    monkeypatch.setattr(
        "sys.argv",
        [
            "orchav-sensing",
            str(saved_windows),
            "--output-dir",
            str(output),
            "--streaming",
            "--result-hop-samples",
            "2",
            "--rebuild-rd",
            "--coherent-clutter-mode",
            "none",
            "--skip-detection",
            "--recording-profile",
            "maps",
        ],
    )
    numeric_rows = set()
    original_projection = Hdf5Provider.load_frame_projection

    def observe(self, step, request):
        projection = original_projection(self, step, request)
        sensing = projection.frame.sensing or {}
        if any(isinstance(value, np.ndarray) for value in sensing.values()):
            numeric_rows.add(step)
        return projection

    with monkeypatch.context() as observed:
        observed.setattr(Hdf5Provider, "load_frame_projection", observe)
        main()
    assert numeric_rows == set(range(7, 24, 2))
    saved = load_frames_sensing(output)
    assert len(saved) == 9
    all_pulses = set()
    for index, row in enumerate(saved):
        source = originals[7 + 2 * index]["_typed_sensing"]
        actual = row["_typed_sensing"]
        assert actual["temporal"]["acquisition"] == source["temporal"]["acquisition"]
        assert (
            actual["temporal"]["result"]["reference_time_s"]
            == source["temporal"]["result"]["reference_time_s"]
        )
        assert (
            actual["temporal"]["result"]["processing_run_id"]
            != source["temporal"]["result"]["processing_run_id"]
        )
        assert actual["temporal"]["display_binding"]["scene_frame_id"] == index
        assert (
            actual["postprocessing"]["source_display_binding"]
            == source["temporal"]["display_binding"]
        )
        assert row["_frame_index"] == index
        assert (
            row["_frame_timestamp_s"]
            == 2 + source["temporal"]["display_binding"]["scene_frame_id"] / PRF
        )
        assert output_hop_samples(actual) == 2
        assert actual["config_resolved"]["acquisition"]["cpi_hop_samples"] == 1
        assert actual["recording"]["profile"] == "maps"
        assert actual["recording"]["canonical_observation"] is None
        assert actual["detections"] == []
        assert not any(
            np.iscomplexobj(value) for value in actual.values() if isinstance(value, np.ndarray)
        )
        for key in (
            "range_doppler_power_by_pair",
            "range_profile_by_pair",
            "micro_doppler_by_pair",
            "micro_doppler_raw_by_pair",
        ):
            np.testing.assert_array_equal(actual[key], source[key])
        all_pulses.update(
            sample["source"]["scene_frame_id"]
            for sample in actual["temporal"]["acquisition"]["samples"]
        )
        invalid = deepcopy(actual)
        invalid["config_resolved"]["processing"]["output_selection"]["hop_samples"] = 3
        with pytest.raises(ValueError, match="recipe"):
            output_hop_samples(invalid)
    assert all_pulses == set(range(24))
    for changed_index in (0, 3):
        invalid = deepcopy(saved[0]["_typed_sensing"])
        invalid["temporal"]["acquisition"]["samples"][changed_index]["sample_time_s"] += 0.0001
        with pytest.raises(ValueError, match="uniform window membership"):
            output_hop_samples(invalid)
    invalid = deepcopy(saved[0]["_typed_sensing"])
    for sample in invalid["temporal"]["acquisition"]["samples"]:
        sample["sample_time_s"] += 1 / PRF
    with pytest.raises(ValueError, match="outside.*schedule"):
        output_hop_samples(invalid)
    from visualizer.src.sensing.motion_overview import _postprocessor_policy

    first = saved[0]["_typed_sensing"]
    second = saved[1]["_typed_sensing"]
    descriptor = first["products"]["range_doppler_power_by_pair"]
    policy = _postprocessor_policy(first, descriptor)
    assert policy == _postprocessor_policy(
        second, second["products"]["range_doppler_power_by_pair"]
    )
    for changed_key, value in (
        ("doppler_taper", "different-window"),
        ("coherent_clutter_mode", "different-clutter"),
    ):
        altered = deepcopy(descriptor)
        altered["processing"][changed_key] = value
        assert _postprocessor_policy(first, altered) != policy
    altered = deepcopy(descriptor)
    altered["processing"]["noise"]["pulses"][0]["variance"] = 123
    assert _postprocessor_policy(first, altered) != policy
    altered = deepcopy(descriptor)
    altered["processing"]["source_product_id"] = "0" * 64
    with pytest.raises(ValueError, match="source observation"):
        _postprocessor_policy(first, altered)
    assert baseline == {
        path.name: sha256(path.read_bytes()).hexdigest() for path in saved_windows.iterdir()
    }
    # Ordinary Motion must consume the derived schedule without inventing gaps
    # or treating changing input-observation identities as a new transform.
    from threading import Event

    from visualizer.src.sensing.dashboard_data import DashboardSession, result_context
    from visualizer.src.sensing.motion_overview import (
        SPECTRAL_FAMILIES,
        load_motion_overview,
    )

    provider = Hdf5Provider(output)
    session = DashboardSession(provider)
    try:
        overview = load_motion_overview(
            session.service,
            result_context(saved[0]["_typed_sensing"]),
            "primary",
            frozenset(SPECTRAL_FAMILIES),
            Event(),
        )
        for family in overview.families.values():
            assert family.coverage.loaded_results == 9
            assert family.coverage.state == "complete"
            np.testing.assert_allclose(family.widths_s, 2 / PRF)
    finally:
        session.close()
        provider.close()


def test_missing_selected_window_fails_before_output_claim(saved_windows, tmp_path, monkeypatch):
    provider = Hdf5Provider(saved_windows.parent, frames_subdir=saved_windows.name)
    missing = tmp_path / "missing"
    try:
        with FrameSetWriter.create_new(missing, compression="lzf") as writer:
            for step in provider.list_frames():
                if step != 9:
                    writer.append(provider.load_frame(step))
            writer.finalize()
    finally:
        provider.close()
    output = tmp_path / "must-not-exist"
    monkeypatch.setattr(
        "sys.argv",
        [
            "orchav-sensing",
            str(missing),
            "--output-dir",
            str(output),
            "--streaming",
            "--result-hop-samples",
            "2",
            "--rebuild-rd",
            "--skip-detection",
        ],
    )
    with pytest.raises(SystemExit):
        main()
    assert not output.exists()


@pytest.mark.parametrize("change", ["clock", "pair", "policy", "overlap"])
def test_selection_rejects_incompatible_source_metadata(saved_windows, monkeypatch, change):
    original = Hdf5Provider.iter_frame_projections

    def changed(self, steps, request):
        for projection in original(self, steps, request):
            if projection.frame.frame_index == 9:
                sensing = deepcopy(projection.frame.sensing)
                acquisition = sensing["temporal"]["acquisition"]
                if change == "clock":
                    sensing["temporal"]["clock"]["clock_id"] = "other-clock"
                elif change == "pair":
                    acquisition["pair_id"] = "other-pair"
                elif change == "policy":
                    sensing["config_resolved"]["acquisition"]["phase_contract"] = "other-contract"
                else:
                    acquisition["samples"][0]["source"]["solve_id"] = "other-observation"
                projection = replace(projection, frame=replace(projection.frame, sensing=sensing))
            yield projection

    monkeypatch.setattr(Hdf5Provider, "iter_frame_projections", changed)
    with pytest.raises(ValueError, match="policy|pulse identity"):
        select_saved_windows(saved_windows, 2)


def test_selected_source_change_is_rejected_before_rebinding(saved_windows):
    selected = select_saved_windows(saved_windows, 2)
    frames = iter_frames_sensing(saved_windows, frame_ids=(7,))
    try:
        frame = next(frames)
    finally:
        frames.close()
    frame["_typed_sensing"]["temporal"]["acquisition"]["acquisition_id"] = "changed"
    with pytest.raises(ValueError, match="changed after preflight"):
        selected.bind(frame, 0)


@pytest.mark.parametrize("explicit_selection", [False, True])
def test_native_sparse_replay_retains_only_due_results(tmp_path, monkeypatch, explicit_selection):
    processor = processor_for(samples=8, cpi_hop_samples=4, recording_profile="reprocessing")
    results = bind_results(processor, run_pulses(processor, pulses_for("oscillating", 24)))
    source = tmp_path / "source"
    with FrameSetWriter.create_new(source, compression="lzf") as writer:
        for step, sensing in enumerate(results):
            writer.append(
                replace(
                    _make_standard_frame(frame_idx=step, sensing=sensing),
                    timestamp_s=2 + step / PRF,
                )
            )
        writer.finalize()
    originals = load_frames_sensing(source)
    output = tmp_path / "replayed"
    args = [
        "orchav-sensing",
        str(source),
        "--output-dir",
        str(output),
        "--streaming",
        "--rebuild-rd",
        "--skip-detection",
    ]
    if explicit_selection:
        args += ["--result-hop-samples", "4"]
    monkeypatch.setattr("sys.argv", args)
    main()
    replayed = load_frames_sensing(output)
    assert len(replayed) == (5 if explicit_selection else 24)
    assert sum("result" in frame["_typed_sensing"]["temporal"] for frame in replayed) == 5
    for index, frame in enumerate(replayed):
        source_index = 7 + 4 * index if explicit_selection else index
        original = originals[source_index]["_typed_sensing"]
        actual = frame["_typed_sensing"]
        np.testing.assert_array_equal(
            actual["cfr_by_pair_rx_element"], original["cfr_by_pair_rx_element"]
        )
        assert actual["temporal"]["acquisition"] == original["temporal"]["acquisition"]
        if not explicit_selection and source_index >= 7 and (source_index - 7) % 4:
            assert actual["postprocessing"]["reason"] == "outside_result_cadence"
            assert "result" not in actual["temporal"]


def test_hop_one_complete_unprocessed_acquisition_remains_replayable(saved_windows):
    from shared.sensing.postprocessor import (
        PostProcessingConfig,
        _build_reprocessed_temporal,
        _unprocessed_acquisition_temporal,
    )

    frame = list(iter_frames_sensing(saved_windows, frame_ids=(7,)))[0]
    temporal = frame["_typed_sensing"]["temporal"]
    for key in ("result", "display_binding", "truth"):
        temporal.pop(key, None)
    assert _unprocessed_acquisition_temporal(frame) is None
    replayed = _build_reprocessed_temporal(
        frame,
        processing_run_id="12345678-1234-5678-9234-567812345679",
        postprocessing_config=PostProcessingConfig(skip_detection=True),
        input_kind="cfr_by_pair_rx_element",
    )
    assert (
        replayed["result"]["reference_time_s"]
        == temporal["acquisition"]["default_reference_time_s"]
    )
    assert replayed["acquisition"] == temporal["acquisition"]


@pytest.mark.parametrize(
    ("profile", "expected_products"),
    [
        (None, {"cfr_by_pair_rx_element"}),
        ("results", set()),
        ("maps", {"delay_power_by_pair", "frequency_power_by_pair"}),
        (
            "reprocessing",
            {
                "cfr_by_pair_rx_element",
                "delay_power_by_pair",
                "frequency_power_by_pair",
            },
        ),
    ],
)
def test_incomplete_replay_keeps_legacy_stripping_and_required_retained_axes(
    profile, expected_products
):
    from shared.sensing.postprocessor import PostProcessingConfig
    from shared.sensing.writer import build_incomplete_sensing_block

    processor = processor_for(samples=8, recording_profile=profile)
    source = bind_results(processor, run_pulses(processor, pulses_for("oscillating", 1)))[0]
    assert source["temporal"]["acquisition"]["completeness"] == "incomplete_unprocessed"
    actual = build_incomplete_sensing_block(
        source_sensing=source,
        temporal=source["temporal"],
        postprocessing_config=PostProcessingConfig(skip_detection=True),
    )
    assert set(actual["products"]) == expected_products
    assert "range_profile_by_pair" not in actual
    assert "result" not in actual["temporal"]
    assert actual["temporal"]["acquisition"] == source["temporal"]["acquisition"]
    for key in expected_products:
        np.testing.assert_array_equal(actual[key], source[key])
    if "delay_power_by_pair" in expected_products:
        np.testing.assert_array_equal(actual["range_axis_m"], source["range_axis_m"])
