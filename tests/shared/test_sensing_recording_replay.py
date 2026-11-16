"""Explicit analysis of canonical-frequency recordings preserves source data."""

import hashlib
import sys

import numpy as np
import pytest

from shared.extensions.sensing import assert_sensing_contract
from shared.sensing.cfr_replay import reconstruct_delay_observation
from shared.sensing.loader import load_frames_sensing
from shared.sensing.plotting.figures import load_cir_sequence
from shared.sensing.postprocessor import (
    PostProcessingConfig,
    SensingPostProcessor,
    _select_typed_detection_input,
    main,
)
from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr
from tests.shared.test_sensing_postprocessor import _make_standard_frame, _write_packed_frame_set


def test_reconstructed_delay_is_typed_exact_and_does_not_mutate_source():
    source = native_cfr(recording="reprocessing", elements=4)
    expected = native_cfr(elements=4)
    keys = tuple(source)
    view = reconstruct_delay_observation(source)
    assert_sensing_contract(
        {"sensing": view}, require_schema_version=True, require_described_products=True
    )
    np.testing.assert_array_equal(
        view["slow_time_by_pair_rx_element"], expected["slow_time_by_pair_rx_element"]
    )
    assert tuple(source) == keys
    descriptor = view["products"]["slow_time_by_pair_rx_element"]
    assert descriptor["lineage"]["source_product_ids"] == [
        source["products"]["cfr_by_pair_rx_element"]["product_id"]
    ]


def test_default_processing_uses_saved_map_while_rebuild_uses_canonical_cfr(tmp_path):
    source = native_cfr(recording="reprocessing")
    _write_packed_frame_set(tmp_path / "source", _make_standard_frame(frame_idx=0, sensing=source))
    frame = load_frames_sensing(tmp_path / "source")[0]
    assert _select_typed_detection_input(frame, rebuild_rd=False).kind == "range_doppler"
    assert (
        _select_typed_detection_input(frame, rebuild_rd=True).kind
        == "baseband_frequency_observation"
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_cli_rebuilds_canonical_frequency_recording(tmp_path, streaming, monkeypatch):
    source = native_cfr(recording="reprocessing", profile=True)
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, _make_standard_frame(frame_idx=0, sensing=source))
    before = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }
    argv = [
        str(directory),
        "--rebuild-rd",
        "--skip-detection",
        "--output-dir",
        str(tmp_path / "derived"),
    ]
    if streaming:
        argv.append("--streaming")
    monkeypatch.setattr(sys, "argv", ["orchav-sensing", *argv])
    main()
    result = load_frames_sensing(tmp_path / "derived" / "frames")[0]["_typed_sensing"]
    np.testing.assert_allclose(
        result["range_doppler_power_by_pair"],
        source["range_doppler_power_by_pair"],
        rtol=1e-12,
        atol=1e-30,
    )
    assert before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.iterdir()
        if p.is_file()
    }


def test_micro_doppler_sequence_reconstructs_retained_cfr(tmp_path):
    source = native_cfr(recording="reprocessing", elements=4)
    expected = native_cfr(elements=4)
    _write_packed_frame_set(tmp_path / "source", _make_standard_frame(frame_idx=0, sensing=source))
    rows, metadata = load_cir_sequence(tmp_path / "source")
    np.testing.assert_array_equal(
        np.stack(rows), expected["slow_time_by_pair_rx_element"][0].sum(axis=0) / 2
    )
    np.testing.assert_array_equal(metadata["range_axis_m"], expected["range_axis_m"])
    assert len(metadata["sample_times_s"]) == 16


def test_angle_refinement_accepts_canonical_frequency_observations():
    source = native_cfr(
        recording="reprocessing", elements=4, rows=2, array_mode="beam_scan", direction=(0.2, 0.15)
    )
    refiner, metadata = SensingPostProcessor(
        PostProcessingConfig(array_detection_mode="beam_scan", beam_angle_oversampling=4)
    )._beam_angle_refiner(source)
    assert callable(refiner)
    assert (
        metadata["source_product_id"] == source["products"]["cfr_by_pair_rx_element"]["product_id"]
    )
    assert "slow_time_by_pair_rx_element" not in source


def _frame_set_hashes(directory):
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in directory.iterdir()
        if path.is_file()
    }


@pytest.mark.parametrize("profile", ["reprocessing", "full"])
def test_cli_rewindowed_beam_stages_bind_exact_rebuilt_angular_parent(
    tmp_path, monkeypatch, profile
):
    source = native_cfr(
        recording=profile, elements=4, rows=2, array_mode="beam_scan", direction=(0.2, 0.15)
    )
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, _make_standard_frame(frame_idx=0, sensing=source))
    before = _frame_set_hashes(directory)
    outputs = []
    for save_stages in (False, True):
        output = tmp_path / f"derived-{save_stages}"
        argv = [
            "orchav-sensing",
            str(directory),
            "--frequency-window",
            "hann",
            "--array-detection-mode",
            "beam_scan",
            "--pre-cfar-doppler-edge-guard-bins",
            "0",
            "--pre-cfar-doppler-dc-guard-bins",
            "0",
            "--output-dir",
            str(output),
        ]
        if save_stages:
            argv.append("--save-stage-artifacts")
        monkeypatch.setattr(sys, "argv", argv)
        main()
        outputs.append(load_frames_sensing(output / "frames")[0]["_typed_sensing"])

    control, result = outputs
    np.testing.assert_array_equal(
        result["range_doppler_power_by_pair"], control["range_doppler_power_by_pair"]
    )
    np.testing.assert_equal(result["detections"], control["detections"])
    np.testing.assert_array_equal(
        result["cfr_by_pair_rx_element"], source["cfr_by_pair_rx_element"]
    )
    parent = result["products"]["range_doppler_angle_by_pair"]
    assert parent["product_id"] != source["products"]["range_doppler_angle_by_pair"]["product_id"]
    assert parent["processing"]["frequency_window"] == "hann"
    assert parent["lineage"]["source_product_ids"] == [
        source["products"]["cfr_by_pair_rx_element"]["product_id"]
    ]
    assert result["detection_filter_metadata"]["angular_input_product_id"] == parent["product_id"]
    stages = [
        (key, descriptor)
        for key, descriptor in result["products"].items()
        if descriptor["processing"].get("execution")
        == "independently_per_visible_beam_before_projection"
    ]
    assert len(stages) == 10
    for key, descriptor in stages:
        assert key in result
        assert parent["product_id"] in descriptor["lineage"]["source_product_ids"]
    detector_input = result["products"]["range_doppler_detection_input_power_by_pair"]
    assert detector_input["lineage"]["source_product_ids"] == [parent["product_id"]]
    visible = (
        result["elevation_direction_cosine"][:, None] ** 2
        + result["azimuth_direction_cosine"][None, :] ** 2
        <= 1.0
    )
    # With explicitly disabled Doppler guards, the projected input must be
    # exactly the strongest physical beam in the persisted angular parent.
    expected = result["range_doppler_angle_by_pair"][0][visible].max(axis=0)
    np.testing.assert_array_equal(
        result["range_doppler_detection_input_power_by_pair"][0], expected
    )
    assert before == _frame_set_hashes(directory)


@pytest.mark.parametrize("profile", ["reprocessing", "full", None])
@pytest.mark.parametrize("streaming", [False, True])
def test_cli_beam_angle_refinement_preserves_signal_and_retained_source_identity(
    tmp_path, monkeypatch, profile, streaming
):
    source = native_cfr(
        recording=profile, elements=4, rows=2, array_mode="beam_scan", direction=(0.2, 0.15)
    )
    power = (
        source["range_doppler_power_by_pair"]
        if profile is not None
        else abs(source["range_doppler_by_pair"]) ** 2
    )
    coarse = SensingPostProcessor(
        PostProcessingConfig(array_detection_mode="beam_scan")
    ).process_from_rd_map(
        power[0],
        source,
        range_axis_m=source["range_axis_m"],
        doppler_axis_hz=source["doppler_axis_hz"],
    )
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, _make_standard_frame(frame_idx=0, sensing=source))
    before = _frame_set_hashes(directory)
    output = tmp_path / "refined"
    argv = [
        "orchav-sensing",
        str(directory),
        "--array-detection-mode",
        "beam_scan",
        "--beam-angle-oversampling",
        "4",
        "--output-dir",
        str(output),
    ]
    if streaming:
        argv.append("--streaming")
    monkeypatch.setattr(sys, "argv", argv)
    main()
    result = load_frames_sensing(output / "frames")[0]["_typed_sensing"]
    np.testing.assert_array_equal(result["range_doppler_power_by_pair"], power)
    metadata = result["detection_filter_metadata"]["angle_estimation"]
    source_key = "cfr_by_pair_rx_element" if profile is not None else "slow_time_by_pair_rx_element"
    assert metadata["input"] == source_key
    assert metadata["source_product_id"] == source["products"][source_key]["product_id"]
    assert metadata["delay_reconstructed"] is (profile is not None)
    assert metadata["oversampling"] == 4
    refined = result["detections"]
    assert len(refined) == len(coarse["detections"]) > 0
    for initial, final in zip(coarse["detections"], refined):
        for key in (
            "range_m",
            "velocity_m_s",
            "power_linear",
            "beam_index",
            "range_bin",
            "doppler_bin",
        ):
            assert final[key] == initial[key]
        assert final["coarse_aoa_az_deg"] == initial["aoa_az_deg"]
        true_angle = np.degrees(np.arcsin(0.2))
        assert abs(final["aoa_az_deg"] - true_angle) < abs(initial["aoa_az_deg"] - true_angle)
    assert before == _frame_set_hashes(directory)


@pytest.mark.parametrize(
    "flags, message",
    [
        (["--rebuild-rd"], "beam_scan uses retained angular products"),
        (["--frequency-window", "hann", "--streaming"], "CFR rewindowing requires"),
    ],
)
def test_cli_beam_replay_keeps_unsupported_reconstruction_guards(
    tmp_path, monkeypatch, capsys, flags, message
):
    source = native_cfr(recording="reprocessing", elements=4, rows=2, array_mode="beam_scan")
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, _make_standard_frame(frame_idx=0, sensing=source))
    output = tmp_path / "unsupported"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "orchav-sensing",
            str(directory),
            "--array-detection-mode",
            "beam_scan",
            *flags,
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_motion_analysis_reads_sequential_canonical_cfr_without_duplicate_epochs(tmp_path):
    from shared.frames.frame_set_writer import FrameSetWriter
    from shared.sensing.motion_analysis import load_motion_recording
    from tests.shared.test_sensing_sequential_cfr import (
        bind_results,
        processor_for,
        pulses_for,
        run_pulses,
    )

    pulses = pulses_for("approaching", 12)
    reference = run_pulses(processor_for(samples=8), pulses)
    processor = processor_for(samples=8, recording_profile="reprocessing")
    results = bind_results(processor, run_pulses(processor, pulses))
    directory = tmp_path / "observations"
    with FrameSetWriter.create_new(directory, compression="lzf") as writer:
        for index, sensing in enumerate(results):
            writer.append(_make_standard_frame(frame_idx=index, sensing=sensing))
        writer.finalize()
    samples, times, _, provenance = load_motion_recording(directory)
    np.testing.assert_array_equal(
        samples, np.stack([r["cir_by_pair_rx_element"][0, 0] for r in reference])
    )
    np.testing.assert_allclose(times, 2 + np.arange(12) / 1000, rtol=0, atol=1e-15)
    assert provenance["source_kind"] == "baseband_frequency_observation"
    assert provenance["delay_reconstructed"] is True
