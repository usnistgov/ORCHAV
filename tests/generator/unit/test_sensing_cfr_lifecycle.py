"""Native strategy, typed persistence and CPU replay qualification (T19-T22)."""

from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from generator.core.configuration import AntennaConfig, SensingConfig
from generator.core.sensing.acquisition import resolve_sensing_execution_values
from generator.core.sensing.processor import create_sensing_processor
from shared.extensions.sensing import (
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    finalize_sensing_v4_publication,
)
from shared.sensing.postprocessor import (
    PostProcessingConfig,
    SensingPostProcessor,
    _build_effective_processing_recipe,
    _build_reprocessed_temporal,
)
from shared.sensing.products import materialize_product_specs
from shared.sensing.range_response import FrequencyNoise, RangeResponseConfig
from shared.sensing.writer import build_reprocessed_sensing_block


def native_cfr(
    *,
    elements=1,
    rows=1,
    fft=64,
    window="boxcar",
    noise=0.0,
    policy="always",
    clutter="legacy",
    detection=False,
    profile=False,
    carrier=28e9,
    spacing=0.5,
    direction=None,
    array_mode="rd",
    model="cfr_ifft",
    samples=16,
    doppler_window="hann",
    recording=None,
    output_rd=True,
    tracker="m_of_n",
    tracking=False,
):
    cfg = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        bandwidth=100e6,
        prf_hz=1000,
        fft_size_range=fft,
        fft_size_doppler=samples,
        window_range="boxcar",
        window_doppler=doppler_window,
        range_mode="monostatic",
        raw_payload_policy=policy,
        recording_profile=recording,
        output_range_doppler=output_rd,
        output_detections=detection,
        output_range_profile=profile,
        tracker_type=tracker,
        track_confirmation_enabled=tracking,
        coherent_clutter_mode=clutter,
        range_response=(
            RangeResponseConfig("cfr_ifft", 64, window, FrequencyNoise(noise, 123))
            if model == "cfr_ifft"
            else RangeResponseConfig()
        ),
        persist_detection_stage_artifacts=detection,
        array_detection_mode=array_mode,
    )
    execution = resolve_sensing_execution_values(
        cfg,
        cir_time_steps=samples,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )
    processor = create_sensing_processor(cfg, carrier, execution)
    a = (
        np.exp(1j * np.arange(elements) * 0.7)[:, None, None]
        * np.exp(2j * np.pi * 125 * np.arange(samples) / 1000)[None, None, :]
    )
    if direction is not None:
        uy, uz = direction
        a = (
            np.array(
                [
                    np.exp(2j * np.pi * spacing * (col * uy - row * uz)) + 0.05
                    for col in range(elements // rows)
                    for row in range(rows)
                ]
            )[:, None, None]
            * np.exp(2j * np.pi * 123 * np.arange(samples) / 1000)[None, None, :]
        )
    simulation = SimpleNamespace(
        simulation_config=SimpleNamespace(
            rx_antenna=AntennaConfig(
                num_rows=rows,
                num_cols=elements // rows,
                horizontal_spacing=spacing,
                vertical_spacing=spacing,
            )
        )
    )
    sensing = processor.process(
        simulation,
        0,
        None,
        cir_stack=(a[None, :, None, None], np.array([[[20.3 / 100e6]]])),
        source_time_s=0.0,
    )
    sensing["schema_version"] = 4
    generation = str(uuid4())
    sensing["temporal"] = build_sensing_temporal_contract(
        clock_id=generation,
        acquisition_id=str(uuid4()),
        acquisition_config_id=canonical_sensing_hash({"clock": generation}),
        pair_id=canonical_sensing_hash({"tx_sensor_id": "tx", "rx_sensor_id": "rx"}),
        tx_sensor_id="tx",
        rx_sensor_id="rx",
        samples=[
            {
                "sample_index": i,
                "sample_time_s": i / 1000,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": generation,
                    "scene_frame_id": 0,
                    "sample_offset": i,
                },
            }
            for i in range(samples)
        ],
        processing_run_id=str(uuid4()),
        processing_recipe_id=canonical_sensing_hash(
            sensing["config_resolved"]["processing_recipe"]
        ),
        reference_time_s=0.0,
        reference_time_policy="solve_start",
        display_scene_frame_id=0,
        display_policy="source_scene_exact",
        display_reason="shared analytical acquisition",
    )
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    return sensing


@pytest.mark.parametrize("elements,rows", [(16, 1), (32, 4)])
@pytest.mark.parametrize("model", ["point_bin_floor", "cfr_ifft"])
def test_saved_rd_replay_keeps_native_joint_array_directions(elements, rows, model):
    from scripts.benchmarks.benchmark_cfar_replay import saved_processor

    source = native_cfr(
        elements=elements,
        rows=rows,
        model=model,
        detection=True,
        direction=(0.25, 0.5 if rows > 1 else 0.0),
        clutter="none",
    )
    processor = saved_processor(source)
    processor = SensingPostProcessor(
        replace(
            processor.config, range_response=None, range_ifft_size=None, coherent_clutter_mode=None
        )
    )
    power = abs(source["range_doppler_by_pair"][0]) ** 2
    result = processor.process_from_rd_map(
        power,
        source,
        range_axis_m=source["range_axis_m"],
        doppler_axis_hz=source["doppler_axis_hz"],
    )
    assert source["detections"]
    assert len(result["detections"]) == len(source["detections"])
    keys = ("range_m", "velocity_m_s", "aoa_az_deg", "snr_db")
    if rows > 1:
        keys += ("aoa_el_deg",)
    for actual, expected in zip(result["detections"], source["detections"]):
        for key in keys:
            assert actual[key] == pytest.approx(expected[key], rel=2e-7, abs=1e-7)
    assert result["angular_input_product_id"] in {
        p["product_id"] for p in source["products"].values() if p["kind"] == "range_doppler_angle"
    }
    with pytest.raises(ValueError, match="coordinate grids must agree"):
        processor.process_from_rd_map(
            power,
            source,
            range_axis_m=source["range_axis_m"] + 0.1,
            doppler_axis_hz=source["doppler_axis_hz"],
        )


@pytest.mark.parametrize("elements", [1, 4])
@pytest.mark.parametrize(
    "window,fft,clutter",
    [("boxcar", 64, "legacy"), ("hann", 127, "windowed_mean"), ("boxcar", 128, "none")],
)
def test_T20_native_replay_complex_parity(elements, window, fft, clutter):
    source = native_cfr(elements=elements, window=window, fft=fft, noise=0.001, clutter=clutter)
    result = SensingPostProcessor(
        PostProcessingConfig(skip_detection=True)
    ).process_from_frequency_observation(source)
    np.testing.assert_allclose(
        result["rd_cube"], source["range_doppler_by_pair"][0], rtol=2e-13, atol=2e-13
    )
    np.testing.assert_array_equal(
        result["cfr_rebuilt"]["observation"], source["cfr_by_pair_rx_element"]
    )
    np.testing.assert_array_equal(result["range_axis"], source["range_axis_m"])


def test_two_sample_cfr_hann_rejects_silent_signal_erasure():
    with pytest.raises(ValueError, match="zero energy"):
        native_cfr(samples=2)
    source = native_cfr(samples=2, doppler_window="boxcar")
    assert np.max(abs(source["range_doppler_by_pair"])) > 0


@pytest.mark.parametrize("elements", [1, 4])
@pytest.mark.parametrize("clutter", ["none", "legacy", "windowed_mean"])
def test_T19_T22_rewindow_persisted_lineage_and_uncertainty(tmp_path, elements, clutter):
    from shared.sensing.loader import load_frames_sensing
    from tests.shared.test_sensing_postprocessor import (
        _make_standard_frame,
        _write_packed_frame_set,
    )

    source = native_cfr(noise=0.01, detection=True, elements=elements)
    directory = tmp_path / "source"
    _write_packed_frame_set(directory, _make_standard_frame(frame_idx=0, sensing=source))
    import hashlib

    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}
    frame = load_frames_sensing(directory)[0]
    loaded = frame["_typed_sensing"]
    from shared.sensing.postprocessor import _source_sensing_payload

    np.testing.assert_array_equal(
        _source_sensing_payload(frame)["cfr_by_pair_rx_element"], source["cfr_by_pair_rx_element"]
    )
    np.testing.assert_array_equal(
        loaded["cfr_by_pair_rx_element"], source["cfr_by_pair_rx_element"]
    )
    config = PostProcessingConfig(
        range_response=replace(
            RangeResponseConfig.parse(source["config_resolved"]["range_response"]),
            frequency_window="hann",
        ),
        range_ifft_size=256,
        coherent_clutter_mode=clutter,
        skip_detection=True,
    )
    result = SensingPostProcessor(config).process_from_frequency_observation(loaded)
    key = "cfr_by_pair_rx_element"
    result["processing_recipe_resolved"] = _build_effective_processing_recipe(
        frame, postprocessing_config=config, input_kind=key
    )
    result["temporal"] = _build_reprocessed_temporal(
        frame, processing_run_id=str(uuid4()), postprocessing_config=config, input_kind=key
    )
    derived = build_reprocessed_sensing_block(
        source_sensing=loaded, result=result, postprocessing_config=config, input_kind=key
    )
    assert derived["config_resolved"]["coherent_clutter_mode"] == clutter
    assert (
        derived["config_resolved"]["processing_recipe"]["effective_overrides"][
            "coherent_clutter_mode"
        ]
        == clutter
    )
    assert derived["config_resolved"]["fft_size_range"] == 256
    assert (
        derived["config_resolved"]["range_resolution_m"]
        == source["config_resolved"]["range_resolution_m"]
    )
    assert derived["range_axis_m"][1] == pytest.approx(source["range_axis_m"][1] / 4)
    np.testing.assert_array_equal(derived[key], source[key])
    _write_packed_frame_set(
        tmp_path / "derived", _make_standard_frame(frame_idx=0, sensing=derived)
    )
    second = load_frames_sensing(tmp_path / "derived")[0]["_typed_sensing"]
    power_recipe = second["products"]["range_doppler_power_by_pair"]["processing"]
    complex_recipe = second["products"]["range_doppler_complex_by_pair"]["processing"]
    raw_recipe = second["products"]["range_doppler_raw_by_pair"]["processing"]
    assert power_recipe == complex_recipe
    assert power_recipe["coherent_clutter_mode"] == clutter
    assert power_recipe["slow_time_mean_removal_applied"] == (clutter != "none")
    assert raw_recipe == {**power_recipe, "slow_time_mean_removal_applied": False}
    for name in ("range_profile_by_pair", "slow_time_by_pair_rx_element"):
        recipe = second["products"][name]["processing"]
        assert recipe["frequency_window"] == "hann"
        assert recipe["slow_time_mean_removal_applied"] is False
        assert recipe["doppler_window_applied"] is False
        assert recipe["doppler_fft_applied"] is False
        assert "doppler_taper" not in recipe
    np.testing.assert_array_equal(
        second["range_doppler_power_by_pair"],
        abs(second["range_doppler_complex_by_pair"]) ** 2,
    )
    # A loaded derived frame must describe its own window/grid everywhere,
    # including native top-level annotations copied into the new frame.
    expected_spacing = source["range_response"]["range_sample_spacing_m"] / 4
    expected_delay_spacing = source["range_response"]["delay_sample_spacing_s"] / 4
    for metadata in (
        second,
        second["range_response"],
        second["config_resolved"]["range_response_processing"],
    ):
        assert metadata["frequency_window"] == "hann"
        assert metadata["fft_size_range"] == 256
        assert metadata["range_sample_spacing_m"] == pytest.approx(expected_spacing)
        assert metadata["delay_sample_spacing_s"] == pytest.approx(expected_delay_spacing)
        assert (
            metadata["range_resolution_scale_m"]
            == source["range_response"]["range_resolution_scale_m"]
        )
        assert (
            metadata["measurement_uncertainty_policy"]
            == source["range_response"]["measurement_uncertainty_policy"]
        )
    assert second["coherent_gain"] == 32
    assert second["noise_gain"] == pytest.approx(1.5 / 64)
    if elements > 1:
        from shared.sensing.angle import select_range_doppler_angle_product

        assert select_range_doppler_angle_product(second) is not None
    replay = SensingPostProcessor(config).process_from_frequency_observation(second)
    np.testing.assert_array_equal(replay["rd_cube"], result["rd_cube"])
    inherited = SensingPostProcessor(
        PostProcessingConfig(skip_detection=True)
    ).process_from_frequency_observation(second)
    np.testing.assert_array_equal(inherited["rd_cube"], result["rd_cube"])
    assert hashes == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def test_T21_missing_retention_changed_grid_carrier():
    from shared.sensing.cfr_replay import rebuild_frequency_observation

    with pytest.raises(ValueError, match="requires retained"):
        rebuild_frequency_observation(native_cfr(policy="none"))
    for key in ("bandwidth", "carrier_freq_hz", "prf_hz"):
        changed = native_cfr()
        changed["config_resolved"][key] *= 2
        with pytest.raises(ValueError, match="[Cc]hanged"):
            rebuild_frequency_observation(changed)


def test_T21_explicit_array_delays_rejected():
    from generator.core.sensing.strategies.coherent_cpi_fft import _cfr_shared_delays

    with pytest.raises(ValueError, match="element-dependent"):
        _cfr_shared_delays(np.zeros((1, 4, 1, 1, 2)), 2, 4)


def test_T08_adapter_does_not_rotate_existing_baseband_again():
    low = native_cfr(carrier=24e9)
    high = native_cfr(carrier=60e9)
    np.testing.assert_array_equal(low["cfr_by_pair_rx_element"], high["cfr_by_pair_rx_element"])
    np.testing.assert_array_equal(low["range_doppler_by_pair"], high["range_doppler_by_pair"])


@pytest.mark.parametrize("rows", [1, 2])
def test_T15_T20_array_spatial_replay_and_element_identity(rows):
    source = native_cfr(elements=4, rows=rows, detection=True)
    result = SensingPostProcessor(PostProcessingConfig()).process_from_frequency_observation(source)
    np.testing.assert_allclose(
        abs(result["cfr_rebuilt"]["spatial"].spectrum) ** 2,
        source["range_doppler_angle_by_pair"][0],
        rtol=1e-6,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        result["cfr_rebuilt"]["observation"], source["cfr_by_pair_rx_element"]
    )


def test_T22_padded_detection_coordinates_and_fixed_tracker_uncertainty():
    source = native_cfr(detection=True)
    processors = []
    for size in (64, 256):
        config = PostProcessingConfig(
            range_ifft_size=size,
            tracker_type="kalman",
            tracker_m=1,
            cfar_min_snr_db=0,
            include_stage_artifacts=True,
        )
        processor = SensingPostProcessor(config)
        result = processor.process_from_frequency_observation(source, update_time_s=0.0)
        assert result["detections"]
        for detection in result["detections"]:
            assert detection["range_m"] == result["range_axis"][detection["range_bin"]]
            assert detection["doppler_hz"] == result["doppler_axis"][detection["doppler_bin"]]
        processors.append(processor)
    np.testing.assert_array_equal(processors[0]._tracker._R, processors[1]._tracker._R)
    np.testing.assert_array_equal(processors[0]._tracker._R, np.diag([0.25, 0.1]))


def test_T21_invalid_generation_fails_in_preflight():
    from generator.core.exceptions import ConfigurationError

    for bad in (
        dict(processing_mode="analytic_path_doppler_oracle"),
        dict(noise_enabled=True),
        dict(window_range="hann"),
    ):
        with pytest.raises((ValueError, ConfigurationError), match="cfr_ifft"):
            SensingConfig(
                **{
                    "range_response": RangeResponseConfig("cfr_ifft", 64),
                    "window_range": "boxcar",
                    **bad,
                }
            )
    cfg = SensingConfig(
        range_response=RangeResponseConfig("cfr_ifft", 64),
        window_range="boxcar",
        fft_size_doppler=16,
        prf_hz=1000,
    )
    with pytest.raises((ValueError, ConfigurationError), match="[Tt][Xx]|transmit|one"):
        resolve_sensing_execution_values(
            cfg, cir_time_steps=16, cir_sampling_frequency_hz=1000, frame_dt_s=0.001, tx_elements=2
        )


def test_T21_range_overrides_cannot_be_ignored_by_power_api():
    source = native_cfr()
    processor = SensingPostProcessor(PostProcessingConfig(range_ifft_size=256))
    with pytest.raises(ValueError, match="retained CFR"):
        processor.process_from_rd_map(
            abs(source["range_doppler_by_pair"][0]) ** 2,
            source,
            range_axis_m=source["range_axis_m"],
        )


@pytest.mark.parametrize(
    "override", [dict(noise_enabled=True), dict(noise_snr_db=10), dict(window_range="hann")]
)
def test_T18_conflicting_legacy_settings_rejected_for_saved_cfr_power(override):
    source = native_cfr()
    with pytest.raises(ValueError, match="CFR"):
        SensingPostProcessor(PostProcessingConfig(**override)).process_from_rd_map(
            abs(source["range_doppler_by_pair"][0]) ** 2,
            source,
            range_axis_m=source["range_axis_m"],
        )


@pytest.mark.parametrize("spacing", [0.25, 0.4, 0.5])
@pytest.mark.parametrize("mode,skip", [("rd", True), ("rd", False), ("beam_scan", False)])
def test_compact_array_native_and_loaded_cfr_replay(tmp_path, spacing, mode, skip):
    from shared.sensing.loader import load_frames_sensing
    from tests.shared.test_sensing_postprocessor import (
        _make_standard_frame,
        _write_packed_frame_set,
    )

    source = native_cfr(
        elements=16, rows=4, spacing=spacing, noise=0.001, detection=True, array_mode=mode
    )
    _write_packed_frame_set(tmp_path / "input", _make_standard_frame(frame_idx=0, sensing=source))
    loaded = load_frames_sensing(tmp_path / "input")[0]["_typed_sensing"]
    result = SensingPostProcessor(
        PostProcessingConfig(
            skip_detection=skip,
            array_detection_mode=mode,
        )
    ).process_from_frequency_observation(loaded)
    np.testing.assert_array_equal(
        result["cfr_rebuilt"]["observation"], source["cfr_by_pair_rx_element"]
    )
    np.testing.assert_allclose(
        result["rd_cube"], source["range_doppler_by_pair"][0], atol=2e-13, rtol=2e-13
    )
    if not skip:
        assert len(result["detections"]) == len(source["detections"])
        for actual, expected in zip(result["detections"], source["detections"]):
            for field in ("range_m", "doppler_hz", "aoa_az_deg", "aoa_el_deg"):
                assert actual[field] == pytest.approx(expected[field], abs=1e-5)


@pytest.mark.parametrize("mode", ["rd", "beam_scan"])
def test_elevated_cfr_native_and_cpu_replay_report_physical_azimuth(mode):
    source = native_cfr(elements=16, rows=4, direction=(0.5, 0.5), detection=True, array_mode=mode)
    replay = SensingPostProcessor(
        PostProcessingConfig(array_detection_mode=mode)
    ).process_from_frequency_observation(source)
    for results in (source, replay):
        assert results["detections"]
        peak = max(results["detections"], key=lambda d: d["power_linear"])
        assert peak["aoa_az_deg"] == pytest.approx(35.264389682754654, abs=1e-5)
        assert peak["aoa_el_deg"] == pytest.approx(30.0, abs=1e-5)
        np.testing.assert_allclose(
            [peak["x_m"], peak["y_m"], peak["z_m"]],
            peak["range_m"] * np.array([np.sqrt(0.5), 0.5, 0.5]),
            atol=1e-5,
        )


@pytest.mark.parametrize(
    "option",
    [
        dict(vote_fusion_enabled=True),
        dict(angle_peak_detection_enabled=True),
        dict(ghost_filter_azimuth_gate_deg=15.0),
    ],
)
def test_deferred_typed_algorithms_fail_consistently_before_cfr_processing(option):
    source = native_cfr(elements=4, detection=True)
    processor = SensingPostProcessor(PostProcessingConfig(**option))
    for call in (
        lambda: processor.process_from_frequency_observation(source),
        lambda: processor.process_from_rd_map(
            abs(source["range_doppler_by_pair"][0]) ** 2,
            source,
            range_axis_m=source["range_axis_m"],
        ),
        lambda: processor.process_from_slow_time_matrix(
            source["slow_time_by_pair_rx_element"][0, 0],
            source,
            range_axis_m=source["range_axis_m"],
        ),
    ):
        with pytest.raises(ValueError, match="typed.*postprocessing not implemented"):
            call()


@pytest.mark.parametrize(
    "figure,expected_x",
    [
        ("create_range_azimuth_figure", 30.0),
        ("create_doppler_azimuth_figure", 1.0),
        ("create_azimuth_elevation_figure", 30.0),
    ],
)
def test_compact_array_figures_project_spherical_reports_without_mutating_them(
    tmp_path, monkeypatch, figure, expected_x
):
    from matplotlib.axes import Axes

    from shared.sensing.plotting import viz as sensing_viz

    source = native_cfr(elements=16, rows=4, spacing=0.4)
    source["detections"] = [
        {"range_m": 30.0, "velocity_m_s": 1.0, "aoa_az_deg": 35.264389682754654, "aoa_el_deg": 30.0}
    ]
    offsets = []
    scatter = Axes.scatter

    def record_scatter(ax, x, y, *args, **kwargs):
        offsets.append((np.asarray(x), np.asarray(y)))
        return scatter(ax, x, y, *args, **kwargs)

    monkeypatch.setattr(Axes, "scatter", record_scatter)
    path = tmp_path / f"{figure}.png"
    if figure == "create_azimuth_elevation_figure":
        getattr(sensing_viz, figure)(source, path)
    else:
        getattr(sensing_viz, figure)(source, None, path)
    assert path.is_file() and offsets
    np.testing.assert_allclose(offsets[0][0], [expected_x], atol=1e-6)
    np.testing.assert_allclose(offsets[0][1], [30.0], atol=1e-6)
    assert source["detections"][0]["aoa_az_deg"] == 35.264389682754654
