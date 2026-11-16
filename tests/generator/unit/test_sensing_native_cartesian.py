"""Native Cartesian tracking reuses the saved-observation estimator contract."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest
import yaml

from generator import build_simulation_config
from generator.core.configuration import AntennaConfig, SensingConfig
from generator.core.exceptions import ConfigurationError
from generator.core.sensing.acquisition import (
    resolve_sensing_execution_config,
    resolve_sensing_execution_values,
)
from generator.core.sensing.processor import create_sensing_processor
from shared.extensions.sensing import (
    assert_sensing_contract,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    finalize_sensing_v4_publication,
)
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.providers import Hdf5Provider
from shared.scenarios import load_scenario_configuration, validate_scenario_data
from shared.sensing.loader import load_frames_sensing
from shared.sensing.postprocessor import (
    PostProcessingConfig,
    SensingPostProcessor,
    _build_effective_processing_recipe,
    _build_reprocessed_temporal,
)
from shared.sensing.products import materialize_product_specs
from shared.sensing.range_response import RangeResponseConfig
from shared.sensing.writer import build_reprocessed_sensing_block
from tests.shared.test_sensing_postprocessor import _make_standard_frame


def _config(tracker="cartesian3d", **overrides):
    return SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        range_mode="monostatic",
        fft_size_range=64,
        fft_size_doppler=32,
        bandwidth=100e6,
        prf_hz=1000,
        window_range="boxcar",
        window_doppler="hann",
        coherent_clutter_mode="none",
        range_response=RangeResponseConfig("cfr_ifft", 64, "hann"),
        array_detection_mode="beam_scan",
        beam_angle_oversampling=4,
        raw_payload_policy="always",
        cfar_max_dynamic_range_db=24,
        track_confirmation_enabled=True,
        tracker_type=tracker,
        track_confirmation_m=2,
        track_confirmation_n=3,
        track_association_position_gate_m=2.5,
        track_acceleration_psd_m2_s3=0.4,
        persist_tracking_diagnostics=True,
        **overrides,
    )


def _execution(config, *, elements=16):
    return resolve_sensing_execution_values(
        config,
        cir_time_steps=32,
        cir_sampling_frequency_hz=1000,
        frame_dt_s=0.001,
        rx_elements=elements,
    )


def _sequence(tracker="cartesian3d"):
    config = _config(tracker)
    processor = create_sensing_processor(config, 28e9, _execution(config))

    # The estimator receives sensor geometry only. Any target-truth access fails.
    class SensorOnlyScene:
        simulation_config = SimpleNamespace(rx_antenna=AntennaConfig(num_rows=4, num_cols=4))

        def __getattr__(self, name):
            if name.startswith(("tgt", "target")):
                raise AssertionError("Target truth must not enter estimation")
            raise AttributeError(name)

    generation = str(uuid4())
    for index, time_s in enumerate((0.0, 0.032, 0.064, 0.128, 0.160)):
        rows, cols = np.tile(np.arange(4), 4), np.repeat(np.arange(4), 4)
        direction_y = np.array([0.2 + index * 0.01, -0.3 + index * 0.01])
        direction_z = np.array([0.3, -0.2])
        spatial = np.exp(1j * np.pi * (cols[:, None] * direction_y - rows[:, None] * direction_z))
        slow = np.exp(2j * np.pi * np.array([125, -125])[:, None] * np.arange(32) / 1000)
        amplitudes = spatial[:, :, None] * slow[None, :, :]
        if index == 2:
            amplitudes[:] = 0
        source = processor.process(
            SensorOnlyScene(),
            index,
            None,
            cir_stack=(amplitudes[None, :, None, None], np.array([[[20.3, 32.1]]]) / 100e6),
            source_time_s=time_s,
        )
        source["schema_version"] = 4
        source["temporal"] = build_sensing_temporal_contract(
            clock_id=generation,
            acquisition_id=str(uuid4()),
            acquisition_config_id=canonical_sensing_hash({"clock": generation}),
            pair_id=canonical_sensing_hash({"tx_sensor_id": "tx_0", "rx_sensor_id": "rx_0"}),
            tx_sensor_id="tx_0",
            rx_sensor_id="rx_0",
            samples=[
                {
                    "sample_index": i,
                    "sample_time_s": time_s + i / 1000,
                    "valid": True,
                    "phase_coherent": True,
                    "source": {
                        "source_generation_id": generation,
                        "scene_frame_id": index,
                        "sample_offset": i,
                    },
                }
                for i in range(32)
            ],
            processing_run_id=generation,
            processing_recipe_id=canonical_sensing_hash(
                source["config_resolved"]["processing_recipe"]
            ),
            reference_time_s=time_s,
            reference_time_policy="solve_start",
            display_scene_frame_id=index,
            display_policy="source_scene_exact",
            display_reason="Synthetic sensor-only observation",
        )
        materialize_product_specs(source)
        finalize_sensing_v4_publication(source)
        yield config, source


def _postconfig(config):
    return PostProcessingConfig(
        array_detection_mode="beam_scan",
        beam_angle_oversampling=4,
        cfar_max_dynamic_range_db=24,
        tracker_type=config.tracker_type,
        tracker_m=2,
        tracker_n=3,
        tracker_velocity_gate_m_s=config.track_association_velocity_gate_m_s,
        tracker_xy_gate_m=2.5,
        tracker_acceleration_psd_m2_s3=0.4,
        include_tracking_diagnostics=True,
    )


@pytest.mark.parametrize("tracker", ["cartesian", "cartesian3d"])
def test_native_full_complex_replay_matches_cartesian_states_and_preserves_raw_observations(
    tracker,
):
    replay = None
    seen = []
    for index, (config, source) in enumerate(_sequence(tracker)):
        if replay is None:
            replay = SensingPostProcessor(_postconfig(config))
        actual = replay.process_from_frequency_observation(
            source, update_time_s=source["temporal"]["result"]["reference_time_s"]
        )
        np.testing.assert_allclose(
            actual["rd_cube"], source["range_doppler_by_pair"][0], rtol=2e-13, atol=2e-13
        )
        expected = source["detections"]
        assert len(actual["detections"]) == len(expected)
        for got, want in zip(actual["detections"], expected, strict=True):
            assert got.keys() == want.keys()
            for key in want:
                assert got[key] == pytest.approx(want[key], rel=1e-10, abs=1e-10), key
        native_meta = source["detection_filter_metadata"]["tracker"]
        replay_meta = actual["detection_filter_metadata"]["tracker"]
        assert canonical_sensing_hash(native_meta) == canonical_sensing_hash(replay_meta)
        raw = source["detections_before_tracking"]
        assert all(not any(key.startswith("track_") for key in d) for d in raw)
        assert all(
            {"aoa_az_deg", "aoa_el_deg", "snr_db", "range_bin", "doppler_bin", "x_m", "y_m", "z_m"}
            <= d.keys()
            for d in raw
        )
        assert (
            source["detections_before_tracking_metadata"]["reference_time_s"]
            == source["temporal"]["result"]["reference_time_s"]
        )
        if index in (0, 2):
            assert expected == []
        if index == 0:
            assert len(raw) == 2  # Tentative suppression does not censor fusion inputs.
        elif index == 2:
            assert raw == [] and native_meta["prediction_dt_s"] == pytest.approx(0.032)
        else:
            assert {d["track_id"] for d in expected} == {1, 2}
        seen.extend(expected)
    assert seen


@pytest.mark.parametrize("derived_tracker", ["cartesian3d", "none"])
def test_native_cartesian_metadata_survives_packed_storage_and_is_not_inherited_by_reprocessing(
    tmp_path,
    derived_tracker,
):
    rows = list(_sequence())
    source_dir = tmp_path / "native"
    with FrameSetWriter.create_new(source_dir, compression="lzf") as writer:
        for index, (_cfg, sensing) in enumerate(rows):
            writer.append(
                replace(
                    _make_standard_frame(frame_idx=index, sensing=sensing),
                    rx_positions=np.zeros((1, 3)),
                )
            )
        writer.finalize()
    loaded = load_frames_sensing(source_dir)
    provider = Hdf5Provider(source_dir.parent, frames_subdir=source_dir.name)
    try:
        request = FrameReadRequest(
            components=frozenset({FrameComponent.SENSING, FrameComponent.PROVENANCE}),
            sensing_products=frozenset({"products"}),
        )
        metadata_only = list(provider.iter_frame_projections(range(len(rows)), request))
        for projection, (_cfg, source) in zip(metadata_only, rows, strict=True):
            assert "cfr_by_pair_rx_element" not in projection.frame.sensing
            for key in ("detections_before_tracking", "detections_before_tracking_metadata"):
                assert projection.frame.sensing[key] == source[key]
    finally:
        provider.close()
    for item, (_cfg, source) in zip(loaded, rows, strict=True):
        stored = item["_typed_sensing"]
        for key in (
            "detections",
            "detections_before_tracking",
            "detections_before_tracking_metadata",
        ):
            assert stored[key] == source[key]
        assert_sensing_contract({"sensing": stored}, require_schema_version=True)
    config, source = rows[1]
    postconfig = _postconfig(config)
    postconfig.tracker_type = derived_tracker
    result = SensingPostProcessor(postconfig).process_from_frequency_observation(
        source, update_time_s=0.032
    )
    frame = {"_typed_sensing": source, "_sensing_json": source, "_frame_index": 1}
    result["temporal"] = _build_reprocessed_temporal(
        frame,
        processing_run_id=str(uuid4()),
        postprocessing_config=postconfig,
        input_kind="cfr_by_pair_rx_element",
    )
    result["processing_recipe_resolved"] = _build_effective_processing_recipe(
        frame, postprocessing_config=postconfig, input_kind="cfr_by_pair_rx_element"
    )
    derived = build_reprocessed_sensing_block(
        source_sensing=source,
        result=result,
        postprocessing_config=postconfig,
        input_kind="cfr_by_pair_rx_element",
    )
    assert "detections_before_tracking" not in derived
    assert "detections_before_tracking_metadata" not in derived
    assert source["detections_before_tracking"]
    assert derived["config_resolved"]["track_confirmation_enabled"] is (derived_tracker != "none")
    if derived_tracker == "none":
        assert derived["detections"]
        assert all(not any(key.startswith("track_") for key in d) for d in derived["detections"])
        resolved = derived["config_resolved"]
        assert resolved["tracker_type"] == "none"
        assert "tracking_coordinate_frame" not in resolved
        assert "track_association_position_gate_m" not in resolved
        assert "track_acceleration_psd_m2_s3" not in resolved
        assert resolved["source_processing_recipe"]["tracker_type"] == "cartesian3d"
        assert resolved["source_processing_recipe"]["track_confirmation_enabled"] is True
        assert derived["config"] == resolved
    assert_sensing_contract({"sensing": derived}, require_schema_version=True)


@pytest.mark.parametrize(
    "mutation", ["missing_list", "tracked", "time", "missing_height", "incomplete"]
)
def test_raw_observation_contract_rejects_ambiguous_or_stale_evidence(mutation):
    _config, source = next(_sequence())
    if mutation == "missing_list":
        source.pop("detections_before_tracking")
    elif mutation == "tracked":
        source["detections_before_tracking"][0]["track_x_m"] = 0.0
    elif mutation == "time":
        source["detections_before_tracking_metadata"]["reference_time_s"] = 1.0
    elif mutation == "missing_height":
        source["detections_before_tracking"][0].pop("z_m")
    else:
        source["temporal"].pop("result")
    with pytest.raises(ValueError, match="detections_before_tracking|result"):
        assert_sensing_contract({"sensing": source}, require_schema_version=True)


def test_grouped_native_cartesian_fields_resolve_without_changing_original_yaml(tmp_path):
    root = Path(__file__).resolve().parents[3]
    source = root / "scenarios/sensing/applications/drone_tracking/munich/scenario.yaml"
    data = yaml.safe_load(source.read_text())
    data["sensing"]["tracking"] = {
        "enabled": True,
        "type": "cartesian3d",
        "association": {"position_gate_m": 1.0},
        "acceleration_psd_m2_s3": 0.5,
    }
    validate_scenario_data(data)
    destination = tmp_path / "scenario.yaml"
    destination.write_text(yaml.safe_dump(data))
    configuration = load_scenario_configuration(destination)
    simulation = build_simulation_config(configuration)
    assert simulation.sensing.tracker_type == "cartesian3d"
    assert simulation.sensing.track_association_position_gate_m == 1.0
    assert simulation.sensing.track_acceleration_psd_m2_s3 == 0.5
    resolve_sensing_execution_config(
        simulation, frame_dt_s=1 / 2000, tx_device_count=1, rx_device_count=1
    )
    simulation.rx_antenna.num_rows = 1
    with pytest.raises(ConfigurationError, match="planar"):
        resolve_sensing_execution_config(
            simulation, frame_dt_s=1 / 2000, tx_device_count=1, rx_device_count=1
        )


@pytest.mark.parametrize("mode,elements", [("bistatic", 16), ("monostatic", 1)])
def test_cartesian_requires_localized_array_observations(mode, elements):
    cfg = _config()
    cfg.range_mode = mode
    with pytest.raises(ConfigurationError, match="Cartesian|beam_scan"):
        _execution(cfg, elements=elements)


@pytest.mark.parametrize("tracker", ["m_of_n", "kalman", "jpdaf"])
def test_old_tracker_recipe_omits_new_unused_options(tracker):
    cfg = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        fft_size_doppler=32,
        prf_hz=1000,
        tracker_type=tracker,
    )
    processor = create_sensing_processor(cfg, 28e9, _execution(cfg, elements=1))
    source = processor.process(
        None, 0, None, cir_stack=(np.ones((1, 32), complex), np.array([1e-8])), source_time_s=0
    )
    recipe = source["config_resolved"]["processing_recipe"]
    assert "track_association_position_gate_m" not in recipe
    assert "track_acceleration_psd_m2_s3" not in recipe
    assert "detections_before_tracking" not in source
    assert SensingConfig().tracker_type == "m_of_n"


@pytest.mark.parametrize(
    "field,value",
    [
        ("track_association_position_gate_m", 0.0),
        ("track_association_position_gate_m", float("nan")),
        ("track_acceleration_psd_m2_s3", -1.0),
        ("track_acceleration_psd_m2_s3", True),
    ],
)
def test_cartesian_parameters_fail_before_processing(field, value):
    with pytest.raises(ConfigurationError, match=field):
        SensingConfig(tracker_type="cartesian3d", **{field: value})
    with pytest.raises(ConfigurationError, match="Cartesian"):
        SensingConfig(tracker_type="kalman", **{field: 1.0})


def test_partial_cpi_never_retains_unobserved_cartesian_reports():
    from generator.core.pipeline.sensing import SensingPipelineState

    _cfg, source = next(_sequence())
    SensingPipelineState._strip_unprocessed_result_claims(source)
    assert "detections_before_tracking" not in source
    assert "detections_before_tracking_metadata" not in source
