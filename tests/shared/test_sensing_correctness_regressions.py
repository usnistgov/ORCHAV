"""Cross-boundary scientific contracts uncovered by the v0.2 release audit."""

from copy import deepcopy

import numpy as np
import pytest

from shared.extensions.sensing import assert_sensing_contract, canonical_sensing_hash
from shared.sensing.angular import angular_event, angular_residuals
from shared.sensing.detection.angle import augment_detections_with_sensor_frame_xy
from shared.sensing.detection.pipeline import DetectionPipelineConfig, run_detection_pipeline
from shared.sensing.fusion.core import FusionConfig
from shared.sensing.fusion.quality import apply_quality, native_observation
from shared.sensing.postprocessor import PostProcessingConfig, SensingPostProcessor
from shared.sensing.replay import ReplaySceneContext, build_replay_event
from shared.sensing.signal_processing import RangeDopplerResult
from shared.sensing.tracker import (
    CartesianKalmanTracker,
    TrackerConfig,
    prune_correlated_ghost_tracks,
)
from shared.sensing.writer import build_reprocessed_sensing_block
from tests.shared import test_sensing_postprocessor as pp
from tests.shared import test_sensing_replay as replay
from tests.shared.test_sensing_angular import payload


@pytest.mark.parametrize("mode", ["monostatic", "bistatic"])
@pytest.mark.parametrize("velocity", [-3.0, 0.0, 2.0])
def test_angular_association_uses_common_velocity_observable(mode, velocity):
    s = payload()
    s["config_resolved"]["range_mode"] = mode
    observable = s["truth"]["targets"][0]["pair_observables"][0]
    observable.update(
        bistatic_path_length_m=10.0,
        path_closing_rate_m_s=2 * velocity,
        monostatic_equivalent_velocity_m_s=velocity,
    )
    s["detections"] = [
        dict(
            range_m=5.0 if mode == "monostatic" else 10.0,
            velocity_m_s=velocity,
            aoa_az_deg=179.0,
            aoa_el_deg=3.0,
        )
    ]
    result = angular_residuals([angular_event(s)])
    assert len(result["matches"]) == 1
    assert result["matches"][0]["azimuth_error_deg"] == 0.0
    assert result["unmatched_reports"] == result["missed_targets"] == 0


@pytest.mark.parametrize(
    "tx,rx,target",
    [
        ([-10.0, 0, 0], [0.0, 0, 0], [10.0, 0, 0]),
        ([0.0, 20, 5], [10.0, 20, 5], [20.0, 20, 5]),
        ([0.0, 0, 0], [0.0, 0, 0], [15.0, 0, 0]),
        ([0.0, -10, 0], [0.0, 0, 0], [0.0, 10, 0]),
        ([0.0, 0, 0], [10.0, 0, 0], [5.0, 0, 0]),
    ],
)
def test_bistatic_cartesian_requires_localization_instead_of_using_total_path(tx, rx, target):
    tx, rx, target = map(np.asarray, (tx, rx, target))
    total_path = np.linalg.norm(target - tx) + np.linalg.norm(target - rx)
    radius = np.linalg.norm(target - rx)
    assert total_path != radius  # Independent two-leg geometry reference.
    power = np.ones((9, 41))
    power[4, int(total_path)] = 100.0
    rd = RangeDopplerResult(power, np.arange(41.0), np.arange(9.0) - 4.0, None)
    config = DetectionPipelineConfig((0, 0), (1, 1), 2, min_snr_db=6)
    result = run_detection_pipeline(
        rd,
        config=config,
        wavelength_m=1,
        range_mode="bistatic",
        tx_pos=tx,
        rx_pos=rx,
        aoa_az_map=np.zeros_like(power),
        aoa_el_map=np.zeros_like(power),
    )
    assert len(result.detections) == 1
    detection = result.detections[0]
    assert detection["range_m"] == total_path
    assert not {"x_m", "y_m", "z_m"} & detection.keys()
    tracker = CartesianKalmanTracker(TrackerConfig())
    with pytest.raises(ValueError, match="bistatic total path length"):
        tracker.filter_detections(result.detections, update_time_s=0.0)


def test_monostatic_cartesian_projection_preserves_receiver_frame():
    result = augment_detections_with_sensor_frame_xy(
        [dict(range_m=10.0, aoa_az_deg=90.0, aoa_el_deg=30.0)]
    )[0]
    assert [result[k] for k in ("x_m", "y_m", "z_m")] == pytest.approx([0.0, 5 * np.sqrt(3), 5.0])


@pytest.mark.parametrize("name", ["kalamn", "", None, 3])
def test_unknown_builtin_tracker_is_rejected_before_processing(name):
    with pytest.raises(ValueError, match="Unsupported tracker_type"):
        SensingPostProcessor(PostProcessingConfig(tracker_type=name))


def test_custom_tracker_identifier_and_factory_remain_supported():
    seen = []
    marker = object()

    def factory(config):
        seen.append(config.tracker_type)
        return marker

    processor = SensingPostProcessor(
        PostProcessingConfig(tracker_type="research_tracker"), tracker_factory=factory
    )
    processor._ensure_tracker()
    processor._ensure_tracker()
    assert seen == ["research_tracker"]
    assert processor._tracker is marker


@pytest.mark.parametrize("gate,expected_count", [(0.1, 2), (1.0, 1), (6.0, 1)])
@pytest.mark.parametrize("enabled", [False, True])
def test_ghost_containment_honors_explicit_velocity_gate(gate, expected_count, enabled):
    frames = [
        {
            "_frame_index": i,
            "detections": [
                {"track_id": 1, "range_m": 5.0, "velocity_m_s": 0.0},
                {"track_id": 2, "range_m": 12.0, "velocity_m_s": 1.0},
            ],
        }
        for i in range(6)
    ]
    results, metadata = prune_correlated_ghost_tracks(
        frames, velocity_gate_m_s=gate, enable_coexistence=enabled
    )
    assert all(len(r["detections"]) == (expected_count if enabled else 2) for r in results)
    assert metadata["velocity_gate_m_s"] == gate


@pytest.mark.parametrize("mirror", [None, 4])
def test_fusion_adapter_uses_authoritative_resolved_fft(mirror):
    sensing = replay._frame(
        detections=[
            dict(range_m=10.0, aoa_az_deg=0.0, aoa_el_deg=0.0, velocity_m_s=1.0, doppler_bin=3)
        ]
    )["sensing"]
    sensing["cfar_metadata"] = {"range_doppler": {"pre_cfar_doppler_guards": {"dc_bins": 1}}}
    if mirror is not None:
        sensing["fft_size_doppler"] = mirror
    observation = native_observation("sensor", 10, 0, 0.0015, sensing["detections"][0], sensing, {})
    result = apply_quality(observation, FusionConfig())
    assert len(result.components) == 3
    assert "radial_velocity_adjacent_to_dc_exclusion" in result.quality_reasons
    sensing["fft_size_doppler"] = 8
    with pytest.raises(ValueError, match="contradicts authoritative"):
        native_observation("sensor", 10, 0, 0.0015, sensing["detections"][0], sensing, {})


def oracle_source(fft, *, complex_source=False):
    sensing = pp._v4_sensing(
        frame_idx=4,
        sensing={
            "config": {**pp.SENSING_META["config"], "fft_size_doppler": fft},
            "range_doppler_map": np.ones((fft, 64), dtype=complex if complex_source else float),
        },
    )
    cfg = sensing["config_resolved"]
    cfg["processing_mode"] = "analytic_path_doppler_oracle"
    cfg["processing"].update(
        mode="analytic_path_doppler_oracle",
        strategy="analytic_path_doppler_oracle",
        result_stride_samples=1,
    )
    cfg["acquisition"].update(
        sample_source="analytic_path_snapshot",
        sample_period_s=None,
        sample_rate_hz=None,
        cpi_length_samples=1,
        cpi_hop_samples=1,
        overlap_samples=0,
        rt_solve_stride_scene_steps=1,
    )
    sensing["processing_strategy"] = "analytic_path_doppler_oracle"
    sensing["config"] = deepcopy(cfg)
    pp._rematerialize_range_doppler_descriptor(sensing)
    assert_sensing_contract(
        {"sensing": sensing}, require_schema_version=True, require_described_products=True
    )
    return sensing


@pytest.mark.parametrize("fft", [1, 32, 64])
@pytest.mark.parametrize("complex_source", [False, True])
def test_oracle_publication_preserves_ancestry_across_reprocessing(fft, complex_source):
    source = oracle_source(fft, complex_source=complex_source)
    original = deepcopy(source)
    for _ in range(2):
        result = pp._postprocessed_result(doppler_bins=fft, frame_idx=4)
        result["detections"] = [dict(range_m=10.0, velocity_m_s=0.0, power_linear=1.0, track_id=7)]
        result["temporal"] = deepcopy(source["temporal"])
        result["temporal"]["result"]["processing_recipe_id"] = canonical_sensing_hash(
            result["processing_recipe_resolved"]
        )
        derived = build_reprocessed_sensing_block(
            source_sensing=source,
            result=result,
            postprocessing_config=PostProcessingConfig(),
            input_kind=next(iter(source["products"])),
        )
        event = build_replay_event(
            pp._make_standard_frame(frame_idx=4, sensing=derived),
            ReplaySceneContext(4, 4, 4, 4.0, 4.0),
        )
        assert {p.evidence_layer for p in event.products} == {"oracle"}
        assert {item.layer for item in (*event.detections, *event.tracks)} == {"oracle"}
        assert len(event.detections) == len(event.tracks) == 1
        assert (
            derived["products"]["range_doppler_power_by_pair"]["evidence_role"] == "oracle_derived"
        )
        assert (
            derived["config_resolved"]["source_processing"]["mode"]
            == "analytic_path_doppler_oracle"
        )
        source = derived
    original_key = "range_doppler_by_pair" if complex_source else "range_doppler_power_by_pair"
    np.testing.assert_array_equal(original[original_key], np.ones((1, fft, 64)))
    broken = deepcopy(derived)
    broken["config_resolved"].pop("source_processing")
    broken["config"] = deepcopy(broken["config_resolved"])
    with pytest.raises(ValueError, match="oracle ancestry"):
        assert_sensing_contract(
            {"sensing": broken}, require_schema_version=True, require_described_products=True
        )


@pytest.mark.parametrize("fft", [1, 32, 64])
@pytest.mark.parametrize("skip", [False, True])
def test_oracle_actual_cli_publishes_without_changing_source(tmp_path, monkeypatch, fft, skip):
    import hashlib
    import sys

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    pp._write_packed_frame_set(
        source_dir, pp._make_standard_frame(frame_idx=4, sensing=oracle_source(fft))
    )

    def hashes():
        return {
            str(p.relative_to(source_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in source_dir.rglob("*")
            if p.is_file()
        }

    before = hashes()
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys,
        "argv",
        ["orchav-sensing", str(source_dir), "--output-dir", str(output)]
        + (["--skip-detection"] if skip else ["--save-stage-artifacts"]),
    )
    assert pp.sensing_postprocessor_main() is None
    assert (output / "frames" / "frames_manifest.json").exists()
    assert hashes() == before
    derived = pp._load_frames_sensing(output)[0]["_sensing_json"]
    assert {product["evidence_role"] for product in derived["products"].values()} == {
        "oracle_derived"
    }
    if not skip:
        assert "range_doppler_cfar_noise_power_by_pair" in derived["products"]


@pytest.mark.parametrize("num_bins,fft_size", [(5, 8), (8, 8), (8, 16)])
@pytest.mark.parametrize("prf", [1000.0, 2000.0])
@pytest.mark.parametrize("mode", ["monostatic", "bistatic"])
def test_cfr_replay_reports_frequency_grid_ambiguity(num_bins, fft_size, prf, mode):
    from shared.sensing.products import add_product_spec, materialize_product_specs
    from shared.sensing.range_response import RangeResponseConfig, frequency_to_delay

    frame = replay._frame()
    sensing = frame["sensing"]
    cfg = sensing["config_resolved"]
    response = RangeResponseConfig("cfr_ifft", num_bins, "boxcar")
    cfg.update(
        contract_version=2,
        range_response=response.to_dict(),
        bandwidth=20e6,
        window_range="boxcar",
        fft_size_range=fft_size,
        prf_hz=prf,
        range_mode=mode,
    )
    cfg["processing_recipe"]["window_range"] = "boxcar"
    cfg["processing"]["product_semantics"] = "cfr_ifft_slow_time_fft"
    sensing["config"] = deepcopy(cfg)
    sensing["temporal"]["result"]["processing_recipe_id"] = canonical_sensing_hash(
        cfg["processing_recipe"]
    )
    _, axes, metadata = frequency_to_delay(
        np.ones((1, 4, num_bins), complex),
        bandwidth=20e6,
        fft_size=fft_size,
        config=response,
        range_mode=mode,
    )
    sensing["range_axis_m"] = axes["range_axis_m"]
    sensing["range_doppler_by_pair"] = np.ones((1, 4, fft_size), complex)
    old = sensing["products"]["range_doppler_by_pair"]
    product_axes = deepcopy(old["axes"])
    product_axes[2]["size"] = fft_size
    sensing["products"] = {}
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind=old["kind"],
        axes=product_axes,
        representation=old["representation"],
        unit=old["unit"],
        source_product_ids=("a" * 64,),
    )
    materialize_product_specs(sensing)
    event = replay._event(frame)
    observed = next(
        c for c in event.preflight.checks if c.check_id == "declared_resolution_envelope"
    ).metrics
    # Independent grid formula: df=B/N, delay period=1/df. FFT padding and
    # slow-time PRF change neither. Bistatic coordinates use the full path.
    period = num_bins / 20e6
    expected = 299792458.0 * period / (2 if mode == "monostatic" else 1)
    assert observed["delay_ambiguity_period_s"] == pytest.approx(period)
    assert observed["modeled_unambiguous_range_m"] == pytest.approx(expected)
    assert metadata["delay_ambiguity_period_s"] == pytest.approx(period)
    assert observed["monostatic_unambiguous_range_m"] == (
        pytest.approx(expected) if mode == "monostatic" else None
    )


def test_point_bin_replay_does_not_claim_a_frequency_grid_ambiguity():
    event = replay._event(replay._frame())
    observed = next(
        c for c in event.preflight.checks if c.check_id == "declared_resolution_envelope"
    ).metrics
    assert observed["monostatic_unambiguous_range_m"] is None
    assert observed["modeled_unambiguous_range_m"] is None
    assert observed["theoretical_prf_monostatic_range_bound_m"] == pytest.approx(299792458.0 / 2000)


def test_native_generator_bistatic_rd_does_not_publish_false_cartesian():
    from generator.core.configuration import SensingConfig
    from generator.core.sensing.acquisition import resolve_sensing_execution_values
    from generator.core.sensing.processor import create_sensing_processor

    power = np.ones((9, 41))
    power[4, 30] = 100.0
    rd = RangeDopplerResult(power, np.arange(41.0), np.arange(9.0) - 4.0, None)
    cfg = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        range_mode="bistatic",
        fft_size_range=41,
        fft_size_doppler=9,
        prf_hz=9.0,
        output_detections=True,
        track_confirmation_enabled=False,
        cfar_rd_guard_cells_doppler=0,
        cfar_rd_guard_cells_range=0,
        cfar_rd_ref_cells_doppler=1,
        cfar_rd_ref_cells_range=1,
        cfar_threshold_scale=2,
    )
    execution = resolve_sensing_execution_values(
        cfg, cir_time_steps=9, cir_sampling_frequency_hz=9.0, frame_dt_s=1 / 9
    )
    processor = create_sensing_processor(cfg, 299792458.0, execution)
    processor._cached_tx_pos = np.array([-10.0, 0, 0])
    processor._cached_rx_pos = np.zeros(3)
    result = {}
    processor._run_2d_cfar(
        rd,
        result,
        update_time_s=0.0,
        aoa_az_map=np.zeros_like(power),
        aoa_el_map=np.zeros_like(power),
    )
    assert len(result["detections"]) == 1
    assert result["detections"][0]["range_m"] == 30.0
    assert not {"x_m", "y_m", "z_m"} & result["detections"][0].keys()


def test_sampled_cpi_fft_equality_cannot_be_bypassed_by_offline_mode():
    sensing = replay._frame()["sensing"]
    cfg = sensing["config_resolved"]
    cfg["processing"].update(
        mode="offline_reprocess", strategy="shared_sensing_postprocessor", fft_size_doppler=8
    )
    cfg["processing_mode"] = "offline_reprocess"
    sensing["config"] = deepcopy(cfg)
    with pytest.raises(ValueError, match="fft_size_doppler must equal acquisition"):
        assert_sensing_contract(
            {"sensing": sensing}, require_schema_version=True, require_described_products=True
        )
    cfg["source_processing"] = dict(
        mode="analytic_path_doppler_oracle", strategy="analytic_path_doppler_oracle"
    )
    sensing["config"] = deepcopy(cfg)
    with pytest.raises(ValueError, match="oracle ancestry"):
        assert_sensing_contract(
            {"sensing": sensing}, require_schema_version=True, require_described_products=True
        )


@pytest.mark.parametrize("tracker", ["none", "kalman"])
def test_threshold_sweep_accepts_explicit_rd_route_and_physical_tracker_time(tracker):
    source = oracle_source(32)
    points = pp.sweep_thresholds(
        [{"_typed_sensing": source}],
        threshold_steps=2,
        base_config=PostProcessingConfig(tracker_type=tracker),
        rebuild_cfr=False,
    )
    assert len(points) == 2
    assert all(p.evaluated_event_count == 1 for p in points)


def test_threshold_sweep_rebuilds_cfr_recipe_from_retained_frequency(tmp_path):
    from tests.generator.unit.test_sensing_cfr_lifecycle import native_cfr

    source = native_cfr(noise=0.01)
    points = pp.sweep_thresholds(
        [{"_typed_sensing": source}],
        threshold_steps=2,
        base_config=PostProcessingConfig(range_ifft_size=128),
        rebuild_cfr=True,
    )
    assert len(points) == 2
    assert all(p.evaluated_event_count == 1 for p in points)


@pytest.mark.parametrize(
    "count,index,usable",
    [(0, 2, True), (1, 2, False), (1, 3, False), (1, 0, True), (2, 1, False), (2, 3, False)],
)
def test_fusion_adapter_preserves_dc_boundary_policy_without_mirror(count, index, usable):
    sensing = replay._frame()["sensing"]
    sensing["config_resolved"]["fft_size_doppler"] = np.int64(4)
    sensing["cfar_metadata"] = {"range_doppler": {"pre_cfar_doppler_guards": {"dc_bins": count}}}
    detection = dict(range_m=10.0, velocity_m_s=1.0, doppler_bin=index)
    observation = native_observation("sensor", 10, 0, 0.0015, detection, sensing, {})
    assert (len(apply_quality(observation, FusionConfig()).components) == 2) is usable
