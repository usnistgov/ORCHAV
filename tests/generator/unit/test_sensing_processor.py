from __future__ import annotations

import logging
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import generator.core.sensing.processor as sensing_processor_module
from generator.core.configuration import SensingConfig
from generator.core.sensing.acquisition import (
    ANALYTIC_ORACLE_MODE,
    resolve_sensing_execution_values,
)
from generator.core.sensing.processor import (
    DETECTION_STAGE_PRODUCT_KEYS,
    create_sensing_processor,
)
from shared.sensing.signal_processing import RangeDopplerResult


@pytest.mark.parametrize("mode", ["none", "legacy", "windowed_mean"])
def test_coherent_static_clutter_policy_is_explicit_and_retains_raw(mode):
    cfg = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        coherent_clutter_mode=mode,
        output_detections=True,
        fft_size_range=32,
        fft_size_doppler=128,
        raw_payload_policy="always",
        cfar_max_dynamic_range_db=60,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=4.9e9, cir_time_steps=128)
    result = processor.process(
        None,
        frame_idx=127,
        paths=None,
        cir_stack=(np.full((1, 128), 0.123456789 + 0.987654321j), np.array([12 / cfg.bandwidth])),
        source_time_s=0,
    )
    assert np.any(result["range_doppler_raw_by_pair"] != 0)
    primary = result["range_doppler_by_pair"]
    if mode == "windowed_mean":
        assert np.count_nonzero(primary) == 0
        assert result["detections"] == []
        assert result["config_resolved"]["processing_recipe"]["coherent_clutter_mode"] == mode
        assert result["slow_time_mean_removal_policy"] == "window_weighted_mean"
        assert result["slow_time_mean_removal_applied"] is True
    else:
        assert np.any(primary != 0)
        assert result["slow_time_mean_removal_applied"] is False


def test_windowed_mean_rejects_constant_bias_preserves_motion_and_exposes_slow_loss():
    from shared.sensing.signal_processing import _window, remove_window_weighted_mean

    window = _window("hann", 128)
    times = np.arange(128) / 128
    static = np.full((2, 128, 3), 0.123456789 + 0.987654321j)
    assert np.count_nonzero(remove_window_weighted_mean(static, window, axis=1)) == 0
    gains = []
    for cycles in (0.1, 1.0, 8.0):
        tone = np.exp(2j * np.pi * cycles * times)
        centered = remove_window_weighted_mean(tone, window, axis=0)
        np.testing.assert_allclose(np.sum(centered * window), 0, atol=1e-13)
        shifted = remove_window_weighted_mean(tone + 1000 + 2000j, window, axis=0)
        np.testing.assert_allclose(centered, shifted, atol=1e-12)
        gains.append(np.linalg.norm(centered * window) / np.linalg.norm(tone * window))
    assert gains[0] < 0.2  # Real slow targets are attenuated, not magically recovered.
    assert gains[2] == pytest.approx(1, abs=1e-4)


def _create_processor(
    config: SensingConfig,
    *,
    carrier_frequency_hz: float,
    cir_time_steps: int = 1,
):
    """Resolve the same strict execution contract used by orchestration."""

    if config.processing_mode == "auto" and cir_time_steps == 1:
        config.processing_mode = ANALYTIC_ORACLE_MODE
    resolved = resolve_sensing_execution_values(
        config,
        cir_time_steps=cir_time_steps,
        cir_sampling_frequency_hz=(config.prf_hz if cir_time_steps > 1 else None),
        frame_dt_s=1.0 / config.prf_hz,
    )
    return create_sensing_processor(config, carrier_frequency_hz, resolved)


def _assert_temporal_binding_is_deferred(sensing: dict) -> None:
    """Processor output becomes a wire contract only at the file pipeline."""
    assert "schema_version" not in sensing
    assert "temporal" not in sensing


def _queued_product_specs(sensing: dict) -> dict[str, dict]:
    """Index deferred typed-product specs by their canonical data binding."""

    return {spec["data_key"]: spec for spec in sensing["_product_specs"]}


class _FakePaths:
    def __init__(
        self,
        amplitudes: list[complex],
        delay_s: float = 1e-9,
        doppler_hz: float | None = 100.0,
    ):
        self.amplitudes = amplitudes
        self.delay_s = delay_s
        self.doppler_hz = doppler_hz
        self._idx = 0

    def cir(self, normalize_delays: bool = False):
        idx = min(self._idx, len(self.amplitudes) - 1)
        self._idx += 1
        amp = np.array([self.amplitudes[idx]], dtype=np.complex128)
        tau = np.array([self.delay_s], dtype=np.float64)
        return amp, tau

    @property
    def doppler(self):
        if self.doppler_hz is None:
            return None
        return np.array([self.doppler_hz], dtype=np.float64)


class _ArrayWrapper:
    def __init__(self, data: np.ndarray):
        self._data = data

    def numpy(self) -> np.ndarray:
        return self._data


class _FakeCoherentPaths:
    def __init__(self) -> None:
        self.valid = _ArrayWrapper(np.array([[[True, True]]], dtype=bool))
        self.interactions = _ArrayWrapper(np.array([[[[1, 1]]], [[[2, 2]]]], dtype=np.int32))
        self.objects = _ArrayWrapper(np.array([[[[101, 201]]], [[[102, 202]]]], dtype=np.uint32))


def _fake_simulation_objects():
    scene = SimpleNamespace(
        objects={
            "Wall": SimpleNamespace(object_id=101),
            "Pedestrian": SimpleNamespace(object_id=102),
            "Roof": SimpleNamespace(object_id=201),
            "PedestrianCircular": SimpleNamespace(object_id=202),
        }
    )
    target_manager = SimpleNamespace(
        config=SimpleNamespace(name="Pedestrian"),
        current_mesh_idx=0,
        meshes=["/tmp/fitted_Image_Psm_01_00022.ply"],
    )
    return SimpleNamespace(scene=scene, target_managers=[target_manager])


def test_process_sensing_emits_clutter_metadata():
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=True,
        clutter_removal_enabled=True,
        clutter_removal_window=2,
        fft_size_range=64,
        fft_size_doppler=2,
        raw_payload_policy="always",
    )
    paths = _FakePaths([1.0 + 0j, 2.0 + 0j])
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    first = processor.process(None, 0, paths, source_time_s=0.0)
    second = processor.process(None, 1, paths, source_time_s=1.0)

    assert "clutter_metadata" in second
    metadata = second["clutter_metadata"]
    assert "range_profile" in metadata
    assert metadata["range_profile"]["enabled"]
    assert metadata["range_profile"]["window_frames"] == 2
    assert metadata["range_profile"]["baseline_frames"] >= 1
    assert "range_doppler" in metadata
    assert metadata["range_doppler"]["enabled"]
    assert "range_profile" in second
    assert "range_profile_baseline" in second
    assert "range_profile_raw" in second
    assert "range_profile_raw_by_pair" in second
    # After removal, energy should not increase
    assert np.max(second["range_profile"]) <= np.max(first["range_profile"])
    assert "range_doppler_map" in second
    assert "range_doppler_map_raw" in second
    assert "range_doppler_raw_power_by_pair" in second
    specs = _queued_product_specs(second)
    assert specs["range_profile_raw_by_pair"]["variant"] == "raw"
    assert specs["range_profile_raw_by_pair"]["temporal_support"] == "last_valid_acquisition_sample"
    assert specs["range_doppler_raw_power_by_pair"]["variant"] == "raw"
    assert specs["range_doppler_raw_power_by_pair"]["temporal_support"] == "full_acquisition"
    assert "range_profile_cfar_mask" in second
    assert second["range_profile_cfar_mask"].shape[0] == cfg.fft_size_range
    assert "range_doppler_cfar_mask" not in second
    assert not set(DETECTION_STAGE_PRODUCT_KEYS) & set(second)
    assert "range_doppler_cfar_input_power_db" not in second
    assert "range_doppler_cfar_noise_floor" not in second
    assert "range_doppler_nms_mask" not in second
    assert "range_doppler_snr_map" not in second
    assert "range_doppler_cluster_labels" not in second
    assert "cfar_metadata" in second
    assert isinstance(second.get("detections", []), list)
    assert "config_resolved" in second
    assert second["config_resolved"]["rng_seed"] is None
    assert second["config_resolved"]["persist_detection_stage_artifacts"] is False
    assert second["detection_velocity_axis_label"] == "Monostatic-equivalent Velocity (m/s)"
    assert second["detection_velocity_semantics"] == "monostatic_equivalent_velocity"
    assert second["detection_velocity_is_approximate"] is False
    assert second["detection_velocity_geometry_source"] == "none"
    assert second["path_closing_rate_formula"] == "wavelength_m * doppler_hz"
    _assert_temporal_binding_is_deferred(second)


def test_range_doppler_raw_persisted_without_clutter():
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=32,
        fft_size_doppler=2,
        raw_payload_policy="always",
    )
    paths = _FakePaths([1.0 + 0j for _ in range(3)])
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    for idx in range(3):
        result = processor.process(None, idx, paths, source_time_s=float(idx))

    assert result is not None
    assert "range_doppler_map" in result
    assert "range_doppler_map_raw" in result
    assert "range_doppler_raw_power_by_pair" in result
    np.testing.assert_allclose(result["range_doppler_map"], result["range_doppler_map_raw"])
    _assert_temporal_binding_is_deferred(result)


def test_raw_payload_policy_none_omits_raw_arrays() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=32,
        fft_size_doppler=2,
        raw_payload_policy="none",
    )
    paths = _FakePaths([1.0 + 0j for _ in range(3)])
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    result = None
    for idx in range(3):
        result = processor.process(None, idx, paths, source_time_s=float(idx))

    assert result is not None
    assert "range_profile" in result
    assert "range_doppler_map" in result
    assert "range_profile_raw" not in result
    assert "range_doppler_map_raw" not in result
    assert "range_profile_raw_by_pair" not in result
    assert "range_doppler_raw_power_by_pair" not in result
    assert "cir" not in result
    assert result["raw_payload_policy"] == "none"
    _assert_temporal_binding_is_deferred(result)


def test_default_policy_keeps_processed_products_and_omits_raw_arrays() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=True,
        fft_size_range=32,
        fft_size_doppler=2,
    )
    paths = _FakePaths([1.0 + 0j for _ in range(3)])
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    result = None
    for idx in range(3):
        result = processor.process(None, idx, paths, source_time_s=float(idx))

    assert result is not None
    assert cfg.raw_payload_policy == "none"
    assert result["raw_payload_policy"] == "none"
    assert result["config_resolved"]["raw_payload_policy"] == "none"
    assert "range_profile" in result
    assert "range_doppler_map" in result
    assert "range_profile_cfar_mask" in result
    assert "range_doppler_cfar_mask" not in result
    assert not set(DETECTION_STAGE_PRODUCT_KEYS) & set(result)
    assert isinstance(result["detections"], list)
    for raw_key in ("range_profile_raw", "range_doppler_map_raw", "cir"):
        assert raw_key not in result
    _assert_temporal_binding_is_deferred(result)


@pytest.mark.parametrize("invalid_policy", [None, "invalid"])
def test_processor_invalid_policy_falls_back_to_none(invalid_policy: object) -> None:
    cfg = SensingConfig(enabled=True)
    # Exercise the processor boundary as well as dataclass normalization:
    # loaders and programmatic callers can mutate a config after construction.
    cfg.raw_payload_policy = invalid_policy

    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    assert processor.raw_payload_policy == "none"


def test_raw_payload_policy_diagnostic_keeps_cir_but_drops_raw_maps() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=32,
        fft_size_doppler=2,
        raw_payload_policy="diagnostic",
    )
    paths = _FakePaths([1.0 + 0j for _ in range(3)])
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    result = None
    for idx in range(3):
        result = processor.process(None, idx, paths, source_time_s=float(idx))

    assert result is not None
    assert "cir" in result
    assert "range_profile" in result
    assert "range_doppler_map" in result
    assert "range_profile_raw" not in result
    assert "range_doppler_map_raw" not in result
    assert "range_profile_raw_by_pair" not in result
    assert "range_doppler_raw_power_by_pair" not in result
    assert result["raw_payload_policy"] == "diagnostic"


def test_persist_detection_stage_artifacts_is_an_opt_in_runtime_policy() -> None:
    config = SensingConfig(enabled=True, persist_detection_stage_artifacts=True)

    assert config.persist_detection_stage_artifacts is True


@pytest.mark.parametrize(
    ("processing_mode", "primary_data_key", "source_projection"),
    [
        (
            "analytic_path_doppler_oracle",
            "range_doppler_power_by_pair",
            "identity_power_linear",
        ),
        (
            "sequential_stft",
            "range_doppler_by_pair",
            "squared_magnitude_of_complex_voltage",
        ),
        (
            "coherent_cpi",
            "range_doppler_by_pair",
            "squared_magnitude_of_complex_voltage",
        ),
    ],
)
def test_all_strategies_publish_complete_typed_detection_stage_dag(
    processing_mode: str,
    primary_data_key: str,
    source_projection: str,
) -> None:
    config = SensingConfig(
        enabled=True,
        processing_mode=processing_mode,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        persist_detection_stage_artifacts=True,
        fft_size_range=32,
        fft_size_doppler=4,
        raw_payload_policy="none",
    )
    if processing_mode == "coherent_cpi":
        processor = _create_processor(
            config,
            carrier_frequency_hz=28e9,
            cir_time_steps=4,
        )
        amplitudes = np.array(
            [[1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j]],
            dtype=np.complex128,
        )
        result = processor.process(
            None,
            frame_idx=3,
            paths=None,
            cir_stack=(amplitudes, np.array([1e-9], dtype=np.float64)),
            source_time_s=0.3,
        )
    elif processing_mode == "sequential_stft":
        processor = _create_processor(config, carrier_frequency_hz=28e9)
        paths = _FakePaths([1.0 + 0.1j] * 4)
        result = None
        for frame_idx in range(4):
            result = processor.process(
                None,
                frame_idx,
                paths,
                source_time_s=frame_idx / config.prf_hz,
            )
    else:
        processor = _create_processor(config, carrier_frequency_hz=28e9)
        result = processor.process(
            None,
            0,
            _FakePaths([1.0 + 0.1j], doppler_hz=100.0),
            source_time_s=0.0,
        )

    assert result is not None
    assert result["config_resolved"]["persist_detection_stage_artifacts"] is True
    assert primary_data_key in result
    assert "range_doppler_map" not in result
    assert "range_doppler_cfar_mask" not in result
    assert set(DETECTION_STAGE_PRODUCT_KEYS) <= set(result)
    for data_key in DETECTION_STAGE_PRODUCT_KEYS:
        assert result[data_key].shape == (1, 4, 32)

    assert np.issubdtype(
        result["range_doppler_detection_input_power_by_pair"].dtype,
        np.floating,
    )
    assert np.all(result["range_doppler_detection_input_power_by_pair"] >= 0.0)
    assert np.all(np.isfinite(result["range_doppler_snr_db_by_pair"]))
    assert np.issubdtype(
        result["range_doppler_cfar_training_cell_count_by_pair"].dtype,
        np.integer,
    )
    assert np.all(result["range_doppler_cfar_training_cell_count_by_pair"] >= 0)
    assert np.issubdtype(
        result["range_doppler_cluster_labels_by_pair"].dtype,
        np.signedinteger,
    )
    for data_key in (
        "range_doppler_cfar_eligible_mask_by_pair",
        "range_doppler_cfar_mask_by_pair",
        "range_doppler_nms_mask_by_pair",
        "range_doppler_snr_valid_mask_by_pair",
    ):
        assert result[data_key].dtype == np.bool_

    specs = _queued_product_specs(result)
    assert set(DETECTION_STAGE_PRODUCT_KEYS) <= set(specs)
    input_key = "range_doppler_detection_input_power_by_pair"
    noise_key = "range_doppler_cfar_noise_power_by_pair"
    threshold_key = "range_doppler_cfar_threshold_power_by_pair"
    count_key = "range_doppler_cfar_training_cell_count_by_pair"
    eligible_key = "range_doppler_cfar_eligible_mask_by_pair"
    cfar_mask_key = "range_doppler_cfar_mask_by_pair"
    nms_key = "range_doppler_nms_mask_by_pair"
    snr_key = "range_doppler_snr_db_by_pair"
    snr_valid_key = "range_doppler_snr_valid_mask_by_pair"
    labels_key = "range_doppler_cluster_labels_by_pair"
    expected_sources = {
        input_key: [primary_data_key],
        noise_key: [input_key],
        count_key: [input_key],
        eligible_key: [input_key, count_key],
        threshold_key: [noise_key, eligible_key],
        cfar_mask_key: [input_key, threshold_key, eligible_key],
        snr_key: [input_key, noise_key, eligible_key],
        snr_valid_key: [noise_key, eligible_key],
        nms_key: [input_key, cfar_mask_key, snr_valid_key],
        labels_key: [nms_key, snr_key, snr_valid_key],
    }
    for data_key, source_data_keys in expected_sources.items():
        assert specs[data_key]["source_data_keys"] == source_data_keys
        assert [axis["name"] for axis in specs[data_key]["axes"]] == [
            "pair",
            "doppler_frequency",
            "range",
        ]
    assert specs[input_key]["processing"]["source_projection"] == source_projection
    assert specs[noise_key]["processing"]["edge_policy"] == config.cfar_edge_policy
    assert specs[nms_key]["processing"]["candidate_mask"] == ("cfar_mask and snr_valid_mask")
    assert specs[snr_key]["processing"]["formula"] == ("10 * log10(input_power / noise_power)")
    assert specs[labels_key]["processing"]["background_label"] == -1
    assert specs[labels_key]["processing"]["missing_sklearn_fallback"] == (
        "connected_components_of_unit_radius_graph"
    )


def test_coherent_strategy_emits_contract_fields_and_policy() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="diagnostic",
    )
    processor = _create_processor(
        cfg,
        carrier_frequency_hz=28e9,
        cir_time_steps=4,
    )
    assert processor.__class__.__name__ == "CoherentCpiFftProcessor"

    a = np.array([[1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j]], dtype=np.complex128)
    tau = np.array([1e-9], dtype=np.float64)

    result = processor.process(
        None,
        frame_idx=3,
        paths=None,
        cir_stack=(a, tau),
        source_time_s=1.5,
    )
    assert result is not None
    assert result["processing_strategy"] == "coherent_cpi_fft"
    assert result["rd_source"] == "fft_cpi"
    assert result["frame_role"] == "cpi_end"
    assert result["raw_payload_policy"] == "diagnostic"
    assert result["rd_fresh"] is True
    assert result["rp_fresh"] is True
    assert "range_doppler_map" in result
    assert "range_doppler_map_raw" not in result
    assert "cir" in result
    assert "slow_time_matrix" in result
    assert "range_profile_raw" not in result
    specs = _queued_product_specs(result)
    assert specs["range_doppler_by_pair"]["processing"]["slow_time_mean_removal_applied"] is False
    _assert_temporal_binding_is_deferred(result)


def test_coherent_always_policy_queues_canonical_raw_variants() -> None:
    cfg = SensingConfig(
        enabled=True,
        processing_mode="coherent_cpi",
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="always",
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9, cir_time_steps=4)
    amplitudes = np.array(
        [[1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j]],
        dtype=np.complex128,
    )
    result = processor.process(
        None,
        frame_idx=3,
        paths=None,
        cir_stack=(amplitudes, np.array([20e-9], dtype=np.float64)),
        source_time_s=1.5,
    )

    assert result is not None
    assert result["range_profile_raw_by_pair"].shape == (1, 16)
    assert result["range_doppler_raw_by_pair"].shape == (1, 4, 16)
    specs = _queued_product_specs(result)
    assert specs["range_profile_raw_by_pair"]["variant"] == "raw"
    assert specs["range_doppler_raw_by_pair"]["variant"] == "raw"
    primary_processing = specs["range_doppler_by_pair"]["processing"]
    raw_processing = specs["range_doppler_raw_by_pair"]["processing"]
    assert primary_processing["slow_time_mean_removal_policy"] == ("when_row_variation_nonzero")
    assert primary_processing["slow_time_mean_removal_applied"] is True
    assert raw_processing["slow_time_mean_removal_policy"] == ("when_row_variation_nonzero")
    assert raw_processing["slow_time_mean_removal_applied"] is False


def test_sequential_always_policy_queues_canonical_raw_variants() -> None:
    cfg = SensingConfig(
        enabled=True,
        processing_mode="sequential_stft",
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=2,
        window_doppler="boxcar",
        raw_payload_policy="always",
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)
    paths = _FakePaths([1.0 + 0.0j, 2.0 + 0.0j], delay_s=20e-9)
    processor.process(None, frame_idx=0, paths=paths, source_time_s=0.0)
    result = processor.process(None, frame_idx=1, paths=paths, source_time_s=1.0 / cfg.prf_hz)

    assert result is not None
    assert result["range_profile_raw_by_pair"].shape == (1, 16)
    assert result["range_doppler_raw_by_pair"].shape == (1, 2, 16)
    specs = _queued_product_specs(result)
    assert specs["range_profile_raw_by_pair"]["variant"] == "raw"
    assert specs["range_doppler_raw_by_pair"]["variant"] == "raw"
    primary_processing = specs["range_doppler_by_pair"]["processing"]
    raw_processing = specs["range_doppler_raw_by_pair"]["processing"]
    assert primary_processing["slow_time_mean_removal_policy"] == ("when_row_variation_nonzero")
    assert primary_processing["slow_time_mean_removal_applied"] is True
    assert raw_processing["slow_time_mean_removal_policy"] == ("when_row_variation_nonzero")
    assert raw_processing["slow_time_mean_removal_applied"] is False


@pytest.mark.parametrize("clutter_enabled", [False, True])
def test_sequential_exact_config_is_stable_across_first_complete_cpi(clutter_enabled) -> None:
    """Persisted warmup CIR and overlapping full windows belong to one acquisition."""
    cfg = SensingConfig(
        enabled=True,
        processing_mode="sequential_stft",
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="always",
        clutter_removal_enabled=clutter_enabled,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)
    paths = _FakePaths([complex(np.exp(0.4j * index)) for index in range(8)], delay_s=20e-9)
    resolved_configs = []
    frames = []
    for index in range(8):
        result = processor.process(
            None, frame_idx=index, paths=paths, source_time_s=index / cfg.prf_hz
        )
        assert result is not None
        resolved_configs.append(deepcopy(result["config_resolved"]))
        frames.append(result)

    assert all(config == resolved_configs[0] for config in resolved_configs[1:])
    assert resolved_configs[0]["fft_normalization"] == "unitary_1_over_sqrt_n"
    assert "range_doppler_by_pair" not in frames[0]
    assert "range_doppler_by_pair" in frames[3]
    # Acquisition rows remain the input to every overlapping complete window.
    pulse_rows = np.stack([frame["cir_by_pair_rx_element"][0, 0] for frame in frames])
    for index in range(3, 8):
        np.testing.assert_array_equal(
            frames[index]["slow_time_by_pair_rx_element"][0, 0],
            pulse_rows[index - 3 : index + 1],
        )


def test_coherent_strategy_drops_nonfinite_cir_paths() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
        raw_payload_policy="diagnostic",
    )
    processor = _create_processor(
        cfg,
        carrier_frequency_hz=28e9,
        cir_time_steps=4,
    )

    a = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j],
            [np.nan + 1.0j, np.nan + 1.0j, np.nan + 1.0j, np.nan + 1.0j],
        ],
        dtype=np.complex128,
    )
    tau = np.array([1e-9, 2e-9], dtype=np.float64)

    result = processor.process(
        None,
        frame_idx=127,
        paths=None,
        cir_stack=(a, tau),
        source_time_s=63.5,
    )

    assert result is not None
    for key in ("cir", "slow_time_matrix", "range_profile", "range_doppler_map"):
        assert key in result
        assert not np.isnan(result[key]).any()
    _assert_temporal_binding_is_deferred(result)


def test_coherent_strategy_logs_nonfinite_path_context(caplog: pytest.LogCaptureFixture) -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=True,
        output_range_doppler=True,
        output_detections=False,
        fft_size_range=16,
        fft_size_doppler=4,
    )
    processor = _create_processor(
        cfg,
        carrier_frequency_hz=28e9,
        cir_time_steps=4,
    )

    a = np.array(
        [
            [1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j],
            [np.nan + 1.0j, np.nan + 1.0j, np.nan + 1.0j, np.nan + 1.0j],
        ],
        dtype=np.complex128,
    )
    tau = np.array([[[1e-9, 2e-9]]], dtype=np.float64)

    orchav_logger = logging.getLogger("orchav")
    original_handlers = list(orchav_logger.handlers)
    original_propagate = orchav_logger.propagate
    try:
        orchav_logger.handlers = []
        orchav_logger.propagate = True
        with caplog.at_level(
            logging.WARNING,
            logger="orchav.generator.core.sensing.strategies.coherent_cpi_fft",
        ):
            result = processor.process(
                _fake_simulation_objects(),
                frame_idx=7808,
                paths=_FakeCoherentPaths(),
                cir_stack=(a, tau),
                source_time_s=3904.0,
            )
    finally:
        orchav_logger.handlers = original_handlers
        orchav_logger.propagate = original_propagate

    assert result is not None
    messages = [record.getMessage() for record in caplog.records]
    assert any(
        "Dropping 1/2 non-finite coherent CIR paths at frame 7808" in msg for msg in messages
    )
    detail = next(msg for msg in messages if "Non-finite coherent CIR path at frame 7808" in msg)
    assert "rx=0 tx=0 path=1" in detail
    assert "Roof" in detail
    assert "PedestrianCircular" in detail
    assert "specular" in detail
    assert "diffuse" in detail
    assert "fitted_Image_Psm_01_00022.ply" in detail


def _run_detection_sequence_with_target_truth(target_truth: object) -> list[dict[str, object]]:
    """Run identical observations while varying inaccessible target truth."""
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        bandwidth=32.0,
        prf_hz=16.0,
        fft_size_range=32,
        fft_size_doppler=16,
        range_mode="bistatic",
        track_confirmation_enabled=True,
        track_confirmation_m=1,
        track_confirmation_n=2,
        track_output_tentative=True,
        persist_detection_stage_artifacts=True,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=1.0)
    simulation_objects = SimpleNamespace(
        tx_positions_cache=[[np.array([-1.0, 0.0, 0.0])] * 2],
        rx_positions_cache=[[np.array([1.0, 0.0, 0.0])] * 2],
    )
    if target_truth is not _ABSENT_TARGET_TRUTH:
        simulation_objects.tgt_positions_cache = target_truth

    power = np.zeros((16, 32), dtype=np.float64)
    power[10, 12] = 100.0
    power[10, 13] = 30.0
    rd_result = RangeDopplerResult(
        power_linear=power,
        range_axis_m=np.arange(32, dtype=np.float64),
        doppler_axis_hz=np.fft.fftshift(np.fft.fftfreq(16, d=1.0 / 16.0)),
        cube=None,
    )

    outputs: list[dict[str, object]] = []
    for frame_idx in range(2):
        processor._update_geometry_cache(simulation_objects, frame_idx)
        sensing_data: dict[str, object] = {}
        processor._run_2d_cfar(
            rd_result,
            sensing_data,
            update_time_s=float(frame_idx),
        )
        outputs.append(
            {
                "detections": sensing_data["detections"],
                "filter_metadata": sensing_data["detection_filter_metadata"],
                "cfar_mask": sensing_data["range_doppler_cfar_mask_by_pair"],
            }
        )
    return outputs


_ABSENT_TARGET_TRUTH = object()


def test_detection_and_tracking_are_invariant_to_absent_corrupt_or_permuted_truth() -> None:
    """Target truth is evaluation-only and cannot affect estimated products."""
    target_a = [np.array([0.0, 1.0, 0.0])] * 2
    target_b = [np.array([0.0, 20.0, 0.0])] * 2
    baseline = _run_detection_sequence_with_target_truth([target_a, target_b])

    variants = (
        _ABSENT_TARGET_TRUTH,
        object(),  # Deliberately invalid cache: estimation must never inspect it.
        [target_b, target_a],
    )
    for variant in variants:
        np.testing.assert_equal(
            _run_detection_sequence_with_target_truth(variant),
            baseline,
        )


def test_range_gate_filters_out_of_bounds_detections() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        max_detection_range_m=6.0,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    detections = [
        {"range_m": 4.5, "velocity_m_s": 1.4},
        {"range_m": 8.8, "velocity_m_s": 1.0},
    ]
    filtered, metadata = processor._apply_range_gate(detections)

    assert len(filtered) == 1
    assert filtered[0]["range_m"] == 4.5
    assert metadata["enabled"] is True
    assert metadata["rejected"] == 1


def test_invalid_tracker_type_defaults_to_m_of_n() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        tracker_type="typo_tracker",
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    assert cfg.tracker_type == "m_of_n"
    assert processor._tracker.__class__.__name__ == "MofNTracker"


def test_analytic_oracle_noise_keeps_typed_rd_power_consistent_with_detection_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_detection_power: dict[str, np.ndarray] = {}
    original_run_detection_pipeline = sensing_processor_module.run_detection_pipeline

    def capture_detection_input(rd_result, *args, **kwargs):
        captured_detection_power["power_linear"] = np.asarray(rd_result.power_linear).copy()
        return original_run_detection_pipeline(rd_result, *args, **kwargs)

    monkeypatch.setattr(
        sensing_processor_module,
        "run_detection_pipeline",
        capture_detection_input,
    )
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=True,
        output_detections=True,
        fft_size_range=32,
        fft_size_doppler=8,
        noise_enabled=True,
        noise_snr_db=15.0,
        rng_seed=123,
        raw_payload_policy="always",
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    result = processor.process(
        None,
        0,
        _FakePaths([1.0 + 0j], doppler_hz=100.0),
        source_time_s=0.0,
    )

    assert result is not None
    assert "range_doppler_map_raw" in result
    assert not np.allclose(result["range_doppler_map"], result["range_doppler_map_raw"])
    np.testing.assert_allclose(
        result["range_doppler_power_by_pair"][0],
        captured_detection_power["power_linear"],
    )
    _assert_temporal_binding_is_deferred(result)


def test_track_confirmation_keeps_consistent_track_and_rejects_wandering_ghost() -> None:
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=False,
        output_detections=True,
        track_confirmation_enabled=True,
        track_confirmation_m=2,
        track_confirmation_n=3,
        track_max_missed_frames=1,
        track_association_range_gate_m=0.6,
        track_association_velocity_gate_m_s=0.3,
        track_output_tentative=False,
    )
    processor = _create_processor(cfg, carrier_frequency_hz=28e9)

    # Frame 1: one true target-like detection and one ghost.
    frame1 = [
        {"range_m": 4.5, "velocity_m_s": 1.4},
        {"range_m": 8.8, "velocity_m_s": 1.0},
    ]
    out1, meta1 = processor._apply_track_confirmation(frame1, update_time_s=0.0)
    assert out1 == []  # no confirmed tracks yet (M=2)
    assert meta1["confirmed_tracks"] == 0

    # Frame 2: true target persists; ghost shifts too far in range to associate.
    frame2 = [
        {"range_m": 4.6, "velocity_m_s": 1.4},
        {"range_m": 9.8, "velocity_m_s": 1.0},
    ]
    out2, meta2 = processor._apply_track_confirmation(frame2, update_time_s=1.0)
    assert len(out2) == 1
    assert out2[0]["track_confirmed"] is True
    assert out2[0]["range_m"] == 4.6
    assert meta2["confirmed_tracks"] == 1

    # Frame 3: true target still confirmed and emitted.
    frame3 = [
        {"range_m": 4.7, "velocity_m_s": 1.4},
        {"range_m": 10.8, "velocity_m_s": 1.0},
    ]
    out3, meta3 = processor._apply_track_confirmation(frame3, update_time_s=2.0)
    assert len(out3) == 1
    assert out3[0]["track_confirmed"] is True
    assert out3[0]["range_m"] == 4.7
    assert meta3["confirmed_tracks"] == 1


def test_coherent_strategy_stage_profile_is_emitted_when_env_enabled(monkeypatch) -> None:
    monkeypatch.setenv("ORCHAV_SENSING_PROFILE", "1")
    cfg = SensingConfig(
        enabled=True,
        output_range_profile=False,
        output_range_doppler=True,
        output_detections=True,
        fft_size_range=16,
        fft_size_doppler=4,
    )
    processor = _create_processor(
        cfg,
        carrier_frequency_hz=28e9,
        cir_time_steps=4,
    )

    a = np.array([[1.0 + 0.0j, 1.0 + 0.1j, 1.0 + 0.2j, 1.0 + 0.3j]], dtype=np.complex128)
    tau = np.array([1e-9], dtype=np.float64)

    result = processor.process(
        None,
        frame_idx=3,
        paths=None,
        cir_stack=(a, tau),
        source_time_s=1.5,
    )

    assert result is not None
    stage_profile = result["timing"]["stage_profile_ms"]
    assert set(stage_profile) >= {
        "cir_extract",
        "flatten_cir",
        "cir_projection",
        "rd_fft",
        "detection",
        "total_strategy",
    }
