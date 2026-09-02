from types import SimpleNamespace

from generator.core.configuration import (
    CoverageConfig,
    ReceiverConfig,
    SensingConfig,
    SimulationConfig,
    TransmitterConfig,
    load_simulation_config,
)


def test_get_coverage_quality_settings_maps_rays_to_samples():
    # For each coverage preset, samples_per_tx equals samples_per_src from the profile.
    sc = SimulationConfig()
    for preset, profile in SimulationConfig.QUALITY_PRESETS.items():
        sc.coverage.quality = preset
        qs = sc.get_coverage_quality_settings()
        assert qs["max_depth"] == int(profile.get("max_depth", 3))
        assert qs["samples_per_tx"] == int(profile.get("samples_per_src", 100000))
        assert isinstance(qs["specular_reflection"], bool)
        assert isinstance(qs["diffuse_reflection"], bool)
        assert isinstance(qs["refraction"], bool)
        assert isinstance(qs["diffraction"], bool)


def test_get_quality_profile_returns_isolated_copy():
    config = SimulationConfig(quality="medium")
    baseline_depth = SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"]

    profile_a = config.get_quality_profile()
    profile_a["max_depth"] = baseline_depth + 7
    profile_b = config.get_quality_profile()

    assert profile_b["max_depth"] == baseline_depth
    assert SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"] == baseline_depth


def test_custom_quality_profile_returns_isolated_medium_copy():
    config = SimulationConfig(quality="custom")
    baseline_depth = SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"]

    profile = config.get_quality_profile()
    profile["max_depth"] = baseline_depth + 1

    assert config.get_quality_profile()["max_depth"] == baseline_depth
    assert SimulationConfig.QUALITY_PRESETS["medium"]["max_depth"] == baseline_depth


def test_load_simulation_config_missing_scenario_fields_preserve_defaults():
    scenario = SimpleNamespace(raytracing={}, sensing={})

    cfg = load_simulation_config(scenario)

    assert cfg.scene_name == "etoile"
    assert cfg.duration == 10.0
    assert cfg.num_steps == 10
    assert cfg.start_step == 0
    assert cfg.quality == "low"
    assert cfg.view == "top"
    assert cfg.debug_level == "WARNING"
    assert cfg.file_format == "hdf5"
    assert cfg.output_mode == "local"
    assert cfg.sensing.enabled is False


def test_load_simulation_config_applies_explicit_falsy_values_and_empty_collections():
    base = SimulationConfig(
        duration=12.0,
        start_step=4,
        export_path_metrics=True,
        temperature_k=290.0,
        mesh_update_interval_s=2.0,
    )
    base.material_overrides = {"itu_concrete": {"scattering_coefficient": 0.4}}
    scenario = SimpleNamespace(
        timeline=SimpleNamespace(steps=1, duration_s=0.0),
        raytracing={
            "start_step": 0,
            "export_path_metrics": False,
            "temperature_k": 0.0,
            "mesh_update_interval_s": 0.0,
            "materials": {},
        },
        sensing={
            "enabled": True,
            "output_range_doppler": False,
            "output_range_profile": False,
            "output_detections": False,
            "track_confirmation_enabled": False,
            "noise_enabled": False,
            "aoa_filter_enabled": False,
            "rng_seed": 0,
            "pre_cfar_doppler_dc_guard_bins": 0,
            "min_detection_range_m": 0.0,
            "track_max_missed_frames": 0,
            "display_range_xlim": [],
            "display_velocity_ylim": [],
        },
    )

    cfg = load_simulation_config(scenario, base=base)

    assert cfg.duration == 0.0
    assert cfg.start_step == 0
    assert cfg.export_path_metrics is False
    assert cfg.temperature_k == 0.0
    assert cfg.mesh_update_interval_s == 0.0
    assert cfg.material_overrides == {}
    assert cfg.sensing.output_range_doppler is False
    assert cfg.sensing.output_range_profile is False
    assert cfg.sensing.output_detections is False
    assert cfg.sensing.track_confirmation_enabled is False
    assert cfg.sensing.noise_enabled is False
    assert cfg.sensing.aoa_filter_enabled is False
    assert cfg.sensing.rng_seed == 0
    assert cfg.sensing.pre_cfar_doppler_dc_guard_bins == 0
    assert cfg.sensing.min_detection_range_m == 0.0
    assert cfg.sensing.track_max_missed_frames == 0
    assert cfg.sensing.display_range_xlim == []
    assert cfg.sensing.display_velocity_ylim == []


def test_load_simulation_config_reads_steps_and_duration_from_v2_timeline():
    scenario = SimpleNamespace(
        timeline=SimpleNamespace(steps=37, duration_s=4.5),
        raytracing={"quality": {"preset": "low"}},
        sensing={},
    )

    cfg = load_simulation_config(scenario)

    assert cfg.num_steps == 37
    assert cfg.duration == 4.5
    assert cfg.quality == "low"


def test_load_simulation_config_deep_copies_base_nested_mutables():
    base = SimulationConfig()
    base.coverage = CoverageConfig(enabled=True, metrics_store=["path_gain_linear"])
    base.sensing = SensingConfig(
        enabled=True,
        display_range_xlim=[1.0, 10.0],
        display_velocity_ylim=[-2.0, 2.0],
    )
    base.material_overrides = {"mat": {"scattering_coefficient": 0.1}}
    base.transmitters = {
        "tx": TransmitterConfig(
            name="tx",
            mobility=SimpleNamespace(start_pos=(0.0, 0.0, 1.0)),
        )
    }
    base.receivers = {
        "rx": ReceiverConfig(
            name="rx",
            mobility=SimpleNamespace(start_pos=(1.0, 0.0, 1.0)),
        )
    }

    cfg = load_simulation_config(base=base)

    assert cfg is not base
    assert cfg.coverage is not base.coverage
    assert cfg.sensing is not base.sensing
    assert cfg.material_overrides is not base.material_overrides
    assert cfg.material_overrides["mat"] is not base.material_overrides["mat"]
    assert cfg.transmitters is not base.transmitters
    assert cfg.transmitters["tx"] is not base.transmitters["tx"]
    assert cfg.receivers is not base.receivers
    assert cfg.receivers["rx"] is not base.receivers["rx"]

    cfg.coverage.metrics_store.append("rss_w")
    cfg.sensing.display_range_xlim.append(20.0)
    cfg.material_overrides["mat"]["scattering_coefficient"] = 0.2
    cfg.transmitters["tx"].name = "tx-copy"
    cfg.receivers["rx"].name = "rx-copy"

    assert base.coverage.metrics_store == ["path_gain_linear"]
    assert base.sensing.display_range_xlim == [1.0, 10.0]
    assert base.material_overrides["mat"]["scattering_coefficient"] == 0.1
    assert base.transmitters["tx"].name == "tx"
    assert base.receivers["rx"].name == "rx"


def test_load_simulation_config_maps_sensing_yaml_fields():
    sensing = {
        "enabled": True,
        "bandwidth": 80e6,
        "chirps_per_frame": 64,
        "prf_hz": 1500.0,
        "range_mode": "monostatic",
        "fft_size_range": 128,
        "fft_size_doppler": 64,
        "window_range": "blackman",
        "window_doppler": "boxcar",
        "cfar_type": "os",
        "cfar_guard_cells": 3,
        "cfar_ref_cells": 12,
        "cfar_threshold_scale": 4.5,
        "cfar_os_rank": 0.6,
        "cfar_rd_guard_cells_range": 5,
        "cfar_rd_guard_cells_doppler": 2,
        "cfar_rd_ref_cells_range": 9,
        "cfar_rd_ref_cells_doppler": 6,
        "cfar_min_snr_db": 7.5,
        "pre_cfar_doppler_edge_guard_bins": 1,
        "pre_cfar_doppler_dc_guard_bins": 2,
        "min_detection_range_m": 0.5,
        "max_detection_range_m": 60.0,
        "track_confirmation_enabled": True,
        "track_confirmation_m": 3,
        "track_confirmation_n": 5,
        "track_max_missed_frames": 4,
        "track_association_range_gate_m": 1.2,
        "track_association_velocity_gate_m_s": 0.8,
        "track_output_tentative": True,
        "tracker_type": "kalman",
        "eval_association_range_gate_m": 12.0,
        "eval_association_velocity_gate_m_s": 6.0,
        "output_range_doppler": False,
        "output_range_profile": True,
        "output_detections": False,
        "persist_detection_stage_artifacts": True,
        "raw_payload_policy": "diagnostic",
        "clutter_removal_enabled": True,
        "clutter_removal_window": 5,
        "processing_mode": "sequential_stft",
        "display_range_xlim": [0.0, 30.0],
        "display_velocity_ylim": [-3.0, 3.0],
        "noise_enabled": True,
        "noise_snr_db": 20.0,
        "rng_seed": 42,
        "aoa_filter_enabled": True,
        "aoa_az_min_deg": -45.0,
        "aoa_az_max_deg": 45.0,
        "aoa_el_min_deg": -20.0,
        "aoa_el_max_deg": 20.0,
    }
    scenario = SimpleNamespace(raytracing={}, sensing=sensing)

    cfg = load_simulation_config(scenario)
    actual = cfg.sensing

    assert actual.bandwidth == 80e6
    assert actual.chirps_per_frame == 64
    assert actual.prf_hz == 1500.0
    assert actual.range_mode == "monostatic"
    assert actual.fft_size_range == 128
    assert actual.fft_size_doppler == 64
    assert actual.window_range == "blackman"
    assert actual.window_doppler == "boxcar"
    assert actual.cfar_type == "os"
    assert actual.cfar_guard_cells == 3
    assert actual.cfar_ref_cells == 12
    assert actual.cfar_threshold_scale == 4.5
    assert actual.cfar_os_rank == 0.6
    assert actual.cfar_rd_guard_cells_range == 5
    assert actual.cfar_rd_guard_cells_doppler == 2
    assert actual.cfar_rd_ref_cells_range == 9
    assert actual.cfar_rd_ref_cells_doppler == 6
    assert actual.cfar_min_snr_db == 7.5
    assert actual.pre_cfar_doppler_edge_guard_bins == 1
    assert actual.pre_cfar_doppler_dc_guard_bins == 2
    assert actual.min_detection_range_m == 0.5
    assert actual.max_detection_range_m == 60.0
    assert actual.track_confirmation_enabled is True
    assert actual.track_confirmation_m == 3
    assert actual.track_confirmation_n == 5
    assert actual.track_max_missed_frames == 4
    assert actual.track_association_range_gate_m == 1.2
    assert actual.track_association_velocity_gate_m_s == 0.8
    assert actual.track_output_tentative is True
    assert actual.tracker_type == "kalman"
    assert actual.eval_association_range_gate_m == 12.0
    assert actual.eval_association_velocity_gate_m_s == 6.0
    assert actual.output_range_doppler is False
    assert actual.output_range_profile is True
    assert actual.output_detections is False
    assert actual.persist_detection_stage_artifacts is True
    assert actual.raw_payload_policy == "diagnostic"
    assert actual.clutter_removal_enabled is True
    assert actual.clutter_removal_window == 5
    assert actual.processing_mode == "sequential_stft"
    assert actual.display_range_xlim == [0.0, 30.0]
    assert actual.display_velocity_ylim == [-3.0, 3.0]
    assert actual.noise_enabled is True
    assert actual.noise_snr_db == 20.0
    assert actual.rng_seed == 42
    assert actual.aoa_filter_enabled is True
    assert actual.aoa_az_min_deg == -45.0
    assert actual.aoa_az_max_deg == 45.0
    assert actual.aoa_el_min_deg == -20.0
    assert actual.aoa_el_max_deg == 20.0
