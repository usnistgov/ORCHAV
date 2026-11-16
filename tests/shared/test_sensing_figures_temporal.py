"""Physical-time contract tests for sensing micro-Doppler inputs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID, uuid5

import numpy as np
import pytest

import shared.sensing.plotting.figures as sensing_figures
from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_sensing_hash,
    canonical_sensing_pair_id,
    finalize_sensing_v4_publication,
)
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.sensing.evaluation import ThresholdSweepPoint, evaluate_scenario
from shared.sensing.products import (
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)

_GENERATION_ID = "00000000-0000-4000-8000-000000000091"
_PROCESSING_RUN_ID = "00000000-0000-4000-8000-000000000092"


def test_threshold_annotations_skip_duplicate_operating_points() -> None:
    mean_false_alarms = np.asarray([3.0, 99.0, 3.0, 99.0, 2.0, 99.0, 1.0, 99.0, 0.0, 99.0, 0.0])
    detection_probability = np.asarray([1.0, 99.0, 1.0, 99.0, 0.8, 99.0, 0.7, 99.0, 0.5, 99.0, 0.5])

    assert sensing_figures._threshold_annotation_indices(
        mean_false_alarms,
        detection_probability,
    ) == [0, 4, 6, 8]


def test_generate_figures_emits_gt_evolution_from_evaluable_schema_v4_truth(
    tmp_path: Path,
) -> None:
    pytest.importorskip("matplotlib")
    pair_id = "a" * 64
    typed_sensing = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "config_resolved": {"range_mode": "monostatic"},
        "temporal": {
            "acquisition": {"pair_id": pair_id},
            "truth": {"reference_time_s": 1.0},
        },
        "truth": {
            "targets": [
                {
                    "target_id": "walker",
                    "lifecycle": "active",
                    "valid": True,
                    "pair_observables": [
                        {
                            "pair_id": pair_id,
                            "bistatic_path_length_m": 10.0,
                            "monostatic_equivalent_range_m": 5.0,
                            "monostatic_equivalent_velocity_m_s": 1.0,
                            "velocity_valid": True,
                            "velocity_derivation_source": "analytic_velocity_projection",
                            "velocity_reference_dt_s": None,
                            "undefined_reason": None,
                        }
                    ],
                }
            ]
        },
    }
    result = {
        "_frame_index": 7,
        "_typed_sensing": typed_sensing,
        "_range_mode": "monostatic",
        "config": {"range_mode": "monostatic", "carrier_freq_hz": 28.0e9},
        "rd_map": np.ones((2, 2), dtype=np.float64),
        "range_axis": np.asarray([4.0, 5.0]),
        "doppler_axis": np.asarray([-1.0, 1.0]),
        "detections": [{"range_m": 5.0, "velocity_m_s": 1.0}],
    }

    sensing_figures.generate_figures(
        tmp_path,
        1,
        all_results=[result],
        figure_names={"detection_timeline"},
    )

    assert (tmp_path / "figures" / "detection_timeline.png").is_file()


def test_evaluation_figures_render_undefined_metrics_honestly(tmp_path: Path) -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    empty = np.array([], dtype=np.float64)
    evaluation = evaluate_scenario(
        [(empty, empty.copy()), (empty, empty.copy())],
        [(empty, empty.copy()), (empty, empty.copy())],
    )

    history = sensing_figures._build_association_history(evaluation)
    assert np.isnan(history[0]["range_rmse"])
    cumulative_path = sensing_figures._plot_cumulative_detection_performance(
        tmp_path,
        evaluation,
        plt,
    )
    assert cumulative_path is not None and cumulative_path.is_file()

    sensing_figures.generate_threshold_sweep_figures(
        tmp_path,
        [
            ThresholdSweepPoint(
                threshold_scale=1.0,
                pd=None,
                mean_false_alarms_per_evaluated_event=0.0,
                miss_rate=None,
                evaluated_event_count=2,
                target_opportunity_count=0,
                matched_pair_count=0,
                total_detections=0,
                total_tp=0,
                total_fp=0,
                total_fn=0,
                range_rmse=None,
                velocity_rmse=None,
            )
        ],
    )
    assert (tmp_path / "figures" / "detection_probability_vs_mean_false_alarms.png").is_file()
    assert (tmp_path / "figures" / "miss_rate_vs_mean_false_alarms.png").is_file()


def _rd_animation_result(*, carrier_hz: float = 24.0e9, range_mode: str = "monostatic"):
    return {
        "config": {
            "carrier_freq_hz": carrier_hz,
            "range_mode": range_mode,
        },
        "_range_mode": range_mode,
    }


def test_rd_animation_uses_exact_persisted_carrier_and_range_mode() -> None:
    assert sensing_figures._exact_rd_animation_contract(
        [_rd_animation_result(carrier_hz=24.25e9, range_mode="monostatic")]
    ) == (24.25e9, "monostatic")


@pytest.mark.parametrize(
    "results,match",
    [
        ([{"config": {"range_mode": "bistatic"}}], "carrier_freq_hz"),
        ([{"config": {"carrier_freq_hz": 28.0e9}}], "range_mode"),
        (
            [
                _rd_animation_result(carrier_hz=24.0e9),
                _rd_animation_result(carrier_hz=28.0e9),
            ],
            "do not share",
        ),
        (
            [
                {
                    **_rd_animation_result(range_mode="monostatic"),
                    "_range_mode": "bistatic",
                }
            ],
            "mirror disagrees",
        ),
    ],
)
def test_rd_animation_rejects_missing_or_inconsistent_exact_contract(
    results: list[dict], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        sensing_figures._exact_rd_animation_contract(results)


def _pose_frame(*, frame_pair_index: int = 1) -> dict:
    return {
        "_typed_sensing": {
            "pair_axis": [
                {
                    "pair_index": 0,
                    "frame_pair_index": frame_pair_index,
                    "pair_id": "pair.exact",
                    "tx_sensor_id": "tx.exact",
                    "rx_sensor_id": "rx.exact",
                }
            ],
            "temporal": {
                "acquisition": {
                    "pair_id": "pair.exact",
                    "tx_sensor_id": "tx.exact",
                    "rx_sensor_id": "rx.exact",
                }
            },
        },
        "rx_positions": np.asarray([[99.0, 98.0, 97.0], [1.0, 2.0, 3.0]]),
        "rx_orientations": np.asarray([[0.0, 0.0, 0.0], [np.pi / 2.0, 0.0, 0.0]]),
    }


def test_sensor_pose_uses_exact_typed_pair_row() -> None:
    origin_xy, yaw_deg, origin_z = sensing_figures._extract_sensor_pose(_pose_frame())
    np.testing.assert_array_equal(origin_xy, [1.0, 2.0])
    assert yaw_deg == pytest.approx(90.0)
    assert origin_z == 3.0


def test_sensor_pose_rejects_pair_identity_mismatch() -> None:
    frame = _pose_frame()
    frame["_typed_sensing"]["pair_axis"][0]["rx_sensor_id"] = "rx.wrong"
    with pytest.raises(ValueError, match="rx_sensor_id disagrees"):
        sensing_figures._extract_sensor_pose(frame)


def test_xy_truth_history_is_semantic_id_keyed_and_order_invariant() -> None:
    frame = _pose_frame(frame_pair_index=1)
    frame["_frame_index"] = 7
    frame["target_pos"] = np.asarray([[999.0, 999.0, 999.0]])
    frame["_typed_sensing"]["truth"] = {
        "targets": [
            {
                "target_id": "target.beta",
                "lifecycle": "active",
                "valid": True,
                "position_m": [4.0, 5.0, 6.0],
            },
            {
                "target_id": "target.alpha",
                "lifecycle": "active",
                "valid": True,
                "position_m": [1.0, 2.0, 3.0],
            },
        ]
    }

    tracks, radar_history = sensing_figures._build_xy_ground_truth_history([frame])

    assert set(tracks) == {"target.alpha", "target.beta"}
    assert tracks["target.alpha"] == [{"frame": 7, "x_m": 1.0, "y_m": 2.0, "z_m": 3.0}]
    assert tracks["target.beta"] == [{"frame": 7, "x_m": 4.0, "y_m": 5.0, "z_m": 6.0}]
    assert radar_history == [{"frame": 7, "x_m": 1.0, "y_m": 2.0}]


def _sensing_payload(
    *,
    acquisition_name: str,
    sample_times_s: list[float],
    processing_strategy: str,
    payload: dict,
    phase_coherent: bool,
) -> dict:
    sample_times = np.asarray(sample_times_s, dtype=np.float64)
    sample_periods = np.diff(sample_times)
    if sample_periods.size == 0 or not np.allclose(
        sample_periods,
        sample_periods[0],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Fixture sample times must describe a uniform sampled acquisition")
    sample_period_s = float(sample_periods[0])
    sample_rate_hz = 1.0 / sample_period_s
    supplied_config = payload.get("config")
    if not isinstance(supplied_config, dict):
        raise TypeError("Fixture payload must provide an exact config mapping")
    range_source = payload.get("slow_time_matrix", payload.get("cir"))
    if not isinstance(range_source, np.ndarray):
        raise TypeError("Fixture payload must provide a delay-domain numeric product")
    fft_size_range = int(range_source.shape[-1])
    if supplied_config.get("fft_size_range") != fft_size_range:
        raise ValueError("Fixture fft_size_range must match the stored delay-domain product")
    sampling_frequency = supplied_config.get("sampling_frequency")
    if not isinstance(sampling_frequency, (int, float)) or float(sampling_frequency) <= 0.0:
        raise ValueError("Fixture sampling_frequency must be explicit and positive")

    sample_count = len(sample_times_s)
    if processing_strategy == "coherent_cpi_fft":
        processing_mode = "coherent_cpi"
        sample_source = "coherent_one_solve"
        cpi_hop_samples = sample_count
        rt_solve_stride = sample_count
        phase_contract = "fixture_intra_solve_phase_coherent"
    elif processing_strategy == "sequential_stft":
        processing_mode = "sequential_stft"
        sample_source = "independent_solves"
        cpi_hop_samples = 1
        rt_solve_stride = 1
        phase_contract = "fixture_independent_solve_phase"
    else:
        raise ValueError(f"Unsupported fixture processing strategy: {processing_strategy!r}")

    processing_recipe = {
        "window_range": "hann",
        "window_doppler": "hann",
    }
    config_resolved = {
        "contract_version": 1,
        "fft_size_range": fft_size_range,
        "fft_size_doppler": sample_count,
        "sampling_frequency": float(sampling_frequency),
        "prf_hz": sample_rate_hz,
        "carrier_freq_hz": 28e9,
        "range_mode": "bistatic",
        "processing_mode": processing_mode,
        "processing_recipe": processing_recipe,
        "acquisition": {
            "sample_source": sample_source,
            "rt_solve_stride_scene_steps": rt_solve_stride,
            "sample_period_s": sample_period_s,
            "sample_rate_hz": sample_rate_hz,
            "cpi_length_samples": sample_count,
            "cpi_hop_samples": cpi_hop_samples,
            "overlap_samples": sample_count - cpi_hop_samples,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": phase_contract,
        },
        "processing": {
            "mode": processing_mode,
            "strategy": processing_strategy,
            "fft_size_doppler": sample_count,
            "result_stride_samples": cpi_hop_samples,
            "product_semantics": "typed_delay_domain_slow_time",
            "evidence_status": "physical_time_figure_fixture",
        },
        "topology": {
            "pair_count": 1,
            "pair_index": 0,
            "tx_device_count": 1,
            "rx_device_count": 1,
            "tx_elements": 1,
            "rx_elements": 1,
            "rx_array_processing": "siso",
        },
    }
    samples = [
        {
            "sample_index": sample_index,
            "sample_time_s": sample_time_s,
            "valid": True,
            "phase_coherent": phase_coherent,
            "source": {
                "source_generation_id": _GENERATION_ID,
                "scene_frame_id": f"{acquisition_name}-scene-{sample_index}",
                "sample_offset": sample_index,
            },
        }
        for sample_index, sample_time_s in enumerate(sample_times_s)
    ]
    reference_time_s = 0.5 * (sample_times_s[0] + sample_times_s[-1])
    sensing = {
        **payload,
        "schema_version": SENSING_SCHEMA_VERSION,
        "processing_strategy": processing_strategy,
    }
    sensing["config_resolved"] = config_resolved
    sensing["config"] = deepcopy(config_resolved)
    sensing["temporal"] = build_sensing_temporal_contract(
        clock_id=_GENERATION_ID,
        acquisition_id=str(uuid5(UUID(_GENERATION_ID), acquisition_name)),
        acquisition_config_id=canonical_sensing_hash(config_resolved),
        pair_id=canonical_sensing_pair_id("figures-tx", "figures-rx"),
        tx_sensor_id="figures-tx",
        rx_sensor_id="figures-rx",
        samples=samples,
        processing_run_id=_PROCESSING_RUN_ID,
        processing_recipe_id=canonical_sensing_hash(processing_recipe),
        reference_time_s=reference_time_s,
        reference_time_policy="cpi_center",
        display_scene_frame_id=f"{acquisition_name}-display",
        display_policy="fixture_binding",
        display_reason="focused physical-time test",
    )
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    rx_element_axis = catalog_axis(
        "rx_element",
        size=1,
        catalog_key="rx_element_axis",
        coordinate_semantics="sionna_rx_element_index",
        reference_frame="rx_array_local",
    )
    sensing["rx_element_axis"] = [{"element_index": 0, "row": 0, "column": 0}]

    cir = sensing.get("cir")
    slow_time = sensing.get("slow_time_matrix")
    range_size = None
    if isinstance(cir, np.ndarray):
        range_size = int(cir.shape[-1])
        sensing["cir_by_pair_rx_element"] = cir[np.newaxis, np.newaxis, :]
    if isinstance(slow_time, np.ndarray):
        range_size = int(slow_time.shape[-1])
        sensing["slow_time_by_pair_rx_element"] = slow_time[np.newaxis, np.newaxis, :, :]

    if range_size is not None:
        sensing["range_axis_m"] = np.arange(range_size, dtype=np.float64)
        range_axis = dense_axis(
            "range",
            size=range_size,
            unit="m",
            coordinate_key="range_axis_m",
            coordinate_semantics="fixture_range_bin_centers",
        )
        if isinstance(cir, np.ndarray):
            add_product_spec(
                sensing,
                data_key="cir_by_pair_rx_element",
                kind="delay_domain_cir",
                axes=(pair_axis, rx_element_axis, range_axis),
                representation="complex_voltage",
                unit="1",
                combine={"policy": "siso_identity", "normalization": "identity"},
            )
        if isinstance(slow_time, np.ndarray):
            slow_time_coordinate_key = "slow_time_axis_s"
            if slow_time.shape[0] != len(sample_times_s):
                # This negative fixture deliberately records fewer coherent
                # rows than the acquisition declares. Keep the typed product
                # internally self-consistent so load_cir_sequence owns the
                # cross-record membership rejection exercised by the test.
                slow_time_coordinate_key = "stored_slow_time_axis_s"
                sensing[slow_time_coordinate_key] = np.asarray(
                    sample_times_s[: slow_time.shape[0]], dtype=np.float64
                )
            add_product_spec(
                sensing,
                data_key="slow_time_by_pair_rx_element",
                kind="delay_domain_slow_time",
                axes=(
                    pair_axis,
                    rx_element_axis,
                    dense_axis(
                        "slow_time",
                        size=int(slow_time.shape[0]),
                        unit="s",
                        coordinate_key=slow_time_coordinate_key,
                        coordinate_semantics="scenario_clock_sample_time",
                        reference_frame="scenario_clock",
                    ),
                    range_axis,
                ),
                representation="complex_voltage",
                unit="1",
                combine={"policy": "siso_identity", "normalization": "identity"},
            )
        sensing.pop("cir", None)
        sensing.pop("slow_time_matrix", None)
    materialize_product_specs(sensing)
    finalize_sensing_v4_publication(sensing)
    return sensing


def _write_frame_set(frames_dir: Path, sensing: dict) -> None:
    frames_dir.mkdir(parents=True)
    frame = standard_mpc_frame_from_pair_data(
        frame_index=137,
        timestamp_s=999.0,
        tx_rx_pairs=np.asarray([[0, 0]], dtype=np.int32),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray([[1.0, 0.0, 1.0]], dtype=np.float64),
        tx_names=("figures-tx",),
        rx_names=("figures-rx",),
        vertices_by_pair=[np.zeros((1, 0, 3), dtype=np.float32)],
        interactions_by_pair=[np.zeros((1, 0), dtype=np.int32)],
        path_lengths_by_pair=[np.zeros((1,), dtype=np.int64)],
        sensing=sensing,
    )
    chunk = write_packed_mpc_frame_chunk(
        frames_dir,
        [frame],
        generation_id=_GENERATION_ID,
        compression=None,
    )
    manifest = manifest_from_chunks(
        generation_id=_GENERATION_ID,
        frame_set_id="sensing-figures-temporal-frame-set",
        chunks=[chunk],
        compression={"configured": None, "filter": "none", "shuffle": False},
        segmentation={"max_frames": 1},
        provenance={"fixture": "sensing-figures-temporal"},
        created_utc="2026-08-26T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)


def test_load_cir_sequence_aligns_coherent_rows_with_ordered_sample_times(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    matrix = np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64)
    sensing = _sensing_payload(
        acquisition_name="coherent",
        sample_times_s=[5.0, 5.1, 5.2],
        processing_strategy="coherent_cpi_fft",
        phase_coherent=True,
        payload={
            "slow_time_matrix": matrix,
            "config": {
                "fft_size_range": 4,
                "sampling_frequency": 100e6,
            },
        },
    )
    _write_frame_set(frames_dir, sensing)

    samples, metadata = sensing_figures.load_cir_sequence(frames_dir)

    np.testing.assert_array_equal(np.stack(samples), matrix)
    assert metadata["sample_times_s"] == pytest.approx([5.0, 5.1, 5.2])
    assert metadata["sample_time_source"] == "typed_delay_domain_product_coordinates"
    assert metadata["acquisition_sample_counts"] == [3]
    np.testing.assert_array_equal(metadata["range_axis_m"], [0.0, 1.0, 2.0, 3.0])


def test_load_cir_sequence_prefers_exact_typed_slow_time_window(
    tmp_path: Path,
) -> None:
    frames_dir = tmp_path / "frames"
    cir = np.arange(4, dtype=np.float32).astype(np.complex64)
    sensing = _sensing_payload(
        acquisition_name="sequential",
        sample_times_s=[2.0, 2.1, 2.2],
        processing_strategy="sequential_stft",
        phase_coherent=False,
        payload={
            "cir": cir,
            # A sequential result may retain its sliding analysis window, but
            # the per-frame CIR corresponds only to the latest sample.
            "slow_time_matrix": np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64),
            "config": {"fft_size_range": 4, "sampling_frequency": 100e6},
        },
    )
    _write_frame_set(frames_dir, sensing)

    samples, metadata = sensing_figures.load_cir_sequence(frames_dir)

    np.testing.assert_array_equal(
        np.stack(samples),
        np.arange(12, dtype=np.float32).reshape(3, 4).astype(np.complex64),
    )
    assert metadata["sample_times_s"] == pytest.approx([2.0, 2.1, 2.2])


def test_load_cir_sequence_rejects_coherent_row_membership_mismatch(tmp_path: Path) -> None:
    frames_dir = tmp_path / "frames"
    sensing = _sensing_payload(
        acquisition_name="bad-count",
        sample_times_s=[1.0, 1.1, 1.2],
        processing_strategy="coherent_cpi_fft",
        phase_coherent=True,
        payload={
            "slow_time_matrix": np.ones((2, 4), dtype=np.complex64),
            "config": {"fft_size_range": 4, "sampling_frequency": 100e6},
        },
    )
    _write_frame_set(frames_dir, sensing)

    with pytest.raises(ValueError, match="do not match ordered temporal membership"):
        sensing_figures.load_cir_sequence(frames_dir)


def test_generate_micro_doppler_rejects_nonuniform_physical_cadence(tmp_path: Path) -> None:
    source_sequence = (
        [np.ones(4, dtype=np.complex64) for _ in range(3)],
        {
            "sample_times_s": [0.0, 0.1, 0.25],
            "range_axis_m": [0.0, 0.5, 1.5, 3.0],
            "fft_size_range": 4,
            "fft_size_doppler": 2,
            "sampling_frequency": 100e6,
            "prf_hz": 1000.0,
        },
    )

    with pytest.raises(ValueError, match="resample onto a uniform physical-time grid"):
        sensing_figures.generate_micro_doppler_figures(
            tmp_path / "unused-frames",
            tmp_path / "output",
            config=SimpleNamespace(mti_enabled=False),
            target_range_m=1.0,
            source_sequence=source_sequence,
        )


def test_generate_micro_doppler_uses_timestamp_cadence_not_configured_prf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[float, float]] = []

    def _capture(**kwargs):
        calls.append((kwargs["prf_hz"], kwargs["frame_duration_s"]))
        return None

    monkeypatch.setattr(sensing_figures, "compute_micro_doppler", lambda _bin, **kw: _capture(**kw))
    monkeypatch.setattr(sensing_figures, "compute_range_collapsed_micro_doppler", _capture)
    source_sequence = (
        [np.ones(4, dtype=np.complex64) for _ in range(4)],
        {
            "sample_times_s": [10.0, 10.25, 10.5, 10.75],
            "range_axis_m": [0.0, 0.5, 1.5, 3.0],
            "acquisition_sample_counts": [1, 1, 1, 1],
            "fft_size_range": 4,
            "fft_size_doppler": 2,
            "sampling_frequency": 100e6,
            "prf_hz": 1000.0,
            "range_mode": "monostatic",
        },
    )

    sensing_figures.generate_micro_doppler_figures(
        tmp_path / "unused-frames",
        tmp_path / "output",
        config=SimpleNamespace(mti_enabled=False),
        target_range_m=1.0,
        stft_window_override=2,
        source_sequence=source_sequence,
    )

    assert calls == [(4.0, 0.25), (4.0, 0.25)]
