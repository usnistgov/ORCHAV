"""Tests for the orchav-inspect CLI."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.cli.inspect import (
    FrameInspectionError,
    _format_sensing_lines,
    _sensing_summary_data,
    run,
)
from shared.extensions.sensing import (
    SENSING_SCHEMA_VERSION,
    build_sensing_temporal_contract,
    canonical_processing_recipe_id,
    canonical_sensing_hash,
)
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.frames.types import StandardMPCFrame
from shared.sensing.products import (
    SENSING_PRODUCT_CONTRACT_VERSION,
    add_product_spec,
    catalog_axis,
    dense_axis,
    materialize_product_specs,
)
from shared.sensing.targets import build_target_catalog, build_target_state, build_target_truth

_SENSING_GENERATION_ID = "a5ef48c3-0c1c-49eb-8890-308e9f9bd8ac"


def _build_frame(
    frame_idx: int = 0,
    sensing: dict | None = None,
) -> StandardMPCFrame:
    return standard_mpc_frame_from_pair_data(
        frame_index=frame_idx,
        provenance={"provider": "test", "frame_idx": frame_idx},
        tx_positions=np.array([[-139.0, 47.5, 40.0]], dtype=np.float64),
        rx_positions=np.array([[-237.0, 67.5, 1.5]], dtype=np.float64),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((1, 3), dtype=np.float64),
        tx_names=("tx-0",),
        rx_names=("rx-0",),
        tx_rx_pairs=np.array([[0, 0]], dtype=np.int32),
        vertices_by_pair=[
            np.array(
                [
                    [[0.0, 0.0, 0.0]],
                    [[-190.0, 55.0, 8.0]],
                ],
                dtype=np.float32,
            )
        ],
        interactions_by_pair=[np.array([[-1], [1]], dtype=np.int32)],
        path_lengths_by_pair=[np.array([0, 1], dtype=np.int64)],
        material_names_by_pair=[np.array([[""], ["mat-itu_concrete"]], dtype=object)],
        material_itu_types_by_pair=[np.array([[""], ["concrete"]], dtype=object)],
        metrics_by_pair={
            "delays_ns": [np.array([1.0, 2.0], dtype=np.float32)],
            "path_loss_db": [np.array([80.0, 95.0], dtype=np.float32)],
            "aoa_az_deg": [np.array([0.0, 5.0], dtype=np.float32)],
            "aoa_el_deg": [np.array([1.0, 2.0], dtype=np.float32)],
            "aod_az_deg": [np.array([3.0, 4.0], dtype=np.float32)],
            "aod_el_deg": [np.array([5.0, 6.0], dtype=np.float32)],
        },
        target_positions_m=np.empty((0, 3), dtype=np.float64),
        targets_metadata=(),
        sensing=sensing,
    )


def _write_scenario(
    root: Path,
    frame_count: int = 1,
    sensing: dict | None = None,
    *,
    directory: str = "frames",
    material_properties: dict | None = None,
) -> Path:
    frames_dir = root / directory
    generation_id = _SENSING_GENERATION_ID if sensing is not None else "inspect-cli-generation"
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id=generation_id,
        compression=None,
    )
    for frame_idx in range(frame_count):
        writer.append(_build_frame(frame_idx, sensing=sensing))
    chunk = writer.finalize_to_range_name()
    provenance = {"fixture": "inspect-cli"}
    if material_properties is not None:
        provenance["material_properties"] = material_properties
    manifest = manifest_from_chunks(
        generation_id=generation_id,
        frame_set_id="inspect-cli-frame-set",
        chunks=[chunk],
        compression={"codec": None, "shuffle": False},
        segmentation={"effective_frame_limit": frame_count},
        provenance=provenance,
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir


def _write_chunked_scenario(root: Path) -> tuple[Path, tuple[Path, ...]]:
    """Write two chunks so exact-file inspection has a distinct frame scope."""
    frames_dir = root / "frames"
    chunks = []
    for frame_ids in ((0, 1), (4, 5)):
        writer = PackedMPCChunkWriter(
            frames_dir,
            generation_id="inspect-cli-generation",
            compression=None,
        )
        for frame_idx in frame_ids:
            writer.append(_build_frame(frame_idx))
        chunks.append(writer.finalize_to_range_name())
    manifest = manifest_from_chunks(
        generation_id="inspect-cli-generation",
        frame_set_id="inspect-cli-frame-set",
        chunks=chunks,
        compression={"codec": None, "shuffle": False},
        segmentation={"effective_frame_limit": 2},
        provenance={"fixture": "inspect-cli"},
        created_utc="2026-08-05T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)
    return frames_dir, tuple(frames_dir / chunk.file for chunk in chunks)


def _write_scenario_yaml(root: Path, *, directory: str | None = None) -> Path:
    data = {
        "schema_version": 2,
        "timeline": {"steps": 1, "duration_s": 0.0},
    }
    if directory is not None:
        data["data"] = {"mode": "files", "files": {"directory": directory}}
    path = root / "scenario.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def _build_sensing_payload(*, ambiguous_products: bool = False) -> dict:
    pair_id = canonical_sensing_hash({"tx_sensor_id": "tx-0", "rx_sensor_id": "rx-0"})
    processing_recipe = {"window_range": "hann", "window_doppler": "hann"}
    temporal = build_sensing_temporal_contract(
        clock_id=_SENSING_GENERATION_ID,
        acquisition_id="cfdd44c8-e4cf-4c0e-8cee-6c67b0e98bc4",
        acquisition_config_id=canonical_sensing_hash({"samples": 4, "prf_hz": 1000.0}),
        pair_id=pair_id,
        tx_sensor_id="tx-0",
        rx_sensor_id="rx-0",
        samples=[
            {
                "sample_index": sample_index,
                "sample_time_s": sample_index / 1000.0,
                "valid": True,
                "phase_coherent": True,
                "source": {
                    "source_generation_id": _SENSING_GENERATION_ID,
                    "scene_frame_id": 0,
                    "sample_offset": sample_index,
                },
            }
            for sample_index in range(4)
        ],
        processing_run_id=_SENSING_GENERATION_ID,
        processing_recipe_id=canonical_processing_recipe_id(processing_recipe),
        reference_time_s=0.0015,
        reference_time_policy="cpi_center",
        display_scene_frame_id=0,
        display_policy="source_scene_exact",
        display_reason="Inspect CLI fixture result is stored on source frame 0.",
        truth={
            "reference_time_s": 0.0015,
            "scene_frame_id": 0,
            "policy": "cpi_reference_exact",
            "source": "inspect_cli_fixture",
        },
    )
    catalog = build_target_catalog(
        [
            {
                "target_id": "inspect.walker.sm",
                "display_name": "Inspect walker SM",
                "identity_source": "inspect_fixture.subject_code",
            }
        ]
    )
    state = build_target_state(
        catalog,
        scene_frame_id=0,
        reference_time_s=0.0015,
        targets=[
            {
                "target_id": "inspect.walker.sm",
                "lifecycle": "active",
                "valid": True,
                "position_m": [2.5, 0.0, 0.0],
                "orientation_rad": None,
                "velocity_m_s": [-1.0, 0.0, 0.0],
                "invalid_reason": None,
            }
        ],
    )
    truth = build_target_truth(
        catalog,
        targets=[
            {
                "target_id": "inspect.walker.sm",
                "lifecycle": "active",
                "valid": True,
                "position_m": [2.5, 0.0, 0.0],
                "orientation_rad": None,
                "velocity_m_s": [-1.0, 0.0, 0.0],
                "invalid_reason": None,
                "pair_observables": [
                    {
                        "pair_id": pair_id,
                        "bistatic_path_length_m": 5.0,
                        "monostatic_equivalent_range_m": 2.5,
                        "path_closing_rate_m_s": 2.0,
                        "monostatic_equivalent_velocity_m_s": 1.0,
                        "bistatic_bisector_velocity_m_s": 1.0,
                        "undefined_reason": None,
                    }
                ],
            }
        ],
    )
    config_resolved = {
        "contract_version": 1,
        "fft_size_range": 8,
        "fft_size_doppler": 4,
        "sampling_frequency": 100.0e6,
        "prf_hz": 1000.0,
        "carrier_freq_hz": 28.0e9,
        "range_mode": "monostatic",
        "processing_mode": "coherent_cpi",
        "processing_recipe": processing_recipe,
        "acquisition": {
            "sample_source": "coherent_one_solve",
            "rt_solve_stride_scene_steps": 4,
            "sample_period_s": 0.001,
            "sample_rate_hz": 1000.0,
            "cpi_length_samples": 4,
            "cpi_hop_samples": 4,
            "overlap_samples": 0,
            "partial_window_policy": "retain_unprocessed",
            "phase_contract": "sionna_intra_solve_phase_coherent",
        },
        "processing": {
            "mode": "coherent_cpi",
            "strategy": "coherent_cpi_fft",
            "fft_size_doppler": 4,
            "result_stride_samples": 4,
            "product_semantics": "cir_point_bin_slow_time_fft",
            "evidence_status": "phase2_controlled_specular_validated",
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
        "range_resolution_m": 0.3,
        "velocity_resolution_m_s": 0.042,
    }
    sensing = {
        "schema_version": SENSING_SCHEMA_VERSION,
        "temporal": temporal,
        "config_resolved": config_resolved,
        "config": deepcopy(config_resolved),
        "processing_strategy": "coherent_cpi_fft",
        "frame_role": "cpi_end",
        "target_catalog": catalog,
        "target_state": state,
        "truth": truth,
        "range_axis_m": np.linspace(0.0, 2.1, 8, dtype=np.float64),
        "doppler_axis_hz": np.linspace(-500.0, 250.0, 4, dtype=np.float64),
        "range_profile_by_pair": np.ones((1, 8), dtype=np.float32),
        "range_doppler_by_pair": np.ones((1, 4, 8), dtype=np.float32),
        "detections": [
            {
                "range_m": 2.4,
                "velocity_m_s": -1.0,
                "power_linear": 1.0,
                "snr_db": 12.0,
            }
        ],
    }
    pair_axis = catalog_axis("pair", size=1, catalog_key="pair_axis")
    range_axis = dense_axis(
        "range",
        size=8,
        unit="m",
        coordinate_key="range_axis_m",
        coordinate_semantics="bistatic_path_length",
        reference_frame="tx_target_rx_path",
    )
    add_product_spec(
        sensing,
        data_key="range_profile_by_pair",
        kind="range_profile",
        axes=(pair_axis, range_axis),
        representation="power_linear",
        unit="1",
    )
    add_product_spec(
        sensing,
        data_key="range_doppler_by_pair",
        kind="range_doppler",
        axes=(
            pair_axis,
            dense_axis(
                "doppler_frequency",
                size=4,
                unit="Hz",
                coordinate_key="doppler_axis_hz",
                reference_frame="scenario_clock",
            ),
            range_axis,
        ),
        representation="power_linear",
        unit="1",
    )
    if ambiguous_products:
        sensing["range_doppler_secondary"] = np.full((1, 4, 8), 2.0, dtype=np.float32)
        add_product_spec(
            sensing,
            data_key="range_doppler_secondary",
            kind="range_doppler",
            axes=(
                pair_axis,
                dense_axis(
                    "doppler_frequency",
                    size=4,
                    unit="Hz",
                    coordinate_key="doppler_axis_hz",
                    reference_frame="scenario_clock",
                ),
                range_axis,
            ),
            representation="power_linear",
            unit="1",
        )
    materialize_product_specs(sensing)
    # This fixture publishes only range/RD products. The materializer exposes
    # the acquisition coordinate for producers that declare a slow-time axis;
    # do not leave that numeric coordinate undeclared here.
    sensing.pop("slow_time_axis_s", None)
    return sensing


def test_inspect_cli_prints_human_readable_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario)

    exit_code = run([str(scenario)])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Frame count: 1" in captured.out
    assert "Selected frame: 0" in captured.out
    assert "TX: 1, RX: 1, Targets: 0" in captured.out
    assert "Total MPCs: 2" in captured.out
    assert "Interaction segments: code 1 (specular)=1" in captured.out
    assert "Paths with interaction: code 1 (specular)=1" in captured.out
    assert "TX1: (-139.000, 47.500, 40.000)" in captured.out
    assert captured.err == ""


def test_inspect_cli_accepts_frames_directory(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    frames_dir = _write_scenario(scenario)

    exit_code = run([str(frames_dir), "--frame", "0"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Frames directory: {frames_dir}" in captured.out
    assert "Selected frame: 0" in captured.out


def test_inspect_cli_exact_chunk_limits_frames_and_uses_its_first_frame(
    tmp_path: Path,
    capsys,
) -> None:
    _frames_dir, chunks = _write_chunked_scenario(tmp_path / "scenario")

    exit_code = run([str(chunks[1]), "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_count"] == 2
    assert payload["available_frames"] == [4, 5]
    assert payload["selected_frame"] == 4
    assert payload["frame_source"]["frame_idx"] == 4


def test_inspect_cli_exact_chunk_rejects_frame_from_another_chunk(
    tmp_path: Path,
    capsys,
) -> None:
    _frames_dir, chunks = _write_chunked_scenario(tmp_path / "scenario")

    exit_code = run([str(chunks[1]), "--frame", "1"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Frame 1 is not available" in captured.err
    assert "Available frames: 4-5" in captured.err


def test_inspect_cli_rejects_exact_file_not_advertised_by_manifest(
    tmp_path: Path,
    capsys,
) -> None:
    frames_dir = _write_scenario(tmp_path / "scenario")
    unadvertised = frames_dir / "external.h5"
    unadvertised.write_bytes(b"not part of the frame set")

    exit_code = run([str(unadvertised)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "external.h5 is not advertised" in captured.err
    assert "frames_manifest.json" in captured.err


def test_inspect_cli_scenario_yaml_uses_default_directory_when_data_is_omitted(
    tmp_path: Path,
    capsys,
) -> None:
    scenario = tmp_path / "scenario"
    frames_dir = _write_scenario(scenario)
    scenario_yaml = _write_scenario_yaml(scenario)

    exit_code = run([str(scenario_yaml), "--frame", "0"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Source: {scenario}" in captured.out
    assert f"Frames directory: {frames_dir}" in captured.out
    assert "Selected frame: 0" in captured.out


def test_inspect_cli_scenario_yaml_uses_custom_frame_directory(
    tmp_path: Path,
    capsys,
) -> None:
    scenario = tmp_path / "scenario"
    frames_dir = _write_scenario(scenario, directory="selected/frames")
    scenario_yaml = _write_scenario_yaml(scenario, directory="selected/frames")

    exit_code = run([str(scenario_yaml), "--frame", "0"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert f"Source: {scenario}" in captured.out
    assert f"Frames directory: {frames_dir}" in captured.out
    assert "Selected frame: 0" in captured.out


def test_inspect_cli_prints_manifest_material_coefficients(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(
        scenario,
        material_properties={
            "schema_version": 1,
            "source": "sionna.rt.Scene.radio_materials",
            "properties": {
                "itu_concrete": {"scattering_coefficient": 0.4},
                "itu_wood": {"scattering_coefficient": 0.1},
            },
        },
    )

    exit_code = run([str(scenario), "--materials"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Scene material scattering coefficients:" in captured.out
    assert "itu_concrete: 0.4" in captured.out
    assert "itu_wood: 0.1" in captured.out
    assert captured.err == ""


def test_inspect_cli_json_includes_manifest_material_coefficients(
    tmp_path: Path,
    capsys,
) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(
        scenario,
        material_properties={
            "schema_version": 1,
            "source": "sionna.rt.Scene.radio_materials",
            "properties": {
                "itu_concrete": {"scattering_coefficient": 0.4},
            },
        },
    )

    exit_code = run([str(scenario), "--materials", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["materials"] == {
        "source": "sionna.rt.Scene.radio_materials",
        "scattering_coefficients": {"itu_concrete": 0.4},
    }


def test_inspect_cli_lists_materials_without_loading_a_frame(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(
        scenario,
        material_properties={
            "schema_version": 1,
            "properties": {"itu_concrete": {"scattering_coefficient": 0.4}},
        },
    )

    exit_code = run([str(scenario), "--list", "--materials"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Selected frame:" not in captured.out
    assert "itu_concrete: 0.4" in captured.out
    assert captured.err == ""


def test_inspect_cli_json_output(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario, frame_count=2)

    exit_code = run([str(scenario), "--frame", "1", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["frame_count"] == 2
    assert payload["available_frames"] == [0, 1]
    assert payload["selected_frame"] == 1
    assert payload["summary"]["total_mpcs"] == 2
    assert payload["summary"]["interaction_counts"]["segments"]["1"] == 1
    assert payload["summary"]["interaction_counts"]["paths"]["1"] == 1
    assert payload["positions"]["rx"] == [[-237.0, 67.5, 1.5]]


def test_inspect_cli_prints_sensing_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    sensing = _build_sensing_payload()
    _write_scenario(scenario, sensing=sensing)

    assert run([str(scenario), "--frame", "0"]) == 0

    output = capsys.readouterr().out
    assert "Frame extension: sensing" in output
    assert "Mode: coherent_cpi (cpi_end)" in output
    assert "Range–Doppler: 1 pair × 4 Doppler bins × 8 range bins" in output
    assert "Range profile: 1 pair × 8 range bins" in output
    assert "Targets: 1\n" in output
    assert "Detections: 1" in output
    assert "range 0.3 m" in output
    assert "--verbose or --json" in output
    assert "representation=" not in output
    assert "range_doppler_by_pair" not in output
    assert sensing["products"]["range_doppler_by_pair"]["product_id"] not in output


def test_inspect_cli_prints_verbose_sensing_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario, sensing=_build_sensing_payload())

    exit_code = run([str(scenario), "--frame", "0", "--verbose"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Frame extension: sensing" in captured.out
    assert "Mode: coherent_cpi (cpi_end)" in captured.out
    assert "Strategy: coherent_cpi_fft" in captured.out
    assert "Products: 2 typed descriptor(s)" in captured.out
    assert "range_doppler/primary: key=range_doppler_by_pair" in captured.out
    assert "shape=1x4x8, axes=pair,doppler_frequency,range" in captured.out
    assert "range_profile/primary: key=range_profile_by_pair" in captured.out
    assert "shape=1x8, axes=pair,range" in captured.out
    assert "Target catalog: 1 (inspect.walker.sm)" in captured.out
    assert "Target state: 1 (inspect.walker.sm)" in captured.out
    assert "Target truth: 1 (inspect.walker.sm)" in captured.out
    assert "Detections: 1" in captured.out
    assert "Raw RD" not in captured.out
    assert "Angular:" not in captured.out
    assert "range 0.3 m" in captured.out


def test_inspect_cli_json_includes_sensing_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    sensing = _build_sensing_payload()
    _write_scenario(scenario, sensing=sensing)

    exit_code = run([str(scenario), "--frame", "0", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["features"] == ["sensing"]
    summary = payload["summary"]["sensing"]
    assert summary["schema_version"] == SENSING_SCHEMA_VERSION
    assert summary["product_contract_version"] == SENSING_PRODUCT_CONTRACT_VERSION
    assert summary["processing_mode"] == "coherent_cpi"
    assert summary["processing_strategy"] == "coherent_cpi_fft"
    assert summary["detection_count"] == 1
    assert [product["data_key"] for product in summary["products"]] == [
        "range_doppler_by_pair",
        "range_profile_by_pair",
    ]
    range_doppler = summary["products"][0]
    assert {key: value for key, value in range_doppler.items() if key != "axes"} == {
        "kind": "range_doppler",
        "variant": "primary",
        "data_key": "range_doppler_by_pair",
        "product_id": sensing["products"]["range_doppler_by_pair"]["product_id"],
        "shape": [1, 4, 8],
        "representation": "power_linear",
        "unit": "1",
    }
    assert [axis["name"] for axis in range_doppler["axes"]] == [
        "pair",
        "doppler_frequency",
        "range",
    ]
    assert [axis["coordinate_key"] for axis in range_doppler["axes"]] == [
        "pair_axis",
        "doppler_axis_hz",
        "range_axis_m",
    ]
    assert range_doppler["axes"] == sensing["products"]["range_doppler_by_pair"]["axes"]
    assert summary["targets"] == {
        "catalog": {
            "count": 1,
            "ids": ["inspect.walker.sm"],
            "catalog_revision": sensing["target_catalog"]["catalog_revision"],
        },
        "state": {
            "count": 1,
            "ids": ["inspect.walker.sm"],
            "valid_ids": ["inspect.walker.sm"],
        },
        "truth": {"count": 1, "ids": ["inspect.walker.sm"]},
    }
    # The conflicting raw array is never exposed as an inferred product.
    assert "range_doppler_shape" not in summary


def test_sensing_summary_is_empty_when_extension_is_absent() -> None:
    assert _sensing_summary_data(None) == {}
    assert _sensing_summary_data({}) == {}


def test_compact_sensing_keeps_variants_and_summarizes_diagnostics() -> None:
    summary = _sensing_summary_data(_build_sensing_payload())
    rd = summary["products"][0]
    summary["products"].extend(
        [
            dict(rd, variant="raw"),
            dict(rd, kind="range_doppler_cfar_mask"),
            dict(rd, kind="range_doppler_snr_db"),
            dict(rd, kind="range_doppler_angle"),
            dict(rd, kind="custom_spectrum"),
        ]
    )

    output = "\n".join(_format_sensing_lines(summary))

    assert "Range–Doppler (raw):" in output
    assert "Detector diagnostics: 2 saved arrays" in output
    assert "range_doppler_cfar_mask" not in output
    assert "Angular spectrum:" in output
    assert "Custom spectrum:" in output


def test_sensing_summary_rejects_legacy_truth_aliases() -> None:
    sensing = _build_sensing_payload()
    sensing["gt_sensing"] = {"target_ranges": [2.5], "target_speeds": [-1.0]}

    with pytest.raises(FrameInspectionError, match="forbids positional/alias truth fields"):
        _sensing_summary_data(sensing)


def test_sensing_summary_rejects_malformed_product_descriptor() -> None:
    sensing = deepcopy(_build_sensing_payload())
    sensing["products"]["range_doppler_by_pair"]["shape"] = [1, 99, 8]

    with pytest.raises(FrameInspectionError, match="shape does not match"):
        _sensing_summary_data(sensing)


def test_sensing_summary_lists_ambiguous_kind_variants_without_selecting() -> None:
    summary = _sensing_summary_data(_build_sensing_payload(ambiguous_products=True))

    matching = [
        product
        for product in summary["products"]
        if product["kind"] == "range_doppler" and product["variant"] == "primary"
    ]
    assert [product["data_key"] for product in matching] == [
        "range_doppler_by_pair",
        "range_doppler_secondary",
    ]
    assert matching[0]["product_id"] != matching[1]["product_id"]


def test_inspect_cli_distinguishes_ambiguous_products(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario, sensing=_build_sensing_payload(ambiguous_products=True))

    assert run([str(scenario)]) == 0

    output = capsys.readouterr().out
    for key in ("range_doppler_by_pair", "range_doppler_secondary"):
        assert f"Range–Doppler [{key}]: 1 pair × 4 Doppler bins × 8 range bins" in output


def test_inspect_cli_list_mode_does_not_load_frame(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario, frame_count=3)

    exit_code = run([str(scenario), "--list"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Frame count: 3" in captured.out
    assert "Available frames: 0-2" in captured.out
    assert "Selected frame" not in captured.out


def test_inspect_cli_reports_missing_frame(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario)

    exit_code = run([str(scenario), "--frame", "10"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Frame 10 is not available" in captured.err


def test_inspect_cli_prints_coverage_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    coverage_dir = scenario / "coverage"
    coverage_dir.mkdir(parents=True)
    coverage_file = coverage_dir / "coverage_maps.h5"
    path_gain = np.ones((1, 1, 1, 2, 3), dtype=np.float32) * 1e-8
    save_coverage_hdf5(
        {
            "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
            "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
            "grid_shape": np.array([3, 2, 1], dtype=np.int32),
            "heights": np.array([1.5], dtype=np.float32),
            "path_gain_linear": path_gain,
            "derived": {
                "best_path_loss_db": np.ones((1, 1, 2, 3), dtype=np.float32) * 80.0,
                "serving_tx": np.zeros((1, 1, 2, 3), dtype=np.int16),
            },
            "metric_name": "best_path_loss_db",
            "tx_positions": np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
            "rx_positions": np.empty((0, 3), dtype=np.float32),
            "tx_names": ["TX1"],
            "rx_names": [],
            "tx_power_dbm": np.array([0.0], dtype=np.float32),
            "value_min": 80.0,
            "value_max": 80.0,
            "metadata": {
                "tx_mode": "per_tx",
                "metrics_store": ["path_gain_linear"],
                "metrics_derived": ["path_loss_db", "serving_tx"],
                "noise_power_w": 1e-12,
                "bandwidth_hz": 2e9,
                "temperature_k": 293.0,
                "solver": {"los": True, "max_depth": 4},
            },
        },
        coverage_file,
        compression=None,
    )

    exit_code = run([str(scenario), "--coverage"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Coverage schema: v2" in captured.out
    assert "Grid: 3×2×1" in captured.out
    assert "TX: 1 (TX1)" in captured.out
    assert "RX: 0 (none)" in captured.out
    assert "best_path_loss_db" in captured.out

    exit_code = run([str(scenario), "--coverage", "--json"])

    assert exit_code == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["solver"]["los"] is True
    assert payload["solver"]["max_depth"] == 4
