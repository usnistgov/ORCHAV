"""Tests for the orchav-inspect CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.cli.inspect import run
from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5_writer import PackedMPCChunkWriter
from shared.frames.types import StandardMPCFrame


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
    writer = PackedMPCChunkWriter(
        frames_dir,
        generation_id="inspect-cli-generation",
        compression=None,
    )
    for frame_idx in range(frame_count):
        writer.append(_build_frame(frame_idx, sensing=sensing))
    chunk = writer.finalize_to_range_name()
    provenance = {"fixture": "inspect-cli"}
    if material_properties is not None:
        provenance["material_properties"] = material_properties
    manifest = manifest_from_chunks(
        generation_id="inspect-cli-generation",
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


def _build_sensing_payload() -> dict:
    return {
        "config": {
            "processing_mode": "coherent_cpi",
            "range_resolution_m": 0.3,
            "velocity_resolution_m_s": 0.042,
            "prf_hz": 1000.0,
        },
        "frame_role": "cpi_end",
        "range_profile": np.ones(8, dtype=np.float32),
        "range_doppler_map": np.ones((4, 8), dtype=np.float32),
        "range_doppler_map_raw": np.ones((4, 8), dtype=np.float32) * 2,
        "range_doppler_aoa_az_map": np.ones((4, 8), dtype=np.float32) * 15.0,
        "range_doppler_azimuth_cube": np.ones((16, 4, 8), dtype=np.float32),
        "azimuth_axis_deg": np.linspace(-60.0, 60.0, 16, dtype=np.float32),
        "detections": [
            {
                "range_m": 2.4,
                "velocity_m_s": -1.0,
                "power_linear": 1.0,
                "snr_db": 12.0,
            }
        ],
        "gt_sensing": {
            "target_ranges": [2.5],
            "target_speeds": [-1.0],
        },
    }


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
    _write_scenario(scenario, sensing=_build_sensing_payload())

    exit_code = run([str(scenario), "--frame", "0"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert "Frame extension: sensing" in captured.out
    assert "Mode: coherent_cpi (cpi_end)" in captured.out
    assert "Products: RD 4x8, range profile 8" in captured.out
    assert "Raw RD: 4x8" in captured.out
    assert "Angular: azimuth cube 16x4x8, AoA az map 4x8, az axis 16" in captured.out
    assert "Detections: 1 (GT targets: 1)" in captured.out
    assert "range 0.3 m" in captured.out


def test_inspect_cli_json_includes_sensing_summary(tmp_path: Path, capsys) -> None:
    scenario = tmp_path / "scenario"
    _write_scenario(scenario, sensing=_build_sensing_payload())

    exit_code = run([str(scenario), "--frame", "0", "--json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["features"] == ["sensing"]
    assert payload["summary"]["sensing"]["processing_mode"] == "coherent_cpi"
    assert payload["summary"]["sensing"]["range_doppler_shape"] == [4, 8]
    assert payload["summary"]["sensing"]["azimuth_cube_shape"] == [16, 4, 8]
    assert payload["summary"]["sensing"]["aoa_azimuth_shape"] == [4, 8]
    assert payload["summary"]["sensing"]["detection_count"] == 1


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
