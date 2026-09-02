from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import yaml

from shared.frames.manifest import manifest_from_chunks, write_frame_manifest_atomic
from shared.frames.packed_hdf5_writer import write_packed_mpc_frame_chunk
from shared.frames.types import StandardMPCFrame
from visualizer.src.benchmarking.scenario_profile import profile_generated_scenario


def _write_scenario_yaml(
    scenario_dir: Path,
    *,
    raytracing_enabled: bool = False,
    expected_targets: int = 0,
    directory: str = "frames",
    pattern: str = "mpc_frames_*.h5",
) -> None:
    (scenario_dir / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 2,
                "raytracing": {"enabled": raytracing_enabled},
                "data": {
                    "mode": "files",
                    "files": {
                        "format": "hdf5",
                        "directory": directory,
                        "pattern": pattern,
                    },
                },
                "actors": {"targets": [{"name": f"T{i}"} for i in range(expected_targets)]},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _frame(
    frame_index: int,
    pair_path_counts: Sequence[int],
    target_count: int = 0,
) -> StandardMPCFrame:
    pair_count = len(pair_path_counts)
    pair_path_offsets = np.zeros((pair_count + 1,), dtype=np.int64)
    np.cumsum(pair_path_counts, dtype=np.int64, out=pair_path_offsets[1:])
    path_count = int(pair_path_offsets[-1])
    return StandardMPCFrame(
        frame_index=frame_index,
        tx_rx_pairs=np.asarray([(0, rx_index) for rx_index in range(pair_count)], dtype=np.int32),
        pair_path_offsets=pair_path_offsets,
        bounce_offsets=np.zeros((path_count + 1,), dtype=np.int64),
        tx_positions=np.asarray([[0.0, 0.0, 1.0]], dtype=np.float64),
        rx_positions=np.asarray(
            [[10.0 + rx_index, 0.0, 1.0] for rx_index in range(pair_count)],
            dtype=np.float64,
        ),
        tx_orientations=np.zeros((1, 3), dtype=np.float64),
        rx_orientations=np.zeros((pair_count, 3), dtype=np.float64),
        tx_names=("tx_0",),
        rx_names=tuple(f"rx_{index}" for index in range(pair_count)),
        bounce_xyz_m=np.empty((0, 3), dtype=np.float32),
        interactions=np.empty((0,), dtype=np.uint8),
        material_ids=np.empty((0,), dtype=np.uint16),
        material_names=("",),
        material_itu_types=("",),
        delays_ns=np.full((path_count,), np.nan, dtype=np.float32),
        path_loss_db=np.full((path_count,), np.nan, dtype=np.float32),
        aoa_az_deg=np.full((path_count,), np.nan, dtype=np.float32),
        aoa_el_deg=np.full((path_count,), np.nan, dtype=np.float32),
        aod_az_deg=np.full((path_count,), np.nan, dtype=np.float32),
        aod_el_deg=np.full((path_count,), np.nan, dtype=np.float32),
        metric_valid_bits=np.zeros((path_count,), dtype=np.uint8),
        target_positions_m=np.asarray(
            [[float(index), 0.0, 0.0] for index in range(target_count)],
            dtype=np.float64,
        ).reshape((-1, 3)),
        targets_metadata=tuple({"name": f"T{index}"} for index in range(target_count)),
        provenance={"provider": "benchmark-profile-test", "frame_idx": frame_index},
    )


def _write_frame_set(
    frames_dir: Path,
    frames: Sequence[tuple[Sequence[int], int]],
    *,
    chunk_size: int | None = None,
) -> None:
    frames_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = chunk_size or len(frames)
    chunks = []
    for start in range(0, len(frames), chunk_size):
        group = frames[start : start + chunk_size]
        chunks.append(
            write_packed_mpc_frame_chunk(
                frames_dir,
                (
                    _frame(frame_index, pair_counts, target_count)
                    for frame_index, (pair_counts, target_count) in enumerate(
                        group,
                        start=start,
                    )
                ),
                generation_id="benchmark-profile-test-generation",
                compression=None,
            )
        )
    manifest = manifest_from_chunks(
        generation_id="benchmark-profile-test-generation",
        frame_set_id="benchmark-profile-test-frame-set",
        chunks=chunks,
        compression={"configured": "none", "filter": None, "shuffle": False},
        segmentation={"max_frames": chunk_size},
        provenance={"test": True},
        created_utc="2026-07-29T00:00:00+00:00",
    )
    write_frame_manifest_atomic(frames_dir, manifest)


def _constant_frames(frame_count: int, mpcs_per_frame: int) -> list[tuple[tuple[int], int]]:
    return [((mpcs_per_frame,), 0) for _ in range(frame_count)]


def test_profile_generated_scenario_counts_packed_v2_mpcs(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "synthetic"
    frames_dir = scenario_dir / "frames"
    scenario_dir.mkdir()
    _write_scenario_yaml(scenario_dir)

    _write_frame_set(
        frames_dir,
        _constant_frames(frame_count=10, mpcs_per_frame=7),
        chunk_size=4,
    )

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "ok"
    assert profile["raytracing_enabled"] is False
    assert profile["frame_count"] == 10
    assert [summary["frame_count"] for summary in profile["frame_files"]] == [4, 4, 2]
    assert profile["num_mpcs"]["min"] == 7
    assert profile["num_mpcs"]["max"] == 7
    assert profile["num_mpcs"]["mean"] == 7.0
    assert profile["num_mpcs"]["nonzero_frames"] == 10


def test_profile_generated_scenario_warns_for_constant_raytraced_mpcs(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "raytraced"
    frames_dir = scenario_dir / "frames"
    scenario_dir.mkdir()
    _write_scenario_yaml(scenario_dir, raytracing_enabled=True)

    _write_frame_set(frames_dir, _constant_frames(frame_count=10, mpcs_per_frame=7))

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "warn"
    assert profile["raytracing_enabled"] is True
    assert "suspicious_constant_mpc_count" in profile["issues"]


def test_profile_generated_scenario_counts_pair_paths_and_targets_from_offsets(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "per_pair"
    frames_dir = scenario_dir / "packed" / "frames"
    scenario_dir.mkdir()
    _write_scenario_yaml(
        scenario_dir,
        expected_targets=2,
        directory="packed/frames",
    )
    _write_frame_set(
        frames_dir,
        [
            ((1, 4), 2),
            ((2, 5), 2),
            ((3, 6), 2),
        ],
    )

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "ok"
    assert profile["frame_count"] == 3
    assert profile["num_mpcs"]["sample"] == [5, 7, 9]
    assert profile["num_mpcs"]["max"] == 9
    assert profile["num_targets"]["min"] == 2
    assert profile["num_targets"]["max"] == 2


def test_profile_generated_scenario_normalizes_configured_frame_directory(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "normalized_directory"
    frames_dir = scenario_dir / "packed" / "frames"
    scenario_dir.mkdir()
    _write_scenario_yaml(
        scenario_dir,
        directory=" packed/frames ",
    )
    _write_frame_set(frames_dir, [((3,), 0)])

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "ok"
    assert profile["frame_count"] == 1
    assert profile["frame_files"][0]["file"].startswith(str(frames_dir))


def test_profile_generated_scenario_expands_project_root_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    scenario_dir = project_root / "scenarios" / "profile"
    frames_dir = project_root / "recordings" / "frames"
    scenario_dir.mkdir(parents=True)
    _write_scenario_yaml(
        scenario_dir,
        directory="${PROJECT_ROOT}/recordings/frames",
    )
    _write_frame_set(frames_dir, [((3,), 0)])
    monkeypatch.setattr(
        "visualizer.src.benchmarking.scenario_profile.find_project_root",
        lambda _path: project_root,
    )

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "ok"
    assert profile["frame_files"][0]["file"].startswith(str(frames_dir))


def test_profile_generated_scenario_ignores_files_without_v2_manifest(
    tmp_path: Path,
) -> None:
    scenario_dir = tmp_path / "unmanaged"
    frames_dir = scenario_dir / "frames"
    frames_dir.mkdir(parents=True)
    _write_scenario_yaml(scenario_dir)
    with h5py.File(frames_dir / "mpc_frames_00000-00003.h5", "w") as handle:
        handle.create_dataset("vertices", data=np.zeros((4, 6, 3, 3), dtype=np.float32))
    (frames_dir / "frames_index.json").write_text(
        """{
  "format": "columnar",
  "total_frames": 4
}
""",
        encoding="utf-8",
    )

    profile = profile_generated_scenario(scenario_dir)

    assert profile["status"] == "fail"
    assert profile["frame_count"] == 0
    assert profile["frame_files"] == []
    assert profile["issues"] == ["no_frames"]


def test_profile_generated_scenario_rejects_non_v2_chunk_identity(tmp_path: Path) -> None:
    scenario_dir = tmp_path / "wrong_layout"
    frames_dir = scenario_dir / "frames"
    scenario_dir.mkdir()
    _write_scenario_yaml(scenario_dir)
    _write_frame_set(frames_dir, [((3,), 0)])

    chunk_path = next(frames_dir.glob("mpc_frames_*.h5"))
    with h5py.File(chunk_path, "r+") as handle:
        handle.attrs["storage_layout"] = "ragged_v1"
    manifest_path = frames_dir / "frames_manifest.json"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    manifest["chunks"][0]["size_bytes"] = chunk_path.stat().st_size
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    with np.testing.assert_raises_regex(ValueError, "unsupported HDF5 storage layout"):
        profile_generated_scenario(scenario_dir)
