"""Lightweight contracts for target mesh sequence end behavior."""

from pathlib import Path

import pytest

from generator.core.target.config import TargetConfig
from generator.core.target.mesh import mesh_sequence_index


def _config(mesh_dir: Path, **overrides) -> TargetConfig:
    values = {
        "name": "Target",
        "mobility": None,
        "mesh_pattern": "mesh_*.ply",
        "mesh_directory": str(mesh_dir),
        "resolved_mesh_directory": mesh_dir.resolve(),
        "_initial_position": (0.0, 0.0, 0.0),
    }
    values.update(overrides)
    return TargetConfig(**values)


def test_target_config_defaults_to_loop_mesh_end_behavior(tmp_path: Path) -> None:
    assert _config(tmp_path).mesh_end_behavior == "loop"


def test_target_config_rejects_unknown_mesh_end_behavior(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="mesh_end_behavior"):
        _config(tmp_path, mesh_end_behavior="reverse")


def test_loop_behavior_preserves_start_stride_and_wraparound() -> None:
    assert [
        mesh_sequence_index(
            5,
            call,
            mesh_start_index=3,
            mesh_frame_stride=2,
        )
        for call in range(4)
    ] == [3, 0, 2, 4]
    assert mesh_sequence_index(5, -1, mesh_start_index=3, mesh_frame_stride=2) == 1


def test_hold_last_behavior_clamps_after_start_and_stride_reach_end() -> None:
    assert [
        mesh_sequence_index(
            5,
            call,
            mesh_start_index=3,
            mesh_frame_stride=2,
            mesh_end_behavior="hold_last",
        )
        for call in range(4)
    ] == [3, 4, 4, 4]


def test_hold_last_behavior_clamps_start_beyond_sequence() -> None:
    assert [
        mesh_sequence_index(
            5,
            call,
            mesh_start_index=8,
            mesh_end_behavior="hold_last",
        )
        for call in range(3)
    ] == [4, 4, 4]
