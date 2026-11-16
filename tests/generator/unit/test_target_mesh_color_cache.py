"""Concurrent cache publication must not fall back to an incompatible PLY."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from generator.core.target import mesh


@pytest.fixture
def color_mesh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    source = tmp_path / "colored.ply"
    source.write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
        "0 0 0 255 0 0\n1 0 0 0 255 0\n0 1 0 0 0 255\n3 0 1 2\n",
        encoding="ascii",
    )
    monkeypatch.setattr(mesh, "_NORMALIZED_PLY_CACHE", {})
    monkeypatch.setattr(mesh, "_NORMALIZED_PLY_CACHE_DIR", tmp_path / "cache")
    return source


def test_concurrent_color_cache_writers_publish_complete_file(
    color_mesh: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    worker_count = 4
    started = Barrier(worker_count)
    written = Barrier(worker_count)
    temporary_paths: list[Path] = []
    payload = b"complete normalized PLY payload\n"

    def write_conversion(source: Path, destination: Path) -> None:
        assert source == color_mesh
        temporary_paths.append(destination)
        # Hold every writer inside conversion before any cache result exists.
        started.wait(timeout=10)
        destination.write_bytes(payload)
        # Force the old shared-temp race: all closes precede all renames.
        written.wait(timeout=10)

    monkeypatch.setattr(mesh, "_write_float_vertex_color_ply", write_conversion)
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        results = list(
            pool.map(
                mesh._prepare_mesh_path_for_mitsuba,
                [str(color_mesh)] * worker_count,
            )
        )

    assert len(set(results)) == 1
    result = Path(results[0])
    assert result != color_mesh
    assert result.read_bytes() == payload
    assert len(set(temporary_paths)) == worker_count
    assert all(path.parent == result.parent for path in temporary_paths)
    assert not any(path.exists() for path in temporary_paths)
    assert list(result.parent.iterdir()) == [result]
    assert mesh._prepare_mesh_path_for_mitsuba(str(color_mesh)) == str(result)
    assert len(temporary_paths) == worker_count


def test_failed_color_conversion_removes_partial_temporary_file(
    color_mesh: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_conversion(source: Path, destination: Path) -> None:
        destination.write_bytes(b"partial PLY")
        raise ValueError("conversion failed")

    monkeypatch.setattr(mesh, "_write_float_vertex_color_ply", fail_conversion)
    assert mesh._prepare_mesh_path_for_mitsuba(str(color_mesh)) == str(color_mesh)
    assert list(mesh._NORMALIZED_PLY_CACHE_DIR.iterdir()) == []
    assert mesh._NORMALIZED_PLY_CACHE == {}
