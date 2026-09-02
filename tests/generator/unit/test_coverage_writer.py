import json
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from generator.io.storage import coverage_writer


def _coverage_payload() -> dict:
    return {
        "grid_origin": np.array([0.0, 0.0, 1.5], dtype=np.float32),
        "grid_spacing": np.array([5.0, 5.0], dtype=np.float32),
        "grid_shape": np.array([3, 2, 1], dtype=np.int32),
        "heights": np.array([1.5], dtype=np.float32),
        "path_gain_linear": np.ones((1, 1, 1, 2, 3), dtype=np.float32),
        "derived": {},
        "metric_name": "best_path_loss_db",
        "tx_positions": np.array([[0.0, 0.0, 10.0]], dtype=np.float32),
        "rx_positions": np.empty((0, 3), dtype=np.float32),
        "tx_names": ["TX1"],
        "rx_names": [],
        "tx_power_dbm": np.array([0.0], dtype=np.float32),
        "value_min": 0.0,
        "value_max": 1.0,
        "metadata": {
            "metrics_store": ["path_gain_linear"],
            "metrics_derived": ["best_path_loss_db"],
        },
    }


def _scenario(tmp_path, *, save: dict) -> SimpleNamespace:
    return SimpleNamespace(
        root=tmp_path,
        coverage_cfg={
            "enabled": True,
            "save": save,
        },
    )


def test_save_coverage_map_propagates_invalid_compression(tmp_path):
    scenario = _scenario(
        tmp_path,
        save={"compression": "not-a-filter"},
    )

    with pytest.raises(ValueError, match="Compression"):
        coverage_writer.save_coverage_map(_coverage_payload(), scenario)


def test_failed_rewrite_preserves_existing_output_and_removes_partial_file(tmp_path):
    output_path = tmp_path / "coverage.h5"
    payload = _coverage_payload()
    coverage_writer.save_coverage_hdf5(payload, output_path, compression=None)
    original_contents = output_path.read_bytes()

    with pytest.raises(ValueError, match="Compression"):
        coverage_writer.save_coverage_hdf5(
            payload,
            output_path,
            compression="not-a-filter",
        )

    assert output_path.read_bytes() == original_contents
    assert list(tmp_path.glob(f".{output_path.name}.*.partial")) == []


def test_save_coverage_map_uses_the_fixed_scenario_location(tmp_path):
    scenario = _scenario(
        tmp_path,
        save={"compression": "none"},
    )

    result = coverage_writer.save_coverage_map(_coverage_payload(), scenario)

    expected = tmp_path / "coverage" / "coverage_maps.h5"
    assert result == str(expected)
    assert expected.is_file()


def test_private_file_collision_retries_without_deleting_existing_path(tmp_path, monkeypatch):
    output_path = tmp_path / "coverage.h5"
    collision = tmp_path / f".{output_path.name}.collision.partial"
    collision.write_bytes(b"belongs to another writer")
    generated_ids = iter(
        [
            SimpleNamespace(hex="collision"),
            SimpleNamespace(hex="available"),
        ]
    )
    monkeypatch.setattr(coverage_writer.uuid, "uuid4", lambda: next(generated_ids))

    coverage_writer.save_coverage_hdf5(
        _coverage_payload(),
        output_path,
        compression=None,
    )

    assert output_path.is_file()
    assert collision.read_bytes() == b"belongs to another writer"
    assert not (tmp_path / f".{output_path.name}.available.partial").exists()


def test_replace_failure_preserves_existing_output_and_original_error(tmp_path, monkeypatch):
    output_path = tmp_path / "coverage.h5"
    payload = _coverage_payload()
    coverage_writer.save_coverage_hdf5(payload, output_path, compression=None)
    original_contents = output_path.read_bytes()
    replace_error = PermissionError("atomic coverage promotion denied")

    def fail_replace(*_args, **_kwargs):
        raise replace_error

    monkeypatch.setattr(coverage_writer.os, "replace", fail_replace)

    with pytest.raises(PermissionError) as caught:
        coverage_writer.save_coverage_hdf5(payload, output_path, compression=None)

    assert caught.value is replace_error
    assert output_path.read_bytes() == original_contents
    assert list(tmp_path.glob(f".{output_path.name}.*.partial")) == []


def test_save_coverage_map_propagates_malformed_path_gain_tensor(tmp_path):
    payload = _coverage_payload()
    payload["path_gain_linear"] = np.ones((2, 3), dtype=np.float32)
    scenario = _scenario(
        tmp_path,
        save={"compression": "none"},
    )

    with pytest.raises(ValueError, match="path_gain_linear has unsupported shape"):
        coverage_writer.save_coverage_map(payload, scenario)


def test_sinr_request_advertises_best_server_and_per_tx_views_without_new_datasets(
    tmp_path,
):
    output_path = tmp_path / "coverage.h5"
    payload = _coverage_payload()
    payload["path_gain_linear"] = np.ones((1, 1, 2, 2, 3), dtype=np.float32)
    payload["tx_positions"] = np.zeros((2, 3), dtype=np.float32)
    payload["tx_names"] = ["West", "East"]
    payload["tx_power_dbm"] = np.asarray([0.0, 3.0], dtype=np.float32)
    payload["metadata"]["metrics_derived"] = ["sinr_db", "sinr_linear"]

    coverage_writer.save_coverage_hdf5(payload, output_path, compression=None)

    with h5py.File(output_path, "r") as h5:
        available = json.loads(h5["metadata"].attrs["available_metrics"])
        assert available == [
            "path_gain_linear/West",
            "path_gain_linear/East",
            "serving_path_gain_linear",
            "sinr_db",
            "sinr_db/West",
            "sinr_db/East",
            "sinr_linear",
            "sinr_linear/West",
            "sinr_linear/East",
            "best_path_loss_db",
        ]
        assert list(h5["values"]) == ["path_gain_linear"]
        assert list(h5["derived"]) == []


def test_save_coverage_map_propagates_filesystem_failure(tmp_path, monkeypatch):
    scenario = _scenario(
        tmp_path,
        save={"compression": "lzf"},
    )

    def fail_write(*_args, **_kwargs):
        raise PermissionError("destination is read-only")

    monkeypatch.setattr(coverage_writer, "save_coverage_hdf5", fail_write)

    with pytest.raises(PermissionError, match="read-only"):
        coverage_writer.save_coverage_map(_coverage_payload(), scenario)


def test_save_coverage_map_requires_enabled_output_to_exist(tmp_path, monkeypatch):
    scenario = _scenario(tmp_path, save={"compression": "lzf"})
    monkeypatch.setattr(coverage_writer, "save_coverage_hdf5", lambda *_args, **_kwargs: None)

    with pytest.raises(OSError, match="without creating the enabled output"):
        coverage_writer.save_coverage_map(_coverage_payload(), scenario)


def test_save_coverage_map_returns_none_only_when_persistence_is_disabled(tmp_path, monkeypatch):
    scenario = _scenario(
        tmp_path,
        save={"data": {"enabled": False}},
    )

    def unexpected_write(*_args, **_kwargs):
        raise AssertionError("disabled coverage persistence must not write")

    monkeypatch.setattr(coverage_writer, "save_coverage_hdf5", unexpected_write)

    assert coverage_writer.save_coverage_map(_coverage_payload(), scenario) is None
    assert not (tmp_path / "coverage" / "coverage_maps.h5").exists()


def test_save_coverage_map_requires_a_scenario_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="requires a concrete scenario root"):
        coverage_writer.save_coverage_map(_coverage_payload(), None)

    assert not (tmp_path / "coverage").exists()
