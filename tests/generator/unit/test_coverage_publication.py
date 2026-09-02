from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from generator.io.storage.coverage_publication import (
    CoveragePublication,
    CoveragePublicationError,
)
from generator.io.storage.coverage_writer import save_coverage_hdf5
from shared.coverage.schema import (
    COVERAGE_FRAME_GENERATION_ID_ATTR,
    COVERAGE_FRAME_SET_ID_ATTR,
)
from shared.frames.manifest import FrameSetManifest


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


def _configuration(
    root,
    *,
    coverage_enabled: bool = True,
    save_enabled: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        root=root,
        coverage_cfg={
            "enabled": coverage_enabled,
            "save": {
                "data": {"enabled": save_enabled},
                "compression": "none",
            },
        },
    )


def _manifest(generation_id: str, frame_set_id: str) -> FrameSetManifest:
    return FrameSetManifest(
        generation_id=generation_id,
        frame_set_id=frame_set_id,
        frame_ids=(),
        chunks=(),
        compression={},
        segmentation={},
        provenance={},
        created_utc="2026-08-05T00:00:00+00:00",
    )


def _canonical_path(root):
    return root / "coverage" / "coverage_maps.h5"


def test_staged_map_is_hidden_until_matching_frames_commit(tmp_path):
    configuration = _configuration(tmp_path)
    publication = CoveragePublication(configuration, generation_id="generation-a")

    staged = publication.stage(_coverage_payload(), configuration)
    staged_path = publication.staging_path

    assert staged == str(staged_path)
    assert staged_path.is_file()
    assert not _canonical_path(tmp_path).exists()

    published = publication.finalize(_manifest("generation-a", "frame-set-a"))

    assert published == _canonical_path(tmp_path)
    assert publication.state == "finalized"
    assert not staged_path.exists()
    with h5py.File(published, "r") as coverage_file:
        assert coverage_file.attrs[COVERAGE_FRAME_GENERATION_ID_ATTR] == "generation-a"
        assert coverage_file.attrs[COVERAGE_FRAME_SET_ID_ATTR] == "frame-set-a"


def test_failed_frame_run_discards_staging_and_preserves_prior_map(tmp_path):
    destination = _canonical_path(tmp_path)
    destination.parent.mkdir()
    save_coverage_hdf5(
        _coverage_payload(),
        destination,
        compression=None,
        frame_generation_id="generation-old",
        frame_set_id="frame-set-old",
    )
    prior_bytes = destination.read_bytes()
    configuration = _configuration(tmp_path)
    publication = CoveragePublication(configuration, generation_id="generation-new")
    publication.stage(_coverage_payload(), configuration)

    publication.abort()

    assert destination.read_bytes() == prior_bytes
    assert publication.state == "aborted"
    assert list(tmp_path.glob(".orchav-coverage-s-*.h5")) == []


def test_committed_frames_without_coverage_remove_prior_owned_map(tmp_path):
    destination = _canonical_path(tmp_path)
    destination.parent.mkdir()
    save_coverage_hdf5(_coverage_payload(), destination, compression=None)
    configuration = _configuration(tmp_path, save_enabled=False)
    publication = CoveragePublication(configuration, generation_id="generation-new")

    assert publication.stage(_coverage_payload(), configuration) is None
    assert publication.finalize(_manifest("generation-new", "frame-set-new")) is None

    assert publication.state == "finalized"
    assert not destination.exists()


def test_coverage_disabled_replacement_removes_prior_owned_map(tmp_path):
    destination = _canonical_path(tmp_path)
    destination.parent.mkdir()
    save_coverage_hdf5(_coverage_payload(), destination, compression=None)
    configuration = _configuration(tmp_path, coverage_enabled=False)
    publication = CoveragePublication(configuration, generation_id="generation-new")

    assert publication.finalize(_manifest("generation-new", "frame-set-new")) is None

    assert publication.state == "finalized"
    assert not destination.exists()


def test_coverage_removal_refuses_an_unrecognized_fixed_file(tmp_path):
    destination = _canonical_path(tmp_path)
    destination.parent.mkdir()
    destination.write_bytes(b"belongs to the scenario author")
    publication = CoveragePublication(
        _configuration(tmp_path, save_enabled=False),
        generation_id="generation-new",
    )

    with pytest.raises(CoveragePublicationError, match="not recognized"):
        publication.finalize(_manifest("generation-new", "frame-set-new"))

    assert destination.read_bytes() == b"belongs to the scenario author"


def test_staged_map_rejects_a_different_committed_generation(tmp_path):
    configuration = _configuration(tmp_path)
    publication = CoveragePublication(configuration, generation_id="generation-a")
    publication.stage(_coverage_payload(), configuration)

    with pytest.raises(CoveragePublicationError, match="does not match"):
        publication.finalize(_manifest("generation-b", "frame-set-b"))

    assert publication.state == "aborted"
    assert not _canonical_path(tmp_path).exists()


def test_invalid_private_writer_output_is_removed(tmp_path, monkeypatch):
    configuration = _configuration(tmp_path)
    publication = CoveragePublication(configuration, generation_id="generation-a")
    staging_path = publication.staging_path

    def write_invalid_output(*_args, output_path, **_kwargs):
        output_path.write_bytes(b"not HDF5")
        return str(output_path)

    monkeypatch.setattr(
        "generator.io.storage.coverage_publication.save_coverage_map",
        write_invalid_output,
    )

    with pytest.raises(CoveragePublicationError, match="expected private"):
        publication.stage(_coverage_payload(), configuration)

    assert publication.state == "aborted"
    assert not staging_path.exists()


def test_child_scenarios_publish_coverage_independently(tmp_path):
    roots = [tmp_path / "study-a", tmp_path / "study-b"]
    for root in roots:
        root.mkdir()

    def publish(index: int):
        generation_id = f"generation-{index}"
        configuration = _configuration(roots[index])
        publication = CoveragePublication(configuration, generation_id=generation_id)
        publication.stage(_coverage_payload(), configuration)
        return publication.finalize(_manifest(generation_id, f"frame-set-{index}"))

    with ThreadPoolExecutor(max_workers=2) as executor:
        published = list(executor.map(publish, range(2)))

    assert published == [_canonical_path(root) for root in roots]
    assert all(path.is_file() for path in published)
