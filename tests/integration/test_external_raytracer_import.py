"""End-to-end proof for adapting an independent ray-path producer."""

from __future__ import annotations

import ast
from dataclasses import fields
from pathlib import Path

import numpy as np
import pytest

from examples.external_raytracer_import.fake_raytracer import trace
from examples.external_raytracer_import.import_to_orchav import (
    publish_frames,
    to_standard_frame,
)
from shared.frames import StandardMPCFrame
from shared.frames.provider_base import DataProvider
from shared.frames.providers import Hdf5Provider
from visualizer.src.io.packed_frame_payload import (
    projection_to_visual_frame,
    visual_frame_read_request,
)
from visualizer.src.metrics.scenario_statistics import (
    SCENARIO_STATISTICS_REQUEST,
    ScenarioStatisticsAccumulator,
)


def _assert_frame_equal(actual: StandardMPCFrame, expected: StandardMPCFrame) -> None:
    """Compare every canonical field without relying on dataclass array equality."""

    for field in fields(StandardMPCFrame):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value, field.name


def _published_bytes(destination: Path) -> dict[str, bytes]:
    """Capture the complete public frame-set inventory for mutation checks."""

    return {
        path.name: path.read_bytes() for path in sorted(destination.iterdir()) if path.is_file()
    }


def test_fake_producer_is_independent_of_orchav_and_storage() -> None:
    """Keep the source-side proof usable without ORCHAV or array libraries."""

    source_path = Path("examples/external_raytracer_import/fake_raytracer.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.partition(".")[0])

    assert imported_roots == {"__future__", "dataclasses", "math"}


def test_external_records_normalize_to_compact_canonical_frames() -> None:
    """Exercise pair grouping, unit conversion, materials, metrics, and targets."""

    records = trace()
    first = to_standard_frame(records[0])
    second = to_standard_frame(records[1])

    np.testing.assert_array_equal(first.tx_rx_pairs, [[0, 1], [1, 0]])
    np.testing.assert_array_equal(first.pair_path_offsets, [0, 2, 3])
    np.testing.assert_array_equal(first.bounce_offsets, [0, 0, 1, 2])
    np.testing.assert_array_equal(first.interactions, [1, 1])
    assert first.material_names == ("", "painted_concrete", "window_glass")
    assert first.material_itu_types == ("", "concrete", "glass")
    np.testing.assert_array_equal(first.material_ids, [1, 2])

    np.testing.assert_allclose(first.delays_ns, [45.0, 52.0, 48.0])
    np.testing.assert_allclose(first.path_loss_db, [61.0, 69.0, 66.0])
    np.testing.assert_allclose(first.aoa_az_deg, [180.0, 152.0, -150.0])
    np.testing.assert_allclose(first.aoa_el_deg, [0.0, 3.0, 4.0])
    np.testing.assert_allclose(first.aod_az_deg, [34.0, -18.0, 162.0])
    np.testing.assert_allclose(first.aod_el_deg, [0.0, 2.0, 3.0])
    np.testing.assert_array_equal(first.metric_valid_bits, [63, 63, 63])

    assert first.tx_names == ("tx_west", "tx_east")
    assert first.rx_names == ("rx_west", "rx_east")
    np.testing.assert_allclose(first.tx_orientations[1], [np.pi, 0.0, 0.0])
    np.testing.assert_allclose(first.target_positions_m, [[6.0, 4.0, 0.75]])
    np.testing.assert_allclose(second.target_positions_m, [[6.1, 4.0, 0.75]])
    assert first.targets_metadata[0]["name"] == "delivery_cart"
    assert first.provenance == {
        "source": "fake_external_raytracer",
        "source_frame_index": 0,
    }


def test_external_import_round_trips_through_all_consumer_seams(tmp_path: Path) -> None:
    """Publish once, reload canonically, and verify visual/statistics semantics."""

    records = trace()
    expected = tuple(to_standard_frame(record) for record in records)
    destination = tmp_path / "external_frames"

    publish_frames(records, destination)
    assert (destination / "frames_manifest.json").is_file()
    assert list(destination.glob("*.h5"))

    with Hdf5Provider(destination.parent, frames_subdir=destination.name) as provider:
        assert isinstance(provider, DataProvider)
        assert provider.list_frames() == [0, 1]
        for frame_id, expected_frame in enumerate(expected):
            _assert_frame_equal(provider.load_frame(frame_id), expected_frame)

        visual_projection = provider.load_frame_projection(0, visual_frame_read_request())
        visual_payload = projection_to_visual_frame(visual_projection)
        canonical = visual_payload["canonical_data"]
        assert canonical.points.shape == (8, 3)
        assert canonical.lines.shape == (5, 2)
        np.testing.assert_array_equal(canonical.path_orders, [0, 1, 1])
        np.testing.assert_array_equal(canonical.path_tx, [0, 0, 1])
        np.testing.assert_array_equal(canonical.path_rx, [1, 1, 0])
        np.testing.assert_allclose(canonical.path_delays, expected[0].delays_ns)
        np.testing.assert_allclose(canonical.path_losses, expected[0].path_loss_db)
        assert canonical.material_id_to_name == {
            0: "",
            1: "painted_concrete",
            2: "window_glass",
        }
        np.testing.assert_allclose(visual_payload["target_pos"], [[6.0, 4.0, 0.75]])

        projections = provider.iter_frame_projections(
            provider.list_frames(),
            SCENARIO_STATISTICS_REQUEST,
        )
        stats = ScenarioStatisticsAccumulator().collect_from_projections(
            projections,
            total_frames=2,
        )

    assert stats["total_mpcs"] == 6
    assert stats["reflection_order_dist"] == {0: 2, 1: 4}
    assert stats["mpc_type_dist"] == {0: 2, 1: 4}
    np.testing.assert_allclose(
        np.sort(stats["delay_values"]),
        np.sort([45.0, 52.0, 48.0, 45.5, 52.5, 48.5]),
    )
    np.testing.assert_allclose(
        np.sort(stats["path_loss_values"]),
        np.sort([61.0, 69.0, 66.0, 61.5, 69.5, 66.5]),
    )

    before = _published_bytes(destination)
    with pytest.raises(FileExistsError):
        publish_frames(records, destination)
    assert _published_bytes(destination) == before
