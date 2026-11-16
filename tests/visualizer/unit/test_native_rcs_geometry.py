"""Retained native point identity and absorbing proxy geometry contracts."""

import numpy as np
import pytest

from visualizer.src.scene.native_target_geometry import (
    native_proxy_payload,
    native_scattering_points,
)


def test_box_uses_effective_dimensions_without_applying_scale_twice():
    payload = native_proxy_payload(
        {
            "geometry": {
                "type": "box",
                "length_m": 4,
                "width_m": 2,
                "height_m": 1,
                "scaling": [2, 2, 2],
            }
        }
    )
    np.testing.assert_allclose(np.ptp(payload.vertices, axis=0), [4, 2, 1])
    np.testing.assert_allclose(payload.vertices.mean(axis=0), [0, 0, 0])
    assert len(payload.triangles) == 12
    assert not payload.vertices.flags.writeable


def test_points_preserve_exported_order_ids_and_world_coordinates():
    metadata = {
        "scattering_point_ids": [4, 1],
        "scattering_points_world_m": [[12, 3, 4], [-1, 7, 9]],
        "scattering_points_local_m": [[0, 0, 1], [1, 0, 0]],
        "current_position": [900, 900, 900],
    }
    ids, points = native_scattering_points(metadata)
    assert ids == (4, 1)
    np.testing.assert_array_equal(points, metadata["scattering_points_world_m"])
    _, local = native_scattering_points(metadata, world=False)
    np.testing.assert_array_equal(local, metadata["scattering_points_local_m"])


def test_missing_points_do_not_invent_native_preset_scatterers():
    ids, points = native_scattering_points({"representation": "tr38901"})
    assert ids == ()
    assert points.shape == (0, 3)


@pytest.mark.parametrize(
    "ids,points",
    [
        ([0, 0], [[0, 0, 0], [1, 0, 0]]),
        ([1], [[0, 0, 0], [1, 0, 0]]),
        ([True], [[0, 0, 0]]),
        ([0], [[float("nan"), 0, 0]]),
    ],
)
def test_bad_point_evidence_is_rejected(ids, points):
    with pytest.raises(ValueError):
        native_scattering_points({"scattering_point_ids": ids, "scattering_points_world_m": points})


def test_unresolved_native_dimensions_are_not_guessed():
    with pytest.raises(ValueError, match="effective dimensions"):
        native_proxy_payload({"geometry": {"type": "box"}})
