from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from generator.core.materials.path_metadata import materials_per_bounce
from shared.frames.sionna_metadata import (
    SIONNA_INTERACTION_DIFFRACTION,
    SIONNA_INTERACTION_DIFFUSE,
    SIONNA_INTERACTION_REFRACTION,
    SIONNA_INTERACTION_SPECULAR,
    SIONNA_INVALID_OBJECT_ID,
)


class _Tensor:
    def __init__(self, data: np.ndarray) -> None:
        self._data = data

    def numpy(self) -> np.ndarray:
        return self._data


@dataclass
class _Material:
    name: Any
    itu_type: Any = None


@dataclass
class _SceneObject:
    object_id: Any
    radio_material: Any = None


@dataclass
class _Scene:
    objects: dict[str, _SceneObject]


@dataclass
class _Paths:
    objects: _Tensor
    interactions: _Tensor


class _BadString:
    def __str__(self) -> str:
        raise TypeError("cannot stringify")


def _sample_paths() -> _Paths:
    invalid = SIONNA_INVALID_OBJECT_ID
    objects = np.full((3, 1, 1, 4), invalid, dtype=np.uint32)
    interactions = np.zeros((3, 1, 1, 4), dtype=np.int32)

    objects[0, 0, 0, 0] = 1
    interactions[0, 0, 0, 0] = SIONNA_INTERACTION_SPECULAR

    objects[0, 0, 0, 1] = 2
    interactions[0, 0, 0, 1] = SIONNA_INTERACTION_DIFFUSE
    objects[1, 0, 0, 1] = 3
    interactions[1, 0, 0, 1] = SIONNA_INTERACTION_REFRACTION

    objects[0, 0, 0, 2] = 99
    interactions[0, 0, 0, 2] = SIONNA_INTERACTION_DIFFRACTION

    return _Paths(objects=_Tensor(objects), interactions=_Tensor(interactions))


def _sample_scene() -> _Scene:
    return _Scene(
        objects={
            "target": _SceneObject(
                object_id=np.uint32(1),
                radio_material=_Material(name="mat-itu_glass_drone", itu_type="glass"),
            ),
            "wall": _SceneObject(
                object_id=np.uint32(2),
                radio_material=_Material(name="itu_concrete"),
            ),
            "unassigned": _SceneObject(object_id=np.uint32(3), radio_material=None),
            "invalid": _SceneObject(
                object_id=SIONNA_INVALID_OBJECT_ID,
                radio_material=_Material(name="mat-itu_metal", itu_type="metal"),
            ),
        }
    )


def test_materials_per_bounce_compacts_physical_order_and_normalizes_names() -> None:
    mapping = materials_per_bounce(_sample_scene(), _sample_paths())

    assert set(mapping) == {(0, 0)}
    assert set(mapping[(0, 0)]) == {0, 1, 2, 3}
    assert mapping[(0, 0)][0] == [
        {"name": "mat-itu_glass", "itu_type": "glass"},
    ]
    assert mapping[(0, 0)][1] == [
        {"name": "mat-itu_concrete", "itu_type": None},
        {"name": "no-material", "itu_type": None},
    ]
    assert mapping[(0, 0)][2] == [
        {"name": "unknown", "itu_type": None},
    ]
    assert mapping[(0, 0)][3] == []


def test_material_extraction_tolerates_bad_material_strings() -> None:
    objects = np.array([[[[1]]]], dtype=np.uint32)
    interactions = np.array([[[[SIONNA_INTERACTION_SPECULAR]]]], dtype=np.int32)
    scene = _Scene(
        objects={
            "bad": _SceneObject(
                object_id=np.uint32(1),
                radio_material=_Material(name=_BadString(), itu_type=_BadString()),
            )
        }
    )
    paths = _Paths(objects=_Tensor(objects), interactions=_Tensor(interactions))

    assert materials_per_bounce(scene, paths) == {
        (0, 0): {0: [{"name": "unknown", "itu_type": None}]}
    }


def test_materials_per_bounce_handles_zero_depth_paths() -> None:
    paths = _Paths(
        objects=_Tensor(np.empty((0, 1, 1, 2), dtype=np.uint32)),
        interactions=_Tensor(np.empty((0, 1, 1, 2), dtype=np.int32)),
    )

    assert materials_per_bounce(_Scene(objects={}), paths) == {(0, 0): {0: [], 1: []}}


def test_materials_per_bounce_keeps_unknown_slot_for_invalid_object_id() -> None:
    objects = np.full((2, 1, 1, 1), SIONNA_INVALID_OBJECT_ID, dtype=np.uint32)
    interactions = np.zeros((2, 1, 1, 1), dtype=np.int32)
    interactions[1, 0, 0, 0] = SIONNA_INTERACTION_DIFFRACTION
    paths = _Paths(objects=_Tensor(objects), interactions=_Tensor(interactions))

    assert materials_per_bounce(_Scene(objects={}), paths) == {
        (0, 0): {0: [{"name": "unknown", "itu_type": None}]}
    }


def test_materials_per_bounce_compacts_sparse_depths_without_reordering() -> None:
    invalid = SIONNA_INVALID_OBJECT_ID
    objects = np.full((3, 1, 1, 1), invalid, dtype=np.uint32)
    interactions = np.zeros((3, 1, 1, 1), dtype=np.int32)
    objects[0, 0, 0, 0] = 1
    interactions[0, 0, 0, 0] = SIONNA_INTERACTION_SPECULAR
    objects[2, 0, 0, 0] = 2
    interactions[2, 0, 0, 0] = SIONNA_INTERACTION_DIFFUSE
    scene = _Scene(
        objects={
            "first": _SceneObject(1, _Material("first")),
            "last": _SceneObject(2, _Material("last")),
        }
    )
    paths = _Paths(objects=_Tensor(objects), interactions=_Tensor(interactions))

    assert materials_per_bounce(scene, paths) == {
        (0, 0): {
            0: [
                {"name": "first", "itu_type": None},
                {"name": "last", "itu_type": None},
            ]
        }
    }
