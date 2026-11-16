"""Native target identities survive filtering, freezing and CPU HDF5 replay."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from generator.core.materials.path_metadata import materials_per_bounce
from generator.core.propagation.frozen_paths import FrozenPaths
from generator.io.frames.sensing_identity import pack_sensing_identity
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.frame_set_writer import FrameSetWriter
from shared.frames.normalization import standard_mpc_frame_from_pair_data
from shared.frames.packed_hdf5 import PackedHDF5Reader
from shared.frames.sensing_identity import sensing_bounce_identity, validate_sensing_identity


class Array:
    def __init__(self, values):
        self.values = np.asarray(values)

    def numpy(self):
        return self.values


def fixture():
    # Path 0 is removed, path 1 has a mesh bounce then sensing, path 2 sensing.
    interactions = np.array([[[[16, 1, 16]]], [[[0, 16, 0]]]], dtype=np.uint32)
    paths = SimpleNamespace(
        interactions=Array(interactions),
        objects=Array(np.array([[[[0, 0, 1]]], [[[0, 0, 0]]]], dtype=np.uint32)),
        primitives=Array(np.array([[[[0, 8, 4]]], [[[0, 2, 0]]]], dtype=np.uint32)),
    )
    metadata = (
        {"name": "mesh", "representation": "mesh_material"},
        {"name": "vehicle", "representation": "tr38901", "scattering_point_ids": list(range(5))},
        {"name": "agv", "representation": "tr38901", "scattering_point_ids": list(range(5))},
    )
    provenance = {
        "schema_version": 1,
        "target_catalog_frame_indices": [1, 2],
        "target_catalog_names": ["vehicle", "agv"],
        "channel": "combined",
    }
    return paths, metadata, provenance


def test_filtered_native_ids_are_not_mesh_object_ids():
    paths, metadata, provenance = fixture()
    packed = np.array([1, 16, 16], dtype=np.uint8)
    result = pack_sensing_identity(
        paths, paths.interactions.numpy(), [np.array([1, 2])], packed, metadata, provenance
    )
    assert result["bounce_indices"] == [1, 2]
    assert result["target_indices"] == [1, 2]
    assert result["scattering_point_indices"] == [2, 4]
    assert "bounce_indices" not in provenance


def test_frozen_paths_retain_primitive_indices():
    paths, metadata, provenance = fixture()
    frozen = FrozenPaths(paths)
    result = pack_sensing_identity(
        frozen,
        frozen.interactions.numpy(),
        [np.array([1, 2])],
        np.array([1, 16, 16]),
        metadata,
        provenance,
    )
    assert result["scattering_point_indices"] == [2, 4]


def test_sensing_event_never_looks_up_colliding_mesh_material():
    paths, _, _ = fixture()
    scene = SimpleNamespace(
        objects={
            "wall": SimpleNamespace(
                object_id=0, radio_material=SimpleNamespace(name="wall-material", itu_type=None)
            )
        }
    )
    mapping = materials_per_bounce(scene, paths)
    assert mapping[(0, 0)][1][0]["name"] == "wall-material"
    assert mapping[(0, 0)][1][1]["name"] == "no-material"
    assert mapping[(0, 0)][0][0]["name"] == "no-material"


@pytest.mark.parametrize("missing", ["objects", "primitives"])
def test_missing_identity_is_not_silently_discarded(missing):
    paths, metadata, provenance = fixture()
    delattr(paths, missing)
    with pytest.raises(ValueError, match="require retained"):
        pack_sensing_identity(
            paths,
            paths.interactions.numpy(),
            [np.array([1, 2])],
            np.array([1, 16, 16]),
            metadata,
            provenance,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("bounce_indices", [2, 1]),
        ("target_indices", [9, 2]),
        ("scattering_point_indices", [9, 4]),
        ("bounce_indices", [True, 2]),
    ],
)
def test_invalid_retained_identity_rejected(field, value):
    paths, metadata, provenance = fixture()
    payload = pack_sensing_identity(
        paths,
        paths.interactions.numpy(),
        [np.array([1, 2])],
        np.array([1, 16, 16]),
        metadata,
        provenance,
    )
    payload[field] = value
    with pytest.raises(ValueError):
        validate_sensing_identity(payload, np.array([1, 16, 16]), metadata)


def test_empty_return_retains_model_and_channel_provenance():
    paths, metadata, provenance = fixture()
    payload = pack_sensing_identity(
        paths,
        paths.interactions.numpy(),
        [np.array([], dtype=int)],
        np.array([], dtype=np.uint8),
        metadata,
        provenance,
    )
    assert payload["target_indices"] == []
    assert payload["channel"] == "combined"


def test_native_identity_hdf5_roundtrip(tmp_path):
    from shared.frames.protobuf import standard_mpc_frame_from_proto, standard_mpc_frame_to_proto

    paths, metadata, provenance = fixture()
    payload = pack_sensing_identity(
        paths,
        paths.interactions.numpy(),
        [np.array([1, 2])],
        np.array([1, 16, 16]),
        metadata,
        provenance,
    )
    frame = standard_mpc_frame_from_pair_data(
        frame_index=0,
        tx_rx_pairs=[(0, 0)],
        tx_positions=[[0, 0, 0]],
        rx_positions=[[0, 0, 0]],
        vertices_by_pair=[[np.array([[1, 0, 0], [2, 0, 0]]), np.array([[3, 0, 0]])]],
        interactions_by_pair=[[np.array([1, 16]), np.array([16])]],
        target_positions_m=np.zeros((3, 3)),
        targets_metadata=metadata,
        provenance={"sionna_sensing": payload},
    )
    directory = tmp_path / "frames"
    writer = FrameSetWriter.create_new(directory)
    writer.append(frame)
    writer.finalize()
    reader = PackedHDF5Reader(directory)
    try:
        restored = reader.load_standard_frame(0)
        assert restored.provenance["sionna_sensing"] == payload
        assert sensing_bounce_identity(restored, 0) is None
        assert sensing_bounce_identity(restored, 1) == (1, 2)
        assert sensing_bounce_identity(restored, 2) == (2, 4)
        transported = standard_mpc_frame_from_proto(standard_mpc_frame_to_proto(restored))
        assert transported.provenance["sionna_sensing"] == payload
        assert sensing_bounce_identity(transported, 2) == (2, 4)
        with pytest.raises(ValueError, match="require retained"):
            replace(restored, provenance={})
        projection = reader.load_projection(
            0,
            FrameReadRequest(
                components=frozenset(
                    {
                        FrameComponent.PATH_INTERACTIONS,
                        FrameComponent.PROVENANCE,
                        FrameComponent.TARGETS,
                    }
                )
            ),
        )
        with pytest.raises(ValueError, match="require retained"):
            replace(projection, frame=replace(projection.frame, provenance=None))
        bad = deepcopy(payload)
        bad["target_indices"][0] = 0
        with pytest.raises(ValueError, match="non-native"):
            validate_sensing_identity(bad, restored.interactions, restored.targets_metadata)
    finally:
        reader.close()
