"""Public UAV mesh dimensions, topology, hashes and rotor geometry."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "name,dimensions", [("compact", [0.3, 0.4, 0.2]), ("large", [1.6, 1.5, 0.7])]
)
def test_dimensioned_meshes_have_valid_closed_components_and_rotor_pivots(name, dimensions):
    directory = ROOT / "libraries/targets/uav" / name
    obj = (directory / "drone.obj").read_text()
    metadata = json.loads((directory / "geometry.json").read_text())
    v = np.array(
        [[float(n) for n in row.split()[1:]] for row in obj.splitlines() if row.startswith("v ")]
    )
    f = np.array(
        [[int(n) - 1 for n in row.split()[1:]] for row in obj.splitlines() if row.startswith("f ")]
    )
    assert np.allclose(np.ptp(v, axis=0), dimensions, atol=1e-8)
    assert np.allclose(v.min(axis=0), -np.array(dimensions) / 2, atol=1e-8)
    assert metadata["sha256"] == hashlib.sha256((directory / "drone.obj").read_bytes()).hexdigest()
    assert len(f) == metadata["triangles"] and f.min() == 0 and f.max() < len(v)
    assert np.all(
        np.linalg.norm(np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]]), axis=1) > 1e-12
    )
    for part in metadata["parts"]:
        from collections import Counter

        edges = Counter(
            tuple(sorted((face[i], face[(i + 1) % 3])))
            for face in f[part["first_face"] : part["first_face"] + part["face_count"]]
            for i in range(3)
        )
        assert set(edges.values()) == {2}
    assert len(metadata["rotors"]) == 4
    for rotor in metadata["rotors"]:
        assert rotor["axis"] == [0, 0, 1] and rotor["radius_m"] > 0
