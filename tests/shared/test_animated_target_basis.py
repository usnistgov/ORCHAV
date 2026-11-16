"""Body-local basis changes preserve full motion and support root replacement."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from shared.animated_target import (
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    load_animated_target,
)
from shared.animated_target_export import export_animated_target

FACES = np.array(((0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)))
BODY = np.array(((0, 0, 0), (0.4, 0, 0), (0, 0.2, 0), (0, 0, 1.8)))
BASES = (
    ("+X", "+Z", np.eye(3)),
    ("+Z", "+Y", np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))),
    ("-Y", "+Z", np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))),
)


def _export(path, *, source_axes, local_axes=None, scale=1.0, rotations=None):
    local = BASES[1] if local_axes is None else local_axes
    # Independently authored body coordinates: a 1.8m up edge, 0.4m front
    # edge and 0.2m left edge, expressed in the requested local basis.
    vertices = np.stack((BODY @ local[2], (BODY * (1, 1, 1.1)) @ local[2]))
    rotations = (
        rotations
        if rotations is not None
        else Rotation.from_euler("xyz", ((90, 20, -15), (70, -25, 65)), degrees=True).as_quat()
    )
    keywords = (
        {}
        if local_axes is None
        else {"root_local_front_axis": local_axes[0], "root_local_up_axis": local_axes[1]}
    )
    return export_animated_target(
        path,
        asset_id="body:explicit-basis",
        vertices=vertices,
        faces=FACES,
        times_s=[0, 1],
        root_translation=[[2, -3, 4], [5, -1, 7]],
        root_rotation_xyzw=rotations,
        source_type="numeric_fixture",
        source_records={
            "fixture": {
                "sha256": hashlib.sha256(BODY.tobytes()).hexdigest(),
                "size_bytes": BODY.nbytes,
            }
        },
        adapter_metadata={"purpose": "independent body-basis check"},
        front_axis=source_axes[0],
        up_axis=source_axes[1],
        meters_per_unit=scale,
        closed_surface=True,
        **keywords,
    )


def _binding(path):
    asset = load_animated_target(path)
    context = AnimatedTargetBindingContext(
        "target:body",
        "clock:scene",
        "clock:motion",
        None,
        1,
        0,
        "affine_scenario_to_motion",
    )
    labels = ("start", "early", "midpoint", "late", "end")
    times = (0, 0.25, 0.5, 0.75, 1)
    samples = tuple(
        AnimatedTargetBindingTime("query:" + name, t, t) for name, t in zip(labels, times)
    )
    return asset, asset.bind(context, samples)


@pytest.mark.parametrize("source_axes", BASES)
@pytest.mark.parametrize("local_axes", BASES)
@pytest.mark.parametrize("scale", (1.0, 0.01))
def test_explicit_body_basis_preserves_world_binding_and_canonical_local_body(
    tmp_path, source_axes, local_axes, scale
):
    output = _export(
        tmp_path / "normalized", source_axes=source_axes, local_axes=local_axes, scale=scale
    )
    asset, binding = _binding(output)
    # Compare to the same authored input without re-expression. The baseline
    # fixture is independent of the implementation's B and compensation math.
    rotations = Rotation.from_euler("xyz", ((90, 20, -15), (70, -25, 65)), degrees=True)
    from scipy.spatial.transform import Slerp

    interpolated = Slerp([0, 1], rotations)([0, 0.25, 0.5, 0.75, 1])
    for index, sample in enumerate(binding.samples):
        fraction = index / 4
        local = (BODY * ((1, 1, 1.1) if index == 4 else (1, 1, 1))) @ local_axes[2]
        translation = np.array([2, -3, 4]) + fraction * np.array([3, 2, 3])
        expected = (interpolated[index].apply(local) + translation) @ source_axes[2].T * scale
        np.testing.assert_allclose(sample.vertices_m, expected, atol=1e-13, rtol=1e-13)
        np.testing.assert_allclose(np.linalg.norm(sample.normals, axis=1), 1)
    np.testing.assert_allclose(
        asset.pose_samples[0].vertices @ asset.source_to_orchav_affine_4x4[:3, :3].T,
        BODY * scale,
        atol=1e-14,
    )
    np.testing.assert_array_equal(asset.canonical_faces, FACES)
    assert [row.time_s for row in asset.pose_samples] == [0, 1]
    np.testing.assert_array_equal(
        [row.translation for row in asset.root_samples], [[2, -3, 4], [5, -1, 7]]
    )
    sidecar = output.parent / "adapter_provenance.json"
    operation = json.loads(sidecar.read_text())["operations"]["root_local_basis"]
    assert operation["input_front_axis"] == local_axes[0]
    assert operation["output_up_axis"] == source_axes[1]
    np.testing.assert_allclose(np.linalg.det(operation["input_to_output_rotation_3x3"]), 1)
    assert (
        "adapter_provenance:sha256:" + hashlib.sha256(sidecar.read_bytes()).hexdigest()
        in asset.provenance["source_ids"]
    )


def test_omitted_local_axes_keep_existing_pose_and_root_data(tmp_path):
    path = _export(tmp_path / "legacy", source_axes=BASES[0])
    asset, _ = _binding(path)
    np.testing.assert_array_equal(asset.pose_samples[0].vertices, BODY @ BASES[1][2])
    np.testing.assert_allclose(
        asset.root_samples[0].rotation_xyzw,
        Rotation.from_euler("xyz", (90, 20, -15), degrees=True).as_quat(),
        atol=1e-14,
    )
    operation = json.loads((path.parent / "adapter_provenance.json").read_text())["operations"][
        "root_local_basis"
    ]
    assert operation == {"policy": "preserve_unspecified_input_basis"}


@pytest.mark.parametrize(
    "keywords",
    (
        {"root_local_front_axis": "+Z"},
        {"root_local_up_axis": "+Y"},
        {"root_local_front_axis": "+Z", "root_local_up_axis": "-Z"},
        {"root_local_front_axis": "automatic", "root_local_up_axis": "+Y"},
    ),
)
def test_bad_or_partial_local_basis_fails_before_publication(tmp_path, keywords):
    with pytest.raises(ValueError):
        export_animated_target(
            tmp_path / "bad",
            asset_id="body:invalid",
            vertices=[BODY],
            faces=FACES,
            times_s=[0],
            root_translation=[[0, 0, 0]],
            root_rotation_xyzw=[[0, 0, 0, 1]],
            source_type="numeric_fixture",
            source_records={
                "fixture": {
                    "sha256": hashlib.sha256(BODY.tobytes()).hexdigest(),
                    "size_bytes": BODY.nbytes,
                }
            },
            adapter_metadata={},
            **keywords,
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad*"))


def test_ambiguous_half_turn_reexpression_refuses_to_choose_a_new_motion(tmp_path):
    rotations = np.array(((0, 0, 0, 1), (1, 0, 0, 0)))
    with pytest.raises(ValueError, match="ambiguous 180-degree"):
        _export(
            tmp_path / "ambiguous", source_axes=BASES[0], local_axes=BASES[1], rotations=rotations
        )
    assert not (tmp_path / "ambiguous").exists()
    assert not list(tmp_path.glob(".ambiguous*"))
    # No-op and legacy imports retain their previous interpolation contract.
    for name, source, local in (("identity", BASES[1], BASES[1]), ("legacy", BASES[0], None)):
        path = _export(tmp_path / name, source_axes=source, local_axes=local, rotations=rotations)
        _binding(path)


@pytest.mark.parametrize("angle", (179.9, 180.1))
def test_near_half_turn_keeps_the_original_shortest_path(tmp_path, angle):
    from scipy.spatial.transform import Slerp

    rotations = Rotation.from_euler("x", [[0], [angle]], degrees=True)
    path = _export(
        tmp_path / "near-half",
        source_axes=BASES[0],
        local_axes=BASES[1],
        rotations=rotations.as_quat(),
    )
    _, binding = _binding(path)
    interpolated = Slerp([0, 1], rotations)([0, 0.25, 0.5, 0.75, 1])
    for index, row in enumerate(binding.samples):
        local = (BODY * ((1, 1, 1.1) if index == 4 else (1, 1, 1))) @ BASES[1][2]
        expected = (
            interpolated[index].apply(local)
            + np.array([2, -3, 4])
            + index / 4 * np.array([3, 2, 3])
        )
        np.testing.assert_allclose(row.vertices_m, expected, atol=1e-12)
