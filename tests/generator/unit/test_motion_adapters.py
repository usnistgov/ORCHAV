"""Independent geometry, timing and isolation tests for offline motion adapters."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from generator.tools.kimodo_to_animated_target import (
    SOMA30_NAMES,
    SOMA30_PARENTS,
    export_kimodo_motion,
    matrices_to_quaternions,
)
from generator.tools.smpl_to_animated_target import export_amass_motion, export_smpl_surfaces
from shared.animated_target import (
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    load_animated_target,
)
from shared.animated_target_export import load_numeric_npz, sample_times, source_affine

VERTICES = np.array(((0, 0, 0), (0.8, 0.1, 0), (0, 1.2, 0.2), (-0.2, 0.3, 1.1)))
FACES = np.array(((1, 2, 3), (0, 3, 2), (0, 1, 3), (0, 2, 1)))
# This independently specified cyclic permutation preserves RH chirality and
# maps source +Z forward to world +X, source +Y up to world +Z.
BASIS = np.array(((0, 0, 1), (1, 0, 0), (0, 1, 0)))
IDENTITY = np.array((0, 0, 0, 1.0))
QUARTER_TURN_Y = np.array(((0, 0, 1), (0, 1, 0), (-1, 0, 0)))


def _bind(path, times):
    asset = load_animated_target(path)
    context = AnimatedTargetBindingContext(
        "target:walker", "clock:scenario", "clock:motion", None, 1, 0, "affine_scenario_to_motion"
    )
    requests = tuple(
        AnimatedTargetBindingTime(f"query:time:{float(time).hex()}", time, time) for time in times
    )
    return asset, asset.bind(context, requests)


def _surface(tmp_path, *, space="world", times=(0, 0.07, 0.19)):
    rotations = np.stack((np.eye(3), QUARTER_TURN_Y, QUARTER_TURN_Y @ QUARTER_TURN_Y))
    quaternions = np.array((IDENTITY, (0, 2**-0.5, 0, 2**-0.5), (0, 1, 0, 0)))
    roots = np.array(((0.2, 0.4, 0.7), (0.3, 0.4, 1.7), (0.8, 0.4, 3.2)))
    local = np.stack((VERTICES, VERTICES * (1, 1.1, 1), VERTICES * (1.1, 1, 1)))
    world = np.einsum("nij,nvj->nvi", rotations, local) + roots[:, None]
    path = tmp_path / f"surface_{space}.npz"
    np.savez(
        path,
        vertices=world if space == "world" else local,
        faces=FACES,
        root_translation=roots,
        root_rotation_xyzw=quaternions,
        time_s=times,
    )
    return path, world


@pytest.mark.parametrize("space", ("world", "root_local"))
def test_surface_export_preserves_world_geometry_root_chirality_and_exact_times(tmp_path, space):
    source, expected_world = _surface(tmp_path, space=space)
    output = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:walker",
        geometry_space=space,
        front_axis="+Z",
        up_axis="+Y",
        closed_surface=True,
    )
    asset, binding = _bind(output, [0, 0.07, 0.19])
    np.testing.assert_array_equal(asset.canonical_faces, FACES)
    assert np.linalg.det(asset.source_to_orchav_basis) == 1
    for row, expected in zip(binding.samples, expected_world):
        np.testing.assert_allclose(row.vertices_m, expected @ BASIS.T, atol=1e-14)
        face_vertices = row.vertices_m[FACES]
        signed_volume = (
            np.einsum(
                "ij,ij->i", face_vertices[:, 0], np.cross(face_vertices[:, 1], face_vertices[:, 2])
            ).sum()
            / 6
        )
        assert signed_volume > 0
        np.testing.assert_allclose(np.linalg.norm(row.normals, axis=1), 1)
    assert [row.time_s for row in asset.pose_samples] == [0, 0.07, 0.19]
    assert asset.license["redistribution"] == "user_supplied_only"
    sidecar = output.parent / "adapter_provenance.json"
    assert (
        f"adapter_provenance:sha256:{hashlib.sha256(sidecar.read_bytes()).hexdigest()}"
        in asset.provenance["source_ids"]
    )


def test_stride_selects_native_times_and_step_pose_does_not_interpolate(tmp_path):
    source, expected_world = _surface(tmp_path)
    path = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:stride",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
        stride=2,
    )
    asset, binding = _bind(path, [0, 0.1, 0.19])
    assert [row.time_s for row in asset.pose_samples] == [0, 0.19]
    assert binding.samples[1].pose_sample_id == binding.samples[0].pose_sample_id
    np.testing.assert_allclose(binding.samples[-1].vertices_m, expected_world[-1] @ BASIS.T)
    with pytest.raises(ValueError, match="outside"):
        _bind(path, [0.2])


def test_publication_is_deterministic_and_never_overwrites(tmp_path):
    source, _ = _surface(tmp_path)
    first = export_smpl_surfaces(
        source,
        tmp_path / "first",
        asset_id="smpl:fixture:same",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
    )
    second = export_smpl_surfaces(
        source,
        tmp_path / "second",
        asset_id="smpl:fixture:same",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
    )
    assert first.read_bytes() == second.read_bytes()
    inventory = {
        path.relative_to(first.parent).as_posix(): path.read_bytes()
        for path in first.parent.rglob("*")
        if path.is_file()
    }
    with pytest.raises((ValueError, RuntimeError, FileExistsError)):
        export_smpl_surfaces(
            source,
            first.parent,
            asset_id="smpl:fixture:same",
            geometry_space="world",
            front_axis="+Z",
            up_axis="+Y",
        )
    assert inventory == {
        path.relative_to(first.parent).as_posix(): path.read_bytes()
        for path in first.parent.rglob("*")
        if path.is_file()
    }


def test_bad_topology_is_rejected_without_partial_destination(tmp_path):
    source, _ = _surface(tmp_path)
    arrays, _ = load_numeric_npz(source)
    arrays["faces"] = np.array(((0, 0, 1),))
    np.savez(source, **arrays)
    with pytest.raises(ValueError):
        export_smpl_surfaces(
            source,
            tmp_path / "bad",
            asset_id="smpl:fixture:bad",
            geometry_space="world",
            front_axis="+Z",
            up_axis="+Y",
        )
    assert not (tmp_path / "bad").exists()
    assert not list(tmp_path.glob(".bad.staging-*"))


def test_object_arrays_are_never_deserialized(tmp_path):
    path = tmp_path / "object.npz"
    np.savez(path, poses=np.array([{"unsafe": True}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        load_numeric_npz(path)


@pytest.mark.parametrize(
    "kwargs",
    (
        {},
        {"source_fps": 0},
        {"source_fps": float("nan")},
        {"times_s": [0, 0]},
        {"times_s": [0, -1]},
        {"times_s": [0, 0.3], "source_fps": 10},
    ),
)
def test_invalid_or_unknown_time_is_not_guessed(kwargs):
    with pytest.raises(ValueError):
        sample_times(2, **kwargs)


def _kimodo_inputs(tmp_path, *, soma30=False):
    names = list(SOMA30_NAMES) + [f"ExtraJoint{index}" for index in range(47)]
    parents = np.concatenate((SOMA30_PARENTS, np.zeros(47, dtype=int)))
    connections = np.array([(parent, joint) for joint, parent in enumerate(parents) if parent >= 0])
    bind = np.broadcast_to(np.eye(4), (77, 4, 4)).copy()
    skin = dict(
        bind_vertices=VERTICES,
        faces=FACES,
        bind_rig_transform=bind,
        rig_joint_names=np.array(names),
        rig_joint_connections=connections,
        lbs_indices=np.zeros((4, 1), dtype=int),
        lbs_weights=np.ones((4, 1)),
    )
    skin_path = tmp_path / "skin.npz"
    np.savez(skin_path, **skin)
    count = 30 if soma30 else 77
    roots = np.array(((0.2, 0.4, 0.7), (0.3, 0.4, 1.7)))
    rotations = np.broadcast_to(
        np.stack((np.eye(3), QUARTER_TURN_Y))[:, None], (2, count, 3, 3)
    ).copy()
    positions = np.broadcast_to(roots[:, None], (2, count, 3)).copy()
    motion_path = tmp_path / "motion.npz"
    np.savez(motion_path, posed_joints=positions, global_rot_mats=rotations, time_s=[0, 0.12])
    expected = np.stack((VERTICES + roots[0], VERTICES @ QUARTER_TURN_Y.T + roots[1]))
    return motion_path, skin_path, skin, expected


@pytest.mark.parametrize("soma30", (False, True))
def test_kimodo_skinning_and_root_decomposition_match_rigid_reference(tmp_path, soma30):
    motion, skin, _, expected = _kimodo_inputs(tmp_path, soma30=soma30)
    extras = {}
    if soma30:
        np.save(tmp_path / "neutral.npy", np.zeros((77, 3)), allow_pickle=False)
        np.save(
            tmp_path / "relaxed.npy", np.broadcast_to(np.eye(3), (77, 3, 3)), allow_pickle=False
        )
        extras = {
            "neutral_joints": tmp_path / "neutral.npy",
            "relaxed_pose": tmp_path / "relaxed.npy",
        }
    path = export_kimodo_motion(
        motion,
        skin,
        tmp_path / "asset",
        asset_id="kimodo:fixture:turn",
        closed_surface=True,
        **extras,
    )
    asset, binding = _bind(path, [0, 0.12])
    assert np.linalg.det(asset.source_to_orchav_basis) == 1
    for row, reference in zip(binding.samples, expected):
        np.testing.assert_allclose(row.vertices_m, reference @ BASIS.T, atol=1e-14)
    np.testing.assert_array_equal(asset.canonical_faces, FACES)
    assert json.loads((path.parent / "adapter_provenance.json").read_text())["operations"][
        "source_joint_count"
    ] == (30 if soma30 else 77)


@pytest.mark.parametrize(
    "bad", ("negative_weights", "nonunit_weights", "fractional_faces", "reflection", "cycle")
)
def test_kimodo_rejects_invalid_skin_and_motion(tmp_path, bad):
    motion, skin_path, skin, _ = _kimodo_inputs(tmp_path)
    if bad == "negative_weights":
        skin["lbs_weights"][0, 0] = -1
    elif bad == "nonunit_weights":
        skin["lbs_weights"][0, 0] = 0.8
    elif bad == "fractional_faces":
        skin["faces"] = FACES.astype(float) + 0.2
    elif bad == "cycle":
        edges = skin["rig_joint_connections"].copy()
        edges[0, 0] = 2
        skin["rig_joint_connections"] = edges
    else:
        arrays, _ = load_numeric_npz(motion)
        arrays["global_rot_mats"][0, 0, :, 0] *= -1
        np.savez(motion, **arrays)
    np.savez(skin_path, **skin)
    with pytest.raises(ValueError):
        export_kimodo_motion(motion, skin_path, tmp_path / "bad", asset_id="kimodo:fixture:bad")
    assert not (tmp_path / "bad").exists()


def test_kimodo_soma30_requires_numeric_skeleton_inputs(tmp_path):
    motion, skin, _, _ = _kimodo_inputs(tmp_path, soma30=True)
    with pytest.raises(ValueError, match="neutral-joints"):
        export_kimodo_motion(motion, skin, tmp_path / "asset", asset_id="kimodo:fixture:missing")


def test_quaternion_conversion_handles_half_turns():
    matrices = np.array(
        (np.eye(3), np.diag((1, -1, -1)), np.diag((-1, 1, -1)), np.diag((-1, -1, 1)))
    )
    expected = np.array((IDENTITY, (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0)))
    np.testing.assert_allclose(matrices_to_quaternions(matrices), expected)


def test_amass_requires_explicit_model_trust_before_loading(tmp_path):
    with pytest.raises(ValueError, match="trusted-model"):
        export_amass_motion(
            tmp_path / "missing.npz",
            tmp_path / "missing.pkl",
            tmp_path / "asset",
            asset_id="smpl:fixture:trust",
            front_axis="+Z",
            up_axis="+Y",
        )


@pytest.mark.parametrize(
    "front_axis,up_axis,world_basis",
    (
        ("+Z", "+Y", BASIS),
        ("+X", "+Z", np.eye(3)),
        ("-Y", "+Z", np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))),
    ),
)
def test_amass_model_boundary_preserves_evaluated_pelvis_and_world_vertices(
    tmp_path, monkeypatch, front_axis, up_axis, world_basis
):
    # This controlled stand-in validates the ORCHAV model adapter, independently
    # of model assets or the optional smplx package. It is not a SMPL golden.
    class Tensor:
        def __init__(self, value):
            self.value = np.asarray(value)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    class Guard:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    expected = []

    class Model:
        faces = FACES

        def __init__(self, **kwargs):
            pass

        def requires_grad_(self, value):
            pass

        def to(self, value):
            return self

        def __call__(self, **kwargs):
            from scipy.spatial.transform import Rotation

            matrix = Rotation.from_rotvec(kwargs["global_orient"].value[0]).as_matrix()
            pelvis = kwargs["transl"].value[0] + np.array((0.1, 0.8, -0.2))
            world = VERTICES @ matrix.T + pelvis
            expected.append(world)
            return SimpleNamespace(vertices=Tensor(world[None]), joints=Tensor(pelvis[None, None]))

    monkeypatch.setitem(sys.modules, "smplx", SimpleNamespace(SMPL=Model))
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            tensor=lambda value, dtype: Tensor(value), float32="float32", no_grad=Guard
        ),
    )
    poses = np.zeros((2, 156))
    poses[0, :3] = (np.pi / 2, 0, 0)
    poses[1, :3] = (1.1, 0.7, -0.3)
    np.savez(
        tmp_path / "motion.npz",
        poses=poses,
        trans=((0, 0, 0), (0, 0, 2)),
        betas=np.zeros(16),
        mocap_framerate=30,
        gender="neutral",
    )
    (tmp_path / "trusted.pkl").write_bytes(b"synthetic model stand-in; never deserialized")
    path = export_amass_motion(
        tmp_path / "motion.npz",
        tmp_path / "trusted.pkl",
        tmp_path / "asset",
        asset_id="smpl:fixture:model",
        trusted_model=True,
        front_axis=front_axis,
        up_axis=up_axis,
    )
    asset, binding = _bind(path, [0, 1 / 30])
    for row, reference in zip(binding.samples, expected):
        np.testing.assert_allclose(row.vertices_m, reference @ world_basis.T, atol=1e-14)
    for pose in asset.pose_samples:
        np.testing.assert_allclose(pose.vertices @ world_basis.T, VERTICES @ BASIS.T, atol=1e-14)
    operation = json.loads((path.parent / "adapter_provenance.json").read_text())["operations"][
        "root_local_basis"
    ]
    assert operation["input_front_axis"] == "+Z"
    assert operation["input_up_axis"] == "+Y"


def test_adapter_modules_and_cli_help_do_not_import_source_frameworks():
    script = """
import sys
from generator.tools.smpl_to_animated_target import build_parser as smpl
from generator.tools.kimodo_to_animated_target import build_parser as kimodo
assert smpl().format_help()
assert kimodo().format_help()
assert not {'torch', 'smplx', 'trimesh', 'sionna', 'mitsuba', 'kimodo'} & set(sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_basis_has_correct_front_up_and_handedness():
    np.testing.assert_array_equal(source_affine("+Z", "+Y")[:3, :3], BASIS)
    assert np.linalg.det(source_affine("-Z", "+Y")[:3, :3]) == 1


def test_kimodo_articulated_weighted_skinning_matches_independent_blend(tmp_path):
    motion_path, skin_path, skin, _ = _kimodo_inputs(tmp_path)
    motion, _ = load_numeric_npz(motion_path)
    roots = motion["posed_joints"][:, 0].copy()
    root_rotations = motion["global_rot_mats"][:, 0].copy()
    bind_joint = np.array((0.05, 0.02, 0.0))
    skin["bind_rig_transform"][1, :3, 3] = bind_joint
    skin["lbs_indices"] = np.tile((0, 1), (4, 1))
    skin["lbs_weights"] = np.tile((0.4, 0.6), (4, 1))
    twist = np.array(((0, -1, 0), (1, 0, 0), (0, 0, 1)))
    expected = []
    for index in range(2):
        root_rotation = root_rotations[index]
        joint_rotation = root_rotation @ (np.eye(3) if index == 0 else twist)
        joint = roots[index] + root_rotation @ np.array((0.25, 0.05, 0.1))
        motion["global_rot_mats"][index, 1] = joint_rotation
        motion["posed_joints"][index, 1] = joint
        root_contribution = VERTICES @ root_rotation.T + roots[index]
        joint_contribution = (VERTICES - bind_joint) @ joint_rotation.T + joint
        expected.append(0.4 * root_contribution + 0.6 * joint_contribution)
    np.savez(motion_path, **motion)
    np.savez(skin_path, **skin)
    path = export_kimodo_motion(
        motion_path, skin_path, tmp_path / "asset", asset_id="kimodo:fixture:weighted"
    )
    _, binding = _bind(path, [0, 0.12])
    for row, reference in zip(binding.samples, expected):
        np.testing.assert_allclose(row.vertices_m, reference @ BASIS.T, atol=1e-14)
    velocity = (binding.samples[1].vertices_m - binding.samples[0].vertices_m) / 0.12
    np.testing.assert_allclose(velocity, (expected[1] - expected[0]) @ BASIS.T / 0.12, atol=1e-13)


def test_surface_timing_conflicts_reject_without_publication(tmp_path):
    source, _ = _surface(tmp_path)
    with pytest.raises(ValueError, match="disagree"):
        export_smpl_surfaces(
            source,
            tmp_path / "asset",
            asset_id="smpl:fixture:rate",
            geometry_space="world",
            front_axis="+Z",
            up_axis="+Y",
            source_fps=30,
        )
    assert not (tmp_path / "asset").exists()


@pytest.mark.parametrize("missing", ("trans", "betas", "gender"))
def test_amass_does_not_invent_missing_source_kinematics(tmp_path, missing):
    values = dict(
        poses=np.zeros((2, 72)),
        trans=np.zeros((2, 3)),
        betas=np.zeros(10),
        gender="neutral",
        mocap_framerate=30,
    )
    del values[missing]
    np.savez(tmp_path / "motion.npz", **values)
    with pytest.raises(KeyError, match=missing):
        export_amass_motion(
            tmp_path / "motion.npz",
            tmp_path / "unused.pkl",
            tmp_path / "asset",
            asset_id="smpl:fixture:required",
            trusted_model=True,
            front_axis="+Z",
            up_axis="+Y",
        )


def test_source_lineage_is_validated_hashed_and_retained_without_paths(tmp_path):
    source, _ = _surface(tmp_path)
    metadata = {
        "upstream_revision": "a" * 40,
        "checkpoint_sha256": "b" * 64,
        "redacted_prompt_sha256": "c" * 64,
        "seed": 1234,
        "generator_version": "v1.2.3",
        "source_id": "dataset:synthetic:motion",
    }
    lineage = tmp_path / "source_metadata.json"
    lineage.write_text(json.dumps(metadata))
    path = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:lineage",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
        source_metadata=lineage,
    )
    sidecar = (path.parent / "adapter_provenance.json").read_text()
    record = json.loads(sidecar)["sources"]["source_metadata"]
    assert record["metadata"] == metadata
    assert record["sha256"] == hashlib.sha256(lineage.read_bytes()).hexdigest()
    assert str(tmp_path) not in sidecar


@pytest.mark.parametrize(
    "payload",
    (
        {"prompt": "raw private text"},
        {"seed": True},
        {"checkpoint_sha256": "not-a-hash"},
        {"source_id": "C:/private/path"},
        {"upstream_revision": "A" * 40},
    ),
)
def test_unreviewed_free_text_and_invalid_source_lineage_are_rejected(tmp_path, payload):
    source, _ = _surface(tmp_path)
    metadata = tmp_path / "source_metadata.json"
    metadata.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="source metadata"):
        export_smpl_surfaces(
            source,
            tmp_path / "bad",
            asset_id="smpl:fixture:lineage",
            geometry_space="world",
            front_axis="+Z",
            up_axis="+Y",
            source_metadata=metadata,
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "damage",
    ("delete", "tamper", "duplicate_identity", "malformed_identity", "unsupported_identity"),
)
def test_loader_authenticates_reserved_adapter_sidecar(tmp_path, damage):
    from shared.animated_target import animated_target_revision_sha256, canonical_json_bytes

    source, _ = _surface(tmp_path)
    path = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:closure",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
    )
    sidecar = path.parent / "adapter_provenance.json"
    manifest = json.loads(path.read_text())
    if damage == "delete":
        sidecar.unlink()
    elif damage == "tamper":
        sidecar.write_bytes(sidecar.read_bytes() + b" ")
    elif damage == "duplicate_identity":
        manifest["provenance"]["source_ids"].append("adapter_provenance:sha256:" + "a" * 64)
    elif damage == "malformed_identity":
        manifest["provenance"]["source_ids"] = ["adapter_provenance:sha256:invalid"]
    else:
        manifest["provenance"]["source_ids"] = ["adapter_provenance:md5:" + "a" * 32]
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="adapter_provenance"):
        load_animated_target(path)


@pytest.mark.parametrize(
    "damage", ("json_type", "source_shape", "source_hash", "version", "source_type", "license")
)
def test_authenticated_adapter_sidecar_still_requires_valid_minimum_contract(tmp_path, damage):
    from shared.animated_target import animated_target_revision_sha256, canonical_json_bytes

    source, _ = _surface(tmp_path)
    path = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:shape",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
    )
    sidecar = path.parent / "adapter_provenance.json"
    payload = json.loads(sidecar.read_text())
    if damage == "json_type":
        payload = []
    elif damage == "source_shape":
        payload["sources"] = []
    elif damage == "source_hash":
        payload["sources"]["surface_npz"]["sha256"] = "invalid"
    elif damage == "version":
        payload["adapter_contract_version"] = True
    elif damage == "source_type":
        payload["source_type"] = "other:source"
    else:
        payload["redistribution"] = "allowed"
    encoded = canonical_json_bytes(payload)
    sidecar.write_bytes(encoded)
    manifest = json.loads(path.read_text())
    manifest["provenance"]["source_ids"] = [
        "adapter_provenance:sha256:" + hashlib.sha256(encoded).hexdigest()
    ]
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    path.write_bytes(canonical_json_bytes(manifest))
    with pytest.raises(ValueError, match="adapter_provenance"):
        load_animated_target(path)


def test_adapter_sidecar_may_not_redirect_to_external_file(tmp_path, monkeypatch):
    source, _ = _surface(tmp_path)
    path = export_smpl_surfaces(
        source,
        tmp_path / "asset",
        asset_id="smpl:fixture:link",
        geometry_space="world",
        front_axis="+Z",
        up_axis="+Y",
    )
    real = Path.is_symlink
    monkeypatch.setattr(
        Path,
        "is_symlink",
        lambda candidate: candidate.name == "adapter_provenance.json" or real(candidate),
    )
    with pytest.raises(ValueError, match="symlink or junction"):
        load_animated_target(path)


def test_numeric_surface_cli_accepts_explicit_body_axes_without_moving_world_geometry(tmp_path):
    from generator.tools.smpl_to_animated_target import main

    source, expected = _surface(tmp_path)
    output = tmp_path / "body-basis"
    assert (
        main(
            [
                "--surface-npz",
                str(source),
                "--geometry-space",
                "world",
                "--front-axis",
                "+X",
                "--up-axis",
                "+Z",
                "--root-local-front-axis",
                "+Z",
                "--root-local-up-axis",
                "+Y",
                "--asset-id",
                "surface:body-basis",
                "--output-dir",
                str(output),
            ]
        )
        == 0
    )
    asset, binding = _bind(output / "animated_target.json", [0, 0.07, 0.19])
    for row, reference in zip(binding.samples, expected):
        np.testing.assert_allclose(row.vertices_m, reference, atol=1e-14)
    np.testing.assert_allclose(asset.pose_samples[0].vertices, VERTICES @ BASIS.T, atol=1e-14)


@pytest.mark.parametrize("soma30", (False, True))
def test_soma_explicit_body_axes_preserve_skinning_with_nonidentity_root(tmp_path, soma30):
    motion, skin, _, expected = _kimodo_inputs(tmp_path, soma30=soma30)
    extras = {}
    if soma30:
        np.save(tmp_path / "neutral.npy", np.zeros((77, 3)), allow_pickle=False)
        np.save(
            tmp_path / "relaxed.npy", np.broadcast_to(np.eye(3), (77, 3, 3)), allow_pickle=False
        )
        extras = {
            "neutral_joints": tmp_path / "neutral.npy",
            "relaxed_pose": tmp_path / "relaxed.npy",
        }
    path = export_kimodo_motion(
        motion,
        skin,
        tmp_path / "asset",
        asset_id="soma:body-basis",
        root_local_front_axis="+X",
        root_local_up_axis="+Z",
        **extras,
    )
    asset, binding = _bind(path, [0, 0.12])
    for row, reference in zip(binding.samples, expected):
        np.testing.assert_allclose(row.vertices_m, reference @ BASIS.T, atol=1e-14)
    np.testing.assert_allclose(asset.pose_samples[0].vertices @ BASIS.T, VERTICES, atol=1e-14)
