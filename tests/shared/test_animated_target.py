"""Contract tests for source-neutral animated-target assets."""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import pytest

from shared.animated_target import (
    ANIMATED_TARGET_BINDING_VERSION,
    ANIMATED_TARGET_MANIFEST_VERSION,
    AnimatedTargetBindingContext,
    AnimatedTargetBindingTime,
    AnimatedTargetError,
    animated_target_revision_sha256,
    canonical_json_bytes,
    canonical_sha256,
    load_animated_target,
)


def _vertices() -> tuple[np.ndarray, np.ndarray]:
    neutral = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.8, 0.1, 0.0],
            [0.0, 1.2, 0.2],
            [-0.2, 0.3, 1.1],
        ],
        dtype=np.float64,
    )
    reach = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.2, 0.1],
            [0.1, 1.4, 0.25],
            [-0.3, 0.4, 1.3],
        ],
        dtype=np.float64,
    )
    return neutral, reach


def _right_handed_faces() -> np.ndarray:
    return np.array(
        [
            [1, 2, 3],
            [0, 3, 2],
            [0, 1, 3],
            [0, 2, 1],
        ],
        dtype=np.int64,
    )


def _normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_normals = np.cross(
        vertices[faces[:, 1]] - vertices[faces[:, 0]],
        vertices[faces[:, 2]] - vertices[faces[:, 0]],
    )
    result = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(result, faces[:, corner], face_normals)
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    # Triangle winding is a geometric contract using the right-handed cross
    # product even when named source axes form a left-handed frame.
    return result


def test_fast_normals_preserve_corner_major_evidence_bytes():
    from shared.animated_target import _vertex_normals

    rng = np.random.default_rng(4627)
    # A high-valence irregular mesh exercises non-associative sums with widely
    # varying magnitudes, unlike a symmetric tetrahedron.
    vertices = rng.normal(size=(201, 3)) * np.exp(rng.uniform(-5, 5, (201, 1)))
    faces = np.array([[0, i, 1 + i % 200] for i in range(1, 201)])
    for translation in (np.zeros(3), np.array([1234.5, -987.6, 7.8])):
        points = vertices + translation
        expected = _normals(points, faces)
        actual = _vertex_normals(points, faces)
        assert actual.tobytes() == expected.tobytes()


def _pose_payload(
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    normal_policy: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "vertices": vertices.tolist(),
        "faces": faces.tolist(),
    }
    if normal_policy == "transform":
        result["normals"] = _normals(vertices, faces).tolist()
    return result


def _write_canonical(path: Path, value: Any) -> bytes:
    payload = canonical_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def _resource(resource_id: str, relative_path: str, payload: bytes) -> dict[str, Any]:
    return {
        "resource_id": resource_id,
        "path": relative_path,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
        "media_type": "application/vnd.orchav.animated-pose+json",
    }


def _finalize_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    manifest["asset"]["revision_sha256"] = "0" * 64
    manifest["asset"]["revision_sha256"] = animated_target_revision_sha256(manifest)
    _write_canonical(path, manifest)
    return path


def _write_asset(
    directory: Path,
    *,
    reflected: bool = False,
    normal_policy: str = "recompute",
) -> Path:
    neutral, reach = _vertices()
    canonical_faces = _right_handed_faces()
    faces = canonical_faces
    handedness = "left" if reflected else "right"
    basis = (
        [[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        if reflected
        else [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    # Lexical filename order deliberately contradicts physical timestamp order.
    first_path = "poses/z_neutral.json"
    second_path = "poses/a_reach.json"
    first_bytes = _write_canonical(
        directory / first_path,
        _pose_payload(
            neutral,
            faces,
            normal_policy=normal_policy,
        ),
    )
    second_bytes = _write_canonical(
        directory / second_path,
        _pose_payload(
            reach,
            faces,
            normal_policy=normal_policy,
        ),
    )
    topology_sha256 = canonical_sha256(
        {
            "representation": "indexed_triangles",
            "vertex_count": 4,
            "faces": faces.tolist(),
        }
    )
    root_end = [0.0, 1.0, 0.0, 0.0]  # 180 degrees around source +Y, xyzw.
    manifest: dict[str, Any] = {
        "manifest_version": ANIMATED_TARGET_MANIFEST_VERSION,
        "asset": {
            "asset_id": "orchav:synthetic:asymmetric_tetrahedron_v1",
            "revision_sha256": "0" * 64,
        },
        # Declaration order is also intentionally not playback order.
        "resources": [
            _resource("mesh:left_reach", second_path, second_bytes),
            _resource("mesh:neutral_stance", first_path, first_bytes),
        ],
        "coordinates": {
            "source_frame_id": "synthetic:y_up_z_front",
            "source_handedness": handedness,
            "front_axis": "+Z",
            "up_axis": "+Y",
            "length_unit": "m",
            "destination_frame_id": "orchav:world_rh_x_front_z_up_m",
            "source_to_orchav_affine_4x4": [
                [*basis[0], 0.0],
                [*basis[1], 0.0],
                [*basis[2], 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            "scale_ownership": "uniform_scale_in_affine_m_per_source_unit",
            "translation_ownership": "translation_in_affine_destination_m",
        },
        "topology": {
            "representation": "indexed_triangles",
            "mode": "fixed_across_poses",
            "vertex_count": 4,
            "triangle_count": 4,
            "topology_sha256": topology_sha256,
            "closed_surface": True,
            "source_face_winding": "counter_clockwise_from_outside",
            "reflection_winding_policy": "reverse",
            "normal_policy": normal_policy,
        },
        "motion": {
            "time_axis": "asset_time_s",
            "pose": {
                "interpolation": "step",
                "out_of_range": "error",
                "cyclic": False,
                "period_s": None,
                "samples": [
                    {
                        "sample_id": "pose:neutral_stance",
                        "time_s": 0.0,
                        "resource_id": "mesh:neutral_stance",
                    },
                    {
                        "sample_id": "pose:left_reach",
                        "time_s": 1.0,
                        "resource_id": "mesh:left_reach",
                    },
                ],
            },
            "root": {
                "translation_interpolation": "linear",
                "rotation_interpolation": "slerp_shortest",
                "rotation_representation": "active_quaternion_xyzw",
                "out_of_range": "error",
                "cyclic": False,
                "period_s": None,
                "samples": [
                    {
                        "sample_id": "root:origin_facing_front",
                        "time_s": 0.0,
                        "translation": [0.0, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                    {
                        "sample_id": "root:advance_and_turn",
                        "time_s": 1.0,
                        "translation": [0.0, 0.0, 2.0],
                        "rotation_xyzw": root_end,
                    },
                ],
            },
        },
        "provenance": {
            "creator": "ORCHAV contributors",
            "source_type": "orchav_authored_synthetic",
            "source_ids": ["orchav:fixture:asymmetric_pose_root_v1"],
            "description": "Hand-authored asymmetric topology, pose, and root fixture.",
        },
        "license": {
            "spdx_id": "GPL-2.0-only",
            "redistribution": "allowed",
            "notice": "ORCHAV-authored synthetic test data; no third-party motion input.",
        },
    }
    return _finalize_manifest(directory / "animated_target.json", manifest)


def _read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _rewrite_manifest(path: Path, mutate: Callable[[dict[str, Any]], None]) -> None:
    manifest = _read_manifest(path)
    mutate(manifest)
    _finalize_manifest(path, manifest)


def _rewrite_resource(
    manifest_path: Path,
    resource_id: str,
    mutate: Callable[[dict[str, Any]], None],
) -> None:
    manifest = _read_manifest(manifest_path)
    entry = next(item for item in manifest["resources"] if item["resource_id"] == resource_id)
    resource_path = manifest_path.parent / entry["path"]
    payload = json.loads(resource_path.read_text(encoding="utf-8"))
    mutate(payload)
    encoded = _write_canonical(resource_path, payload)
    entry["sha256"] = hashlib.sha256(encoded).hexdigest()
    entry["size_bytes"] = len(encoded)
    _finalize_manifest(manifest_path, manifest)


def _binding_times() -> list[AnimatedTargetBindingTime]:
    return [
        AnimatedTargetBindingTime(
            sample_id="acquisition:alpha:leading_edge",
            scenario_time_s=10.0,
            motion_time_s=0.0,
            pulse_time_s=100.0,
        ),
        AnimatedTargetBindingTime(
            sample_id="acquisition:alpha:mid_turn",
            scenario_time_s=10.5,
            motion_time_s=0.5,
            pulse_time_s=100.5,
        ),
        AnimatedTargetBindingTime(
            sample_id="scene:alpha:reach_keyframe",
            scenario_time_s=11.0,
            motion_time_s=1.0,
            pulse_time_s=None,
        ),
    ]


def _binding_context(
    *,
    retiming_policy: Literal[
        "explicit_per_sample", "affine_scenario_to_motion"
    ] = "affine_scenario_to_motion",
) -> AnimatedTargetBindingContext:
    return AnimatedTargetBindingContext(
        target_id="target:synthetic_asymmetric_subject",
        scenario_clock_id="clock:scenario_reference",
        motion_clock_id="clock:asset_motion",
        pulse_clock_id="clock:sensing_pulse",
        playback_rate=1.0,
        playback_offset_s=-10.0,
        retiming_policy=retiming_policy,
    )


@pytest.mark.parametrize("normal_policy", ["recompute", "transform"])
def test_progress_preserves_asset_binding_and_counts_reused_poses(tmp_path, capsys, normal_policy):
    path = _write_asset(tmp_path, normal_policy=normal_policy)
    manifest = _read_manifest(path)
    first = manifest["motion"]["pose"]["samples"][0]
    manifest["motion"]["pose"]["samples"].append(
        {**first, "sample_id": "pose:repeat", "time_s": 2.0}
    )
    _finalize_manifest(path, manifest)
    baseline = load_animated_target(path)
    expected = baseline.bind(_binding_context(), _binding_times())
    events = []

    def report(stage, done, total):
        events.append((stage, done, total))

    asset = load_animated_target(path, on_progress=report)
    binding = asset.bind(_binding_context(), _binding_times(), on_progress=report)

    assert events == [
        *(("Verifying pose resources", done, 2) for done in range(3)),
        *(("Validating pose geometry", done, 3) for done in range(4)),
        *(("Binding animation poses", done, 3) for done in range(4)),
    ]
    assert asset.revision_sha256 == baseline.revision_sha256
    assert binding.binding_id == expected.binding_id
    assert binding.to_canonical_json() == expected.to_canonical_json()
    np.testing.assert_array_equal(binding.faces, expected.faces)
    for actual, original in zip(binding.samples, expected.samples, strict=True):
        np.testing.assert_array_equal(actual.vertices_m, original.vertices_m)
        np.testing.assert_array_equal(actual.normals, original.normals)
        np.testing.assert_array_equal(
            actual.resolved_root_affine_4x4, original.resolved_root_affine_4x4
        )
    assert asset.pose_samples[-1].vertices is asset.pose_samples[0].vertices
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("failure", ["resource", "geometry", "binding", "identity"])
def test_progress_does_not_complete_failed_stage(tmp_path, monkeypatch, failure):
    import shared.animated_target as module

    path = _write_asset(tmp_path)
    events = []

    def report(stage, done, total):
        events.append((stage, done, total))

    if failure == "resource":
        resource = path.parent / "poses/z_neutral.json"
        resource.write_bytes(resource.read_bytes() + b" ")
        stage = "Verifying pose resources"
        with pytest.raises(AnimatedTargetError, match="size mismatch"):
            load_animated_target(path, on_progress=report)
    elif failure == "geometry":
        _rewrite_resource(
            path,
            "mesh:left_reach",
            lambda payload: payload["faces"][0].__setitem__(1, payload["faces"][0][0]),
        )
        stage = "Validating pose geometry"
        with pytest.raises(AnimatedTargetError, match="repeated triangle indices"):
            load_animated_target(path, on_progress=report)
    else:
        asset = load_animated_target(path)
        times = _binding_times()
        context = _binding_context(retiming_policy="explicit_per_sample")
        stage = "Binding animation poses"
        if failure == "binding":
            times[-1] = replace(times[-1], motion_time_s=3.0)
            message = "outside"
        else:

            def fail_identity(_payload):
                raise AnimatedTargetError("binding identity failure")

            monkeypatch.setattr(module, "canonical_sha256", fail_identity)
            message = "binding identity failure"
        with pytest.raises(AnimatedTargetError, match=message):
            asset.bind(context, times, on_progress=report)

    failed_stage = [(done, total) for label, done, total in events if label == stage]
    assert failed_stage[0][0] == 0
    assert all(done < total for done, total in failed_stage)


@pytest.mark.parametrize("normal_policy", ["recompute", "transform"])
def test_repeated_pose_resources_decode_once_and_keep_sample_identity(
    tmp_path, monkeypatch, normal_policy
):
    import shared.animated_target as module

    path = _write_asset(tmp_path, normal_policy=normal_policy)
    manifest = _read_manifest(path)
    original = manifest["motion"]["pose"]["samples"][0]
    manifest["motion"]["pose"]["samples"].append(
        {**original, "sample_id": "pose:repeat", "time_s": 2.0}
    )
    _finalize_manifest(path, manifest)
    parsed_vertices = []
    parse = module._numeric_matrix

    def count(value, field, columns):
        if field.endswith(".vertices"):
            parsed_vertices.append(field)
        return parse(value, field, columns)

    monkeypatch.setattr(module, "_numeric_matrix", count)
    asset = load_animated_target(path)
    assert len(parsed_vertices) == 2
    first, _, repeated = asset.pose_samples
    assert repeated.sample_id == "pose:repeat" and repeated.time_s == 2.0
    assert repeated.vertices is first.vertices
    assert repeated.faces is first.faces
    assert repeated.normals is first.normals
    assert not repeated.vertices.flags.writeable and not repeated.faces.flags.writeable


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("policy", ["error", "hold", "loop"])
def test_center_projection_matches_binding_without_normals_or_evidence(
    tmp_path, monkeypatch, reflected, policy
):
    import shared.animated_target as module

    path = _write_asset(tmp_path, reflected=reflected)
    manifest = _read_manifest(path)
    for track in ("pose", "root"):
        if policy == "loop":
            first = manifest["motion"][track]["samples"][0]
            manifest["motion"][track]["samples"].append(
                {**first, "sample_id": f"{track}:loop_end", "time_s": 2.0}
            )
        manifest["motion"][track].update(
            out_of_range=policy, cyclic=policy == "loop", period_s=2.0 if policy == "loop" else None
        )
    _finalize_manifest(path, manifest)
    asset = load_animated_target(path)
    requests = _binding_times()
    if policy != "error":
        requests = [
            replace(request, motion_time_s=t) for request, t in zip(requests, (-0.5, 0.5, 2.5))
        ]
    context = _binding_context(retiming_policy="explicit_per_sample")
    binding = asset.bind(context, requests)
    expected = np.array(
        [(s.vertices_m.min(axis=0) + s.vertices_m.max(axis=0)) * 0.5 for s in binding.samples]
    )
    monkeypatch.setattr(module, "_vertex_normals", lambda *_: pytest.fail("Unneeded normals"))
    monkeypatch.setattr(module, "canonical_sha256", lambda *_: pytest.fail("Unneeded evidence"))
    actual = asset.sample_centers(context, requests)
    np.testing.assert_array_equal(actual, expected)
    assert not actual.flags.writeable
    with pytest.raises(AnimatedTargetError, match="strictly increasing"):
        asset.sample_centers(
            context, (requests[0], replace(requests[0], sample_id="duplicate_time"))
        )
    if policy == "error":
        with pytest.raises(AnimatedTargetError, match="outside"):
            asset.sample_centers(context, (replace(requests[0], motion_time_s=5.0),))


def test_load_and_bind_uses_manifest_time_not_filename_or_resource_order(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path))

    assert [sample.resource_id for sample in asset.pose_samples] == [
        "mesh:neutral_stance",
        "mesh:left_reach",
    ]
    assert [resource.path for resource in asset.resources] == [
        "poses/a_reach.json",
        "poses/z_neutral.json",
    ]
    assert [sample.time_s for sample in asset.pose_samples] == [0.0, 1.0]

    binding = asset.bind(_binding_context(), _binding_times())
    repeated = asset.bind(_binding_context(), _binding_times())

    assert binding.binding_id == repeated.binding_id
    assert binding.to_canonical_json() == repeated.to_canonical_json()
    assert len(binding.binding_id) == 64
    exported = json.loads(binding.to_canonical_json())
    assert exported["binding_version"] == ANIMATED_TARGET_BINDING_VERSION
    assert exported["binding_id"] == binding.binding_id
    assert exported["context"] == _binding_context().to_dict()
    assert exported["samples"][1]["scenario_time_s"] == 10.5
    assert exported["samples"][1]["pulse_time_s"] == 100.5

    middle = binding.samples[1]
    assert middle.pose_sample_id == "pose:neutral_stance"  # step, never vertex blending
    assert middle.root_lower_sample_id == "root:origin_facing_front"
    assert middle.root_upper_sample_id == "root:advance_and_turn"
    assert middle.root_fraction == pytest.approx(0.5)
    assert middle.evaluated_pose_time_s == 0.5
    assert middle.evaluated_root_time_s == 0.5
    assert middle.valid is True
    assert middle.resolved_root_affine_4x4[:3, 3] == pytest.approx([1.0, 0.0, 0.0])
    assert exported["samples"][1]["validity"] == {"valid": True, "reason": None}
    assert len(exported["samples"][1]["root"]["resolved_affine_sha256"]) == 64
    # At t=.5 the root has translated one source +Z meter and SLERPed to
    # +90 degrees around source +Y. The declared basis maps (x,y,z)->(z,x,y).
    assert middle.vertices_m[1] == pytest.approx([0.2, 0.0, 0.1], abs=1e-12)
    assert middle.vertices_m[3] == pytest.approx([1.2, 1.1, 0.3], abs=1e-12)
    assert binding.samples[2].pose_sample_id == "pose:left_reach"
    assert not middle.vertices_m.flags.writeable
    assert not binding.faces.flags.writeable
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        middle.vertices_m.setflags(write=True)
    with pytest.raises(ValueError, match="cannot set WRITEABLE flag"):
        binding.faces.setflags(write=True)


def test_reflection_reverses_winding_and_recomputes_outward_normals(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path, reflected=True))
    binding = asset.bind(_binding_context(), [_binding_times()[0]])
    vertices = binding.samples[0].vertices_m
    faces = binding.faces
    normals = binding.samples[0].normals

    assert np.linalg.det(asset.source_to_orchav_basis) == pytest.approx(-1.0)
    assert np.array_equal(asset.canonical_faces, _right_handed_faces()[:, [0, 2, 1]])
    assert vertices[1] == pytest.approx([0.0, -0.8, 0.1])
    assert vertices[3] == pytest.approx([1.1, 0.2, 0.3])
    expected = _normals(vertices, faces)
    assert normals == pytest.approx(expected, abs=1e-12)


def test_complete_affine_owns_uniform_scale_translation_and_resolved_root(
    tmp_path: Path,
) -> None:
    manifest_path = _write_asset(tmp_path)

    def transform(manifest: dict[str, Any]) -> None:
        affine = np.asarray(
            manifest["coordinates"]["source_to_orchav_affine_4x4"],
            dtype=np.float64,
        )
        affine[:3, :3] *= 2.0
        affine[:3, 3] = (3.0, 4.0, 5.0)
        manifest["coordinates"]["source_to_orchav_affine_4x4"] = affine.tolist()

    _rewrite_manifest(manifest_path, transform)
    asset = load_animated_target(manifest_path)
    bound = asset.bind(_binding_context(), [_binding_times()[0]]).samples[0]

    assert asset.meters_per_source_unit == pytest.approx(2.0)
    assert asset.source_to_orchav_translation_m == pytest.approx([3.0, 4.0, 5.0])
    assert asset.source_to_orchav_affine_4x4.shape == (4, 4)
    assert bound.vertices_m[1] == pytest.approx([3.0, 5.6, 5.2])
    assert bound.vertices_m[3] == pytest.approx([5.2, 3.6, 5.6])
    assert bound.resolved_root_affine_4x4 == pytest.approx(asset.source_to_orchav_affine_4x4)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda matrix: matrix[3].__setitem__(3, 2.0),
            "final row",
        ),
        (
            lambda matrix: matrix[0].__setitem__(2, 2.0),
            "positive uniform scale",
        ),
    ],
    ids=("non-affine-final-row", "non-uniform-scale"),
)
def test_rejects_incomplete_or_nonuniform_coordinate_affines(
    tmp_path: Path,
    mutate: Callable[[list[list[float]]], None],
    match: str,
) -> None:
    manifest_path = _write_asset(tmp_path)

    def change(manifest: dict[str, Any]) -> None:
        matrix = manifest["coordinates"]["source_to_orchav_affine_4x4"]
        mutate(matrix)

    _rewrite_manifest(manifest_path, change)
    with pytest.raises(AnimatedTargetError, match=match):
        load_animated_target(manifest_path)


def test_reflection_transforms_declared_source_normals(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path, reflected=True, normal_policy="transform"))
    binding = asset.bind(_binding_context(), [_binding_times()[0]])
    expected = _normals(binding.samples[0].vertices_m, binding.faces)

    assert binding.samples[0].normals == pytest.approx(expected, abs=1e-12)


@pytest.mark.parametrize(
    ("track", "mutate", "match"),
    [
        (
            "pose",
            lambda samples: samples[1].pop("time_s"),
            r"motion\.pose\.samples\[1\].*missing=\['time_s'\]",
        ),
        (
            "pose",
            lambda samples: samples[1].update(time_s=0.0),
            "pose.samples time_s must be strictly increasing",
        ),
        (
            "pose",
            lambda samples: samples[0].update(time_s=2.0),
            "pose.samples time_s must be strictly increasing",
        ),
        (
            "root",
            lambda samples: samples[1].pop("time_s"),
            r"motion\.root\.samples\[1\].*missing=\['time_s'\]",
        ),
        (
            "root",
            lambda samples: samples[1].update(time_s=0.0),
            "root.samples time_s must be strictly increasing",
        ),
        (
            "root",
            lambda samples: samples[0].update(time_s=2.0),
            "root.samples time_s must be strictly increasing",
        ),
    ],
    ids=(
        "pose-missing",
        "pose-duplicate",
        "pose-out-of-order",
        "root-missing",
        "root-duplicate",
        "root-out-of-order",
    ),
)
def test_rejects_missing_duplicate_and_out_of_order_track_times(
    tmp_path: Path,
    track: str,
    mutate: Callable[[list[dict[str, Any]]], None],
    match: str,
) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: mutate(manifest["motion"][track]["samples"]),
    )

    with pytest.raises(AnimatedTargetError, match=match):
        load_animated_target(manifest_path)


@pytest.mark.parametrize("bad_id", ["17", "pose-00017", "sample_9", "mesh.json", "root"])
def test_rejects_ordinal_filename_and_generic_sample_ids(tmp_path: Path, bad_id: str) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest["motion"]["pose"]["samples"][0].update(sample_id=bad_id),
    )

    with pytest.raises(AnimatedTargetError, match="semantic|suffix|filename"):
        load_animated_target(manifest_path)


@pytest.mark.parametrize(
    "bad_id",
    [
        "/mnt/private/subject",
        "../private/subject",
        r"C:\\private\\subject",
        "C:private_subject",
        "file:/home/private/subject",
        "https://private.example/subject",
        "~/private_subject",
    ],
)
def test_rejects_path_like_asset_identity(tmp_path: Path, bad_id: str) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest["asset"].update(asset_id=bad_id),
    )

    with pytest.raises(AnimatedTargetError, match="semantic ID, not a path"):
        load_animated_target(manifest_path)


@pytest.mark.parametrize("bad_clock_id", ["/private/clock", r"C:\\private\\clock", "file:/clock"])
def test_binding_context_rejects_path_like_clock_identity(bad_clock_id: str) -> None:
    with pytest.raises(AnimatedTargetError, match="semantic ID, not a path"):
        AnimatedTargetBindingContext(
            target_id="target:subject",
            scenario_clock_id=bad_clock_id,
            motion_clock_id="clock:motion",
            pulse_clock_id=None,
            playback_rate=1.0,
            playback_offset_s=0.0,
            retiming_policy="explicit_per_sample",
        )


def test_loader_rejects_noncanonical_quaternion_antipode(tmp_path: Path) -> None:
    negative_path = _write_asset(tmp_path)
    _rewrite_manifest(
        negative_path,
        lambda manifest: manifest["motion"]["root"]["samples"][1].update(
            rotation_xyzw=[0.0, -1.0, 0.0, 0.0]
        ),
    )

    with pytest.raises(AnimatedTargetError, match="canonical quaternion antipode"):
        load_animated_target(negative_path)


def test_binding_rejects_duplicate_ordinal_and_nonphysical_clock_order(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path))
    duplicate = _binding_times()[:2]
    duplicate[1] = AnimatedTargetBindingTime(
        sample_id=duplicate[0].sample_id,
        scenario_time_s=10.1,
        motion_time_s=0.5,
        pulse_time_s=100.1,
    )
    with pytest.raises(AnimatedTargetError, match="duplicate binding sample_id"):
        asset.bind(_binding_context(), duplicate)

    ordinal = [
        AnimatedTargetBindingTime("pulse-0001", 0.0, 0.0, 0.0),
    ]
    with pytest.raises(AnimatedTargetError, match="ordinal-derived"):
        asset.bind(_binding_context(), ordinal)

    bad_scenario = _binding_times()[:2]
    bad_scenario.reverse()
    with pytest.raises(AnimatedTargetError, match="scenario_time_s must be strictly increasing"):
        asset.bind(_binding_context(), bad_scenario)

    bad_pulses = _binding_times()[:2]
    bad_pulses[1] = AnimatedTargetBindingTime(
        sample_id=bad_pulses[1].sample_id,
        scenario_time_s=bad_pulses[1].scenario_time_s,
        motion_time_s=bad_pulses[1].motion_time_s,
        pulse_time_s=99.0,
    )
    with pytest.raises(AnimatedTargetError, match="pulse_time_s must be strictly increasing"):
        asset.bind(_binding_context(), bad_pulses)

    duplicate_pulse = _binding_times()[:2]
    duplicate_pulse[1] = AnimatedTargetBindingTime(
        sample_id=duplicate_pulse[1].sample_id,
        scenario_time_s=duplicate_pulse[1].scenario_time_s,
        motion_time_s=duplicate_pulse[1].motion_time_s,
        pulse_time_s=duplicate_pulse[0].pulse_time_s,
    )
    with pytest.raises(AnimatedTargetError, match="pulse_time_s must be strictly increasing"):
        asset.bind(_binding_context(), duplicate_pulse)


def test_binding_context_requires_clock_identity_and_enforces_affine_retiming(
    tmp_path: Path,
) -> None:
    asset = load_animated_target(_write_asset(tmp_path))
    without_pulse_clock = AnimatedTargetBindingContext(
        target_id="target:synthetic_asymmetric_subject",
        scenario_clock_id="clock:scenario_reference",
        motion_clock_id="clock:asset_motion",
        pulse_clock_id=None,
        playback_rate=1.0,
        playback_offset_s=-10.0,
        retiming_policy="affine_scenario_to_motion",
    )
    with pytest.raises(AnimatedTargetError, match="pulse_clock_id is required"):
        asset.bind(without_pulse_clock, _binding_times())

    inconsistent = _binding_times()
    inconsistent[1] = AnimatedTargetBindingTime(
        sample_id=inconsistent[1].sample_id,
        scenario_time_s=inconsistent[1].scenario_time_s,
        motion_time_s=0.75,
        pulse_time_s=inconsistent[1].pulse_time_s,
    )
    with pytest.raises(AnimatedTargetError, match="affine scenario-to-motion"):
        asset.bind(_binding_context(), inconsistent)

    with pytest.raises(AnimatedTargetError, match="playback_rate must be positive"):
        AnimatedTargetBindingContext(
            target_id="target:synthetic_asymmetric_subject",
            scenario_clock_id="clock:scenario_reference",
            motion_clock_id="clock:asset_motion",
            pulse_clock_id="clock:sensing_pulse",
            playback_rate=0.0,
            playback_offset_s=0.0,
            retiming_policy="explicit_per_sample",
        )


def test_resource_hash_revision_path_and_reference_closure_are_fail_closed(tmp_path: Path) -> None:
    corrupt_path = _write_asset(tmp_path / "corrupt")
    resource = corrupt_path.parent / "poses/z_neutral.json"
    resource.write_bytes(resource.read_bytes() + b"\n")
    with pytest.raises(AnimatedTargetError, match="SHA-256 mismatch|size mismatch"):
        load_animated_target(corrupt_path)

    revision_path = _write_asset(tmp_path / "revision")
    manifest = _read_manifest(revision_path)
    manifest["asset"]["revision_sha256"] = "f" * 64
    _write_canonical(revision_path, manifest)
    with pytest.raises(AnimatedTargetError, match="canonical manifest revision"):
        load_animated_target(revision_path)

    traversal_path = _write_asset(tmp_path / "traversal")
    _rewrite_manifest(
        traversal_path,
        lambda value: value["resources"][0].update(path="../outside.json"),
    )
    with pytest.raises(AnimatedTargetError, match="safe relative path"):
        load_animated_target(traversal_path)

    closure_path = _write_asset(tmp_path / "closure")
    manifest = _read_manifest(closure_path)
    extra_payload = _write_canonical(
        closure_path.parent / "poses/unused.json",
        json.loads((closure_path.parent / "poses/z_neutral.json").read_text()),
    )
    manifest["resources"].append(
        _resource("mesh:unused_semantic_pose", "poses/unused.json", extra_payload)
    )
    _finalize_manifest(closure_path, manifest)
    with pytest.raises(AnimatedTargetError, match="exact hash closure"):
        load_animated_target(closure_path)

    symlink_path = _write_asset(tmp_path / "symlink")
    manifest = _read_manifest(symlink_path)
    entry = next(
        item for item in manifest["resources"] if item["resource_id"] == "mesh:neutral_stance"
    )
    link = symlink_path.parent / "poses/neutral_link.json"
    try:
        link.symlink_to("z_neutral.json")
    except OSError as exc:
        if sys.platform == "win32" and getattr(exc, "winerror", None) == 1314:
            pytest.skip("Windows account lacks symlink creation privilege (WinError 1314)")
        raise
    entry["path"] = "poses/neutral_link.json"
    _finalize_manifest(symlink_path, manifest)
    with pytest.raises(AnimatedTargetError, match="must not traverse a symlink"):
        load_animated_target(symlink_path)


def test_pose_is_parsed_from_the_authenticated_resource_bytes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_path = _write_asset(tmp_path)
    manifest = _read_manifest(manifest_path)
    entry = next(
        item for item in manifest["resources"] if item["resource_id"] == "mesh:neutral_stance"
    )
    resource_path = manifest_path.parent / entry["path"]
    original_bytes = resource_path.read_bytes()
    original_pose = json.loads(original_bytes)
    changed_pose = json.loads(original_bytes)
    changed_pose["vertices"][0][0] += 0.125
    changed_bytes = canonical_json_bytes(changed_pose)
    original_read_bytes = Path.read_bytes
    resource_reads = 0

    def swap_after_authenticated_read(path: Path) -> bytes:
        nonlocal resource_reads
        if path == resource_path:
            resource_reads += 1
            if resource_reads == 1:
                resource_path.write_bytes(changed_bytes)
                return original_bytes
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swap_after_authenticated_read)
    asset = load_animated_target(manifest_path)
    neutral = next(
        sample for sample in asset.pose_samples if sample.resource_id == "mesh:neutral_stance"
    )

    np.testing.assert_allclose(neutral.vertices, original_pose["vertices"], atol=0.0, rtol=0.0)
    assert resource_reads == 1


def test_authenticated_provenance_and_license_are_deeply_immutable(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path))
    revision = asset.revision_sha256

    with pytest.raises(TypeError):
        asset.license["redistribution"] = "prohibited"  # type: ignore[index]
    with pytest.raises(TypeError):
        asset.provenance["creator"] = "changed"  # type: ignore[index]
    assert isinstance(asset.provenance["source_ids"], tuple)
    assert asset.revision_sha256 == revision


def test_manifest_and_pose_resources_reject_duplicate_json_object_keys(tmp_path: Path) -> None:
    manifest_path = _write_asset(tmp_path / "manifest_duplicate")
    manifest_payload = manifest_path.read_text(encoding="utf-8").replace(
        '"manifest_version":1',
        '"manifest_version":1,"manifest_version":1',
        1,
    )
    manifest_path.write_text(manifest_payload, encoding="utf-8")
    with pytest.raises(AnimatedTargetError, match="duplicate JSON object key 'manifest_version'"):
        load_animated_target(manifest_path)

    resource_manifest_path = _write_asset(tmp_path / "resource_duplicate")
    manifest = _read_manifest(resource_manifest_path)
    resource_entry = next(
        entry for entry in manifest["resources"] if entry["resource_id"] == "mesh:neutral_stance"
    )
    resource_path = resource_manifest_path.parent / resource_entry["path"]
    resource_payload = resource_path.read_text(encoding="utf-8").replace(
        '"faces":',
        '"faces":[],"faces":',
        1,
    )
    resource_path.write_text(resource_payload, encoding="utf-8")
    encoded = resource_payload.encode("utf-8")
    resource_entry["sha256"] = hashlib.sha256(encoded).hexdigest()
    resource_entry["size_bytes"] = len(encoded)
    _finalize_manifest(resource_manifest_path, manifest)
    with pytest.raises(AnimatedTargetError, match="duplicate JSON object key 'faces'"):
        load_animated_target(resource_manifest_path)


@pytest.mark.parametrize(
    ("resource_id", "mutate", "match"),
    [
        (
            "mesh:left_reach",
            lambda payload: payload.update(faces=list(reversed(payload["faces"]))),
            "changes topology",
        ),
        (
            "mesh:neutral_stance",
            lambda payload: payload["faces"][0].__setitem__(0, 99),
            "out-of-range face indices",
        ),
        (
            "mesh:neutral_stance",
            lambda payload: payload["vertices"].pop(),
            r"vertices shape must be \(4, 3\)",
        ),
        (
            "mesh:neutral_stance",
            lambda payload: payload["faces"].__setitem__(3, payload["faces"][0]),
            "not a consistently wound closed surface",
        ),
    ],
    ids=("per-pose-face-change", "face-index", "vertex-count", "closed-edges"),
)
def test_rejects_per_pose_topology_changes(
    tmp_path: Path,
    resource_id: str,
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_resource(manifest_path, resource_id, mutate)

    with pytest.raises(AnimatedTargetError, match=match):
        load_animated_target(manifest_path)


def test_large_finite_pose_geometry_fails_before_nonfinite_normals(tmp_path: Path) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_resource(
        manifest_path,
        "mesh:neutral_stance",
        lambda payload: payload.update(
            vertices=[
                [component * 1.0e155 for component in vertex] for vertex in payload["vertices"]
            ]
        ),
    )

    with pytest.raises(AnimatedTargetError, match="derived face geometry is not finite"):
        load_animated_target(manifest_path)


def test_arbitrarily_large_json_integer_raises_animated_contract_error(tmp_path: Path) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_manifest(
        manifest_path,
        lambda manifest: manifest["motion"]["root"]["samples"][0]["translation"].__setitem__(
            0,
            10**1000,
        ),
    )

    with pytest.raises(AnimatedTargetError, match="must be finite"):
        load_animated_target(manifest_path)


def test_arbitrarily_large_face_index_raises_animated_contract_error(tmp_path: Path) -> None:
    manifest_path = _write_asset(tmp_path)
    _rewrite_resource(
        manifest_path,
        "mesh:neutral_stance",
        lambda payload: payload["faces"][0].__setitem__(0, 10**1000),
    )

    with pytest.raises(AnimatedTargetError, match="signed 64-bit integer indices"):
        load_animated_target(manifest_path)


def test_transform_normal_policy_requires_normals_in_every_pose(tmp_path: Path) -> None:
    manifest_path = _write_asset(tmp_path, normal_policy="transform")
    _rewrite_resource(
        manifest_path,
        "mesh:left_reach",
        lambda payload: payload.pop("normals"),
    )

    with pytest.raises(AnimatedTargetError, match=r"missing=\['normals'\]"):
        load_animated_target(manifest_path)


def test_hold_error_and_declared_cyclic_loop_policies(tmp_path: Path) -> None:
    error_asset = load_animated_target(_write_asset(tmp_path / "error"))
    request = AnimatedTargetBindingTime(
        "scene:alpha:after_motion",
        scenario_time_s=2.0,
        motion_time_s=2.0,
    )
    with pytest.raises(AnimatedTargetError, match=r"outside \[0.0, 1.0\]"):
        error_asset.bind(
            _binding_context(retiming_policy="explicit_per_sample"),
            [request],
        )

    hold_path = _write_asset(tmp_path / "hold")

    def hold(manifest: dict[str, Any]) -> None:
        manifest["motion"]["pose"]["out_of_range"] = "hold"
        manifest["motion"]["root"]["out_of_range"] = "hold"

    _rewrite_manifest(hold_path, hold)
    held = (
        load_animated_target(hold_path)
        .bind(
            _binding_context(retiming_policy="explicit_per_sample"),
            [request],
        )
        .samples[0]
    )
    assert held.pose_sample_id == "pose:left_reach"
    assert held.root_lower_sample_id == "root:advance_and_turn"
    assert held.evaluated_pose_time_s == 1.0
    assert held.evaluated_root_time_s == 1.0
    assert held.pose_cycle == held.root_cycle == 0

    invalid_loop_path = _write_asset(tmp_path / "invalid_loop")
    _rewrite_manifest(
        invalid_loop_path,
        lambda manifest: manifest["motion"]["pose"].update(out_of_range="loop"),
    )
    with pytest.raises(AnimatedTargetError, match="requires cyclic=true"):
        load_animated_target(invalid_loop_path)

    loop_path = _write_asset(tmp_path / "loop")
    manifest = _read_manifest(loop_path)
    first_resource_id = manifest["motion"]["pose"]["samples"][0]["resource_id"]
    manifest["motion"]["pose"]["samples"][1]["resource_id"] = first_resource_id
    manifest["resources"] = [
        entry for entry in manifest["resources"] if entry["resource_id"] == first_resource_id
    ]
    for track_name in ("pose", "root"):
        manifest["motion"][track_name].update(
            out_of_range="loop",
            cyclic=True,
            period_s=1.0,
        )
    first_root = manifest["motion"]["root"]["samples"][0]
    manifest["motion"]["root"]["samples"][1].update(
        translation=first_root["translation"],
        rotation_xyzw=first_root["rotation_xyzw"],
    )
    _finalize_manifest(loop_path, manifest)
    looped = (
        load_animated_target(loop_path)
        .bind(
            _binding_context(retiming_policy="explicit_per_sample"),
            [
                AnimatedTargetBindingTime(
                    "scene:alpha:second_cycle_quarter",
                    scenario_time_s=2.0,
                    motion_time_s=1.25,
                )
            ],
        )
        .samples[0]
    )
    assert looped.evaluated_pose_time_s == pytest.approx(0.25)
    assert looped.evaluated_root_time_s == pytest.approx(0.25)
    assert looped.pose_cycle == looped.root_cycle == 1
    assert looped.pose_sample_id == "pose:neutral_stance"


def test_binding_identity_changes_with_explicit_clock_mapping(tmp_path: Path) -> None:
    asset = load_animated_target(_write_asset(tmp_path))
    baseline_times = _binding_times()
    shifted_times = list(baseline_times)
    shifted_times[1] = AnimatedTargetBindingTime(
        sample_id=shifted_times[1].sample_id,
        scenario_time_s=shifted_times[1].scenario_time_s,
        motion_time_s=0.75,
        pulse_time_s=shifted_times[1].pulse_time_s,
    )

    context = _binding_context(retiming_policy="explicit_per_sample")
    baseline = asset.bind(context, baseline_times)
    shifted = asset.bind(context, shifted_times)

    assert baseline.binding_id != shifted.binding_id
    assert baseline.samples[1].root_fraction == pytest.approx(0.5)
    assert shifted.samples[1].root_fraction == pytest.approx(0.75)


def test_core_contract_has_no_restricted_motion_runtime_imports() -> None:
    source_path = Path(__file__).parents[2] / "shared/animated_target.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported_roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_roots.add(node.module.split(".", 1)[0])

    assert imported_roots.isdisjoint({"smplx", "chumpy", "torch", "kimodo"})
