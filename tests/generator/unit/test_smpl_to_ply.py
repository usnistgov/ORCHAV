import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_tool_module():
    tool_path = Path(__file__).resolve().parents[3] / "generator" / "tools" / "smpl_to_ply.py"
    spec = importlib.util.spec_from_file_location("generator.tools.smpl_to_ply_test", tool_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_tool_module = _load_tool_module()
build_parser = _tool_module.build_parser
transform_to_orchav = _tool_module.transform_to_orchav


def test_transform_to_orchav_centers_on_centroid_by_default():
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 5.0],
            [4.0, 3.0, 7.0],
        ]
    )

    transformed = transform_to_orchav(vertices, center_xy=True, ground_feet=True)

    assert np.allclose(transformed[:, :2].mean(axis=0), [0.0, 0.0])
    assert np.isclose(transformed[:, 2].min(), 0.0)


def test_transform_to_orchav_can_center_on_reference_point():
    vertices = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 5.0],
        ]
    )
    pelvis = np.array([5.0, 7.0, 11.0])

    transformed = transform_to_orchav(
        vertices,
        center_xy=True,
        ground_feet=False,
        xy_reference=pelvis,
    )

    expected = np.array(
        [
            [-4.0, 8.0, 2.0],
            [-3.0, 6.0, 3.0],
        ]
    )

    assert np.allclose(transformed, expected)


def test_build_parser_supports_xy_anchor_choice():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--amass-npz",
            "motions/CMU/13/13_27_poses.npz",
            "--model-folder",
            "models/smpl_organized",
            "--xy-anchor",
            "pelvis",
        ]
    )

    assert args.xy_anchor == "pelvis"


def test_build_parser_supports_preview_flags():
    parser = build_parser()

    args = parser.parse_args(
        [
            "--amass-npz",
            "motions/CMU/15/15_08_poses.npz",
            "--model-folder",
            "models/smpl_organized",
            "--render-preview",
            "--preview-contact-samples",
            "8",
            "--preview-gif-frames",
            "40",
        ]
    )

    assert args.render_preview is True
    assert args.preview_contact_samples == 8
    assert args.preview_gif_frames == 40


def test_load_amass_motion_rejects_object_arrays_and_missing_time(tmp_path):
    import pytest

    unsafe = tmp_path / "object.npz"
    np.savez(unsafe, poses=np.array([{"not": "numeric"}], dtype=object))
    with pytest.raises(ValueError, match="Object arrays"):
        _tool_module.load_amass_motion(unsafe)
    missing = tmp_path / "missing.npz"
    np.savez(missing, poses=np.zeros((2, 72)))
    with pytest.raises(ValueError, match="physical source frame rate"):
        _tool_module.load_amass_motion(missing)


def test_load_amass_motion_preserves_scalar_bytes_gender(tmp_path):
    source = tmp_path / "motion.npz"
    np.savez(source, poses=np.zeros((2, 72)), mocap_framerate=30.0, gender=np.array(b"female"))
    motion = _tool_module.load_amass_motion(source)
    assert motion["gender"] == "female"
    assert motion["mocap_framerate"] == 30.0
