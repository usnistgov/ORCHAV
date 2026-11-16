import importlib.util
import sys
from pathlib import Path


def _load_tool_module():
    tool_path = (
        Path(__file__).resolve().parents[3] / "generator" / "tools" / "render_mesh_previews.py"
    )
    spec = importlib.util.spec_from_file_location(
        "generator.tools.render_mesh_previews_test",
        tool_path,
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_tool_module = _load_tool_module()
build_parser = _tool_module.build_parser
find_mesh_libraries = _tool_module.find_mesh_libraries
resolve_preview_outputs = _tool_module.resolve_preview_outputs
select_sample_indices = _tool_module.select_sample_indices


def test_select_sample_indices_evenly_spreads_frames():
    assert select_sample_indices(10, 4) == [0, 3, 6, 9]


def test_resolve_preview_outputs_uses_preview_subdir(tmp_path):
    outputs = resolve_preview_outputs(tmp_path / "smpl_wavepoint_30fps", "preview")

    assert outputs.output_dir == tmp_path / "smpl_wavepoint_30fps" / "preview"
    assert outputs.contact_sheet.name == "contact_sheet.png"
    assert outputs.front_gif.name == "front.gif"


def test_find_mesh_libraries_filters_for_multi_frame_dirs(tmp_path):
    multi = tmp_path / "multi"
    multi.mkdir()
    (multi / "frame_0001.ply").write_text("ply\n")
    (multi / "frame_0002.ply").write_text("ply\n")

    single = tmp_path / "single"
    single.mkdir()
    (single / "frame_0001.ply").write_text("ply\n")

    empty = tmp_path / "empty"
    empty.mkdir()

    found = find_mesh_libraries(tmp_path, min_ply_count=2)

    assert found == [multi]


def test_build_parser_supports_single_and_batch_modes():
    parser = build_parser()

    args = parser.parse_args(["--mesh-dir", "libraries/targets/smpl_wavepoint_30fps"])
    assert args.mesh_dir == Path("libraries/targets/smpl_wavepoint_30fps")
    assert args.root_dir is None

    args = parser.parse_args(["--root-dir", "libraries/targets", "--skip-existing"])
    assert args.root_dir == Path("libraries/targets")
    assert args.skip_existing is True
