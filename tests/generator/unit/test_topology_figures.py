"""Geometry-only overview contracts, independent of ray tracing and recorded data."""

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from generator.core.scenario_actors.quaternion import Quaternion
from generator.figures.topology import (
    is_stationary,
    load_topology,
    orientation_indices,
    render_topology,
    rotation_matrix,
)
from generator.figures.topology_scene import (
    is_etoile_ground,
    material_rgb,
    scene_file,
    transform_matrix,
)


def test_material_precedence_and_native_fallback():
    bsdf = ET.fromstring(
        '<bsdf type="itu-radio-material"><string name="type" value="metal"/></bsdf>'
    )
    np.testing.assert_allclose(material_rgb(bsdf), [0.22, 0.22, 0.254])
    ET.SubElement(bsdf, "rgb", name="reflectance", value="0.1, 0.2, 0.3")
    np.testing.assert_allclose(material_rgb(bsdf), [0.1, 0.2, 0.3])
    ET.SubElement(bsdf, "rgb", name="color", value="0.4")
    np.testing.assert_allclose(material_rgb(bsdf), [0.4, 0.4, 0.4])
    assert material_rgb(ET.fromstring('<bsdf type="unknown"/>')) is None


def test_etoile_ground_exception_and_transform_order():
    shape = ET.fromstring('<shape id="mesh-Plane"/>')
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0.0]])
    assert is_etoile_ground("etoile.xml", shape, Path("Plane.ply"), vertices)
    assert not is_etoile_ground("other.xml", shape, Path("Plane.ply"), vertices)
    raised = vertices.copy()
    raised[0, 2] = 1
    assert not is_etoile_ground("etoile.xml", shape, Path("Plane.ply"), raised)
    matrix = transform_matrix(
        ET.fromstring('<transform><translate x="1"/><scale value="2"/></transform>')
    )
    np.testing.assert_allclose(matrix @ [0, 0, 0, 1], [2, 0, 0, 1])


@pytest.fixture
def authored(tmp_path):
    (tmp_path / "scene.xml").write_text('<scene version="3.0.0"/>')
    mesh = tmp_path / "target"
    mesh.mkdir()
    (mesh / "tri.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n-1 -1 0\n1 -1 0\n0 1 0\n3 0 1 2\n"
    )
    cfg = dict(
        schema_version=2,
        scene=dict(source="local", id="scene.xml"),
        timeline=dict(steps=3, duration_s=1),
        actors=dict(
            tx=[dict(name="RadarTX", mobility=dict(type="stationary", position_m=[0, 0, 1]))],
            rx=[dict(name="RadarRX", mobility=dict(type="stationary", position_m=[0, 0, 1]))],
            targets=[
                dict(
                    name="Reflector",
                    asset=dict(
                        source="directory", path="target", pattern="tri.ply", switch_meshes=False
                    ),
                    mobility=dict(type="linear", start_m=[0, 5, 1], end_m=[0, 4, 1]),
                )
            ],
        ),
    )
    path = tmp_path / "scenario.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg


def test_native_poses_and_views_preserve_geometry(authored, tmp_path):
    path, _ = authored
    case = load_topology(path)
    tx, rx, target = case["actors"]
    assert tx["members"] == ["RadarTX"]
    assert rx["members"] == ["RadarRX"]
    np.testing.assert_allclose(target["positions"], [[0, 5, 1], [0, 4.5, 1], [0, 4, 1]])
    for mesh, p in zip(target["meshes"], target["positions"]):
        np.testing.assert_allclose((mesh.min((0, 1)) + mesh.max((0, 1))) / 2, p)
    before = [m.copy() for m in target["meshes"]]
    for view in ("3d", "top"):
        output = tmp_path / f"{view}.png"
        render_topology(case, output, view=view, dpi=40)
        assert output.is_file()
        assert case["receipt"]["camera"]["projection"] == "orthographic"
    for a, b in zip(before, target["meshes"]):
        np.testing.assert_array_equal(a, b)
    assert case["receipt"]["camera"]["elevation_deg"] == 90
    render_topology(case, tmp_path / "both.png", dpi=40)
    assert [p["view"] for p in case["receipt"]["camera"]["panels"]] == ["3d", "top"]
    assert case["receipt"]["layout"]["panel_widths"] == [0.65, 0.35]
    json.dumps(case["receipt"])


@pytest.mark.parametrize("label_mode", ["role", "name"])
def test_named_monostatic_topology_preserves_device_identity_and_default_roles(
    authored, label_mode
):
    path, cfg = authored
    cfg["sensing"] = {"enabled": True}
    cfg["generator_summary"] = {"visualization": {"actor_label_mode": label_mode}}
    cfg["actors"]["tx"][0]["name"] = "HarborTX"
    cfg["actors"]["rx"][0]["name"] = "HarborRX"
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    radar = case["actors"][0]
    assert radar["label"] == (
        "Harbor (monostatic radar)" if label_mode == "name" else "Monostatic radar"
    )
    assert radar["members"] == ["HarborTX", "HarborRX"]
    np.testing.assert_array_equal(radar["positions"], np.repeat([[0, 0, 1]], 3, axis=0))


def test_native_animation_uses_bound_world_vertices_and_manifest_root(authored, tmp_path):
    from shared.animated_target import (
        AnimatedTargetBindingContext,
        AnimatedTargetBindingTime,
        load_animated_target,
    )
    from shared.animated_target_workflow import _write_synthetic_asset

    path, cfg = authored
    manifest, _ = _write_synthetic_asset(tmp_path / "animation")
    cfg["actors"]["targets"][0] = dict(
        name="Reflector",
        mobility=dict(type="stationary", position_m=[0, 0, 0]),
        asset=dict(
            source="animated_asset",
            path=str(manifest),
            playback=dict(
                scenario_clock_id="scenario:topology",
                motion_clock_id="motion:topology",
                playback_rate=1.0,
                playback_offset_s=0.0,
            ),
        ),
    )
    path.write_text(yaml.safe_dump(cfg))
    before = {p: p.read_bytes() for p in manifest.parent.rglob("*") if p.is_file()}
    case = load_topology(path)
    actor = case["actors"][-1]
    asset = load_animated_target(manifest)
    binding = asset.bind(
        AnimatedTargetBindingContext(
            "Reflector",
            "scenario:topology",
            "motion:topology",
            None,
            1.0,
            0.0,
            "affine_scenario_to_motion",
        ),
        [AnimatedTargetBindingTime(f"review:time:{t.hex()}", t, t) for t in (0.0, 0.5, 1.0)],
    )
    assert actor["animated"]
    np.testing.assert_allclose(
        actor["positions"], [[0.25, -0.1, 0.05], [0.85, -0.05, 0.05], [1.45, 0.05, 0.15]]
    )
    np.testing.assert_allclose(actor["rotations"][-1, :, 0], [0, 1, 0], atol=1e-14)
    for actual, expected in zip(actor["meshes"], binding.samples):
        np.testing.assert_array_equal(actual, expected.vertices_m[binding.faces])
    recorded = case["receipt"]["actors"][-1]
    assert recorded["position_reference"] == "animated_asset_manifest_root"
    assert [r["mesh_index"] for r in recorded["mesh_samples"]] == [0, 1, 2]
    assert [r["animated_sample"]["pose_sample_id"] for r in recorded["mesh_samples"]] == [
        "pose:neutral_stance",
        "pose:left_reach",
        "pose:asymmetric_settle",
    ]
    render_topology(case, tmp_path / "animated.png", dpi=30)
    assert all(p.read_bytes() == data for p, data in before.items())
    assert not (tmp_path / ".orchav").exists()
    json.dumps(case["receipt"])


@pytest.mark.parametrize(
    "detail,orientation", [(False, True), (True, True), (False, False), (True, False)]
)
def test_motion_detail_and_fixed_target_orientation_are_independent(
    authored, tmp_path, monkeypatch, detail, orientation
):
    import generator.figures.topology as topology
    from shared.scenarios.model import VisualizationConfigModel

    path, cfg = authored
    options = VisualizationConfigModel(
        topology_motion_detail=detail, topology_fixed_target_orientation=orientation
    ).model_dump()
    cfg["generator_summary"] = dict(enabled=True, create=["topology"], visualization=options)
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    target = case["actors"][-1]
    poses = target["positions"].copy()
    arrows = []
    monkeypatch.setattr(
        topology,
        "_pointing_arrow",
        lambda ax, position, *args, **kwargs: arrows.append(position.copy()),
    )
    render_topology(case, tmp_path / "detail.png", view="top", dpi=40)
    assert any(np.array_equal(p, poses[0]) for p in arrows) == orientation
    assert any(np.array_equal(p, case["actors"][0]["positions"][0]) for p in arrows)
    assert len(case["receipt"]["motion_details"]) == int(detail)
    if detail:
        record = case["receipt"]["motion_details"][0]
        assert record["displacement_m"] == pytest.approx(1.0)
        assert record["projection_axes"] == ["X", "Y"]
        np.testing.assert_array_equal(record["end_position_m"], poses[-1])
    np.testing.assert_array_equal(target["positions"], poses)


def test_rectangular_viewport_preserves_projected_aspect():
    import matplotlib.pyplot as plt

    from generator.figures.topology import _fit_topology_view, _TopologyAxes

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes((0, 0, 1, 1), axes_class=_TopologyAxes)
    points = np.asarray([[0, 0, 0], [1, 1, 1]])
    _fit_topology_view(ax, points)
    fig.canvas.draw()
    xscale = ax.bbox.width / np.ptp(ax.viewLim.intervalx)
    yscale = ax.bbox.height / np.ptp(ax.viewLim.intervaly)
    assert xscale == pytest.approx(yscale)
    plt.close(fig)


@pytest.mark.parametrize("sensing", [False, True])
def test_small_motion_detail_is_automatic_only_for_sensing(
    authored, tmp_path, monkeypatch, sensing
):
    import generator.figures.topology as topology
    from shared.scenarios.model import VisualizationConfigModel

    case = load_topology(authored[0])
    case["sensing_enabled"] = sensing
    case["visualization"] = VisualizationConfigModel().model_dump()
    target = case["actors"][-1]
    arrows = []
    monkeypatch.setattr(
        topology, "_pointing_arrow", lambda ax, position, *args, **kwargs: arrows.append(position)
    )
    render_topology(case, tmp_path / "auto.png", view="top", dpi=40)
    assert len(case["receipt"]["motion_details"]) == int(sensing)
    assert any(np.array_equal(p, target["positions"][0]) for p in arrows) == (not sensing)
    assert any(np.array_equal(p, case["actors"][0]["positions"][0]) for p in arrows)


def test_motion_detail_distinguishes_translation_from_articulation_and_long_loops(authored):
    from generator.figures.topology_motion import motion_detail_enabled

    actor = load_topology(authored[0])["actors"][-1]
    assert motion_detail_enabled(actor, "auto", sensing_enabled=True, scene_span=10)
    assert not motion_detail_enabled(
        dict(actor, animated=True), True, sensing_enabled=True, scene_span=10
    )
    assert not motion_detail_enabled(
        dict(actor, physical_path=False), True, sensing_enabled=True, scene_span=10
    )
    loop = dict(actor, positions=np.array([[0, 0, 1], [0, 50, 1], [0, 0.01, 1]]))
    assert not motion_detail_enabled(loop, "auto", sensing_enabled=True, scene_span=60)
    assert motion_detail_enabled(loop, True, sensing_enabled=True, scene_span=60)


def test_overlap_uses_sampled_meshes_for_translation_longer_than_target(authored):
    from generator.figures.topology_motion import (
        motion_detail_enabled,
        overlapping_rigid_snapshots,
    )

    actor = load_topology(authored[0])["actors"][-1]
    # Three true poses span 2.046 m, slightly more than this 2 m mesh. Each
    # neighboring pair overlaps even though the endpoints no longer do.
    positions = actor["positions"].copy()
    actor["positions"] = positions[0] + (positions - positions[0]) * 2.046
    actor["meshes"] = [
        mesh + actor["positions"][sample["index"]] - positions[sample["index"]]
        for mesh, sample in zip(actor["meshes"], actor["mesh_samples"])
    ]
    assert overlapping_rigid_snapshots(actor)
    assert motion_detail_enabled(actor, "auto", sensing_enabled=True, scene_span=10)
    assert not overlapping_rigid_snapshots(dict(actor, animated=True))
    assert not overlapping_rigid_snapshots(dict(actor, stationary=True))
    separated = dict(
        actor, meshes=[mesh + [i * 20, 0, 0] for i, mesh in enumerate(actor["meshes"])]
    )
    assert not overlapping_rigid_snapshots(separated)


def test_overlapping_rigid_meshes_fade_with_matching_legend(authored, tmp_path, monkeypatch):
    import matplotlib.figure

    import generator.figures.topology as topology

    case = load_topology(authored[0])
    actor = case["actors"][-1]
    original_meshes = [mesh.copy() for mesh in actor["meshes"]]
    original_samples = [dict(sample) for sample in actor["mesh_samples"]]
    solid_meshes = []
    shade = topology._shaded

    def capture(mesh, *args, **kwargs):
        solid_meshes.append(mesh.copy())
        return shade(mesh, *args, **kwargs)

    monkeypatch.setattr(topology, "_shaded", capture)
    keys = {}
    savefig = matplotlib.figure.Figure.savefig

    def capture_legend(fig, *args, **kwargs):
        for legend in fig.legends:
            keys.update(
                (text.get_text(), handle.get_alpha())
                for text, handle in zip(legend.get_texts(), legend.legend_handles)
            )
        return savefig(fig, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "savefig", capture_legend)
    render_topology(case, tmp_path / "motion.png", view="top", dpi=30)
    # All three actual positions are filled. The key explains the same opacity
    # progression shown in the scene, without changing the source geometry.
    for original in original_meshes:
        assert any(np.array_equal(mesh, original) for mesh in solid_meshes)
    records = sorted(case["receipt"]["mesh_snapshot_styles"], key=lambda item: item["index"])
    assert [record["style"] for record in records] == ["filled"] * 3
    assert [record["label"] for record in records] == ["Start", "Middle", "End"]
    assert [record["opacity"] for record in records] == [0.16, 0.42, 0.88]
    assert [keys[record["label"]] for record in records] == [
        record["opacity"] for record in records
    ]
    assert [record["time_s"] for record in records] == [0, 0.5, 1]
    for actual, expected in zip(actor["meshes"], original_meshes):
        np.testing.assert_array_equal(actual, expected)
    assert actor["mesh_samples"] == original_samples
    # Returning to a sampled pose retains the latest representation once.
    actor["meshes"][-1] = actor["meshes"][0].copy()
    render_topology(case, tmp_path / "return.png", view="top", dpi=30)
    assert [record["label"] for record in case["receipt"]["mesh_snapshot_styles"]] == [
        "End",
        "Middle",
    ]


def test_transparent_faces_preserve_mesh_surface():
    from generator.figures.topology_motion import transparent_mesh_faces

    square = np.array(
        [[[0, 0, 0], [1, 0, 0], [1, 1, 0]], [[0, 0, 0], [1, 1, 0], [0, 1, 0]]], dtype=float
    )
    original = square.copy()
    polygons = transparent_mesh_faces(square)
    assert len(polygons) == 1 and polygons[0].shape == (4, 3)
    np.testing.assert_array_equal(
        np.unique(polygons[0], axis=0), np.unique(square.reshape(-1, 3), axis=0)
    )
    np.testing.assert_array_equal(square, original)
    # A fold, disconnected triangles, and a concave union must keep both faces.
    folded = square.copy()
    folded[1, 2, 2] = 0.2
    disconnected = square.copy()
    disconnected[1] += [3, 0, 0]
    concave = square.copy()
    concave[1, 2] = [-1, -0.5, 0]
    for mesh in (folded, disconnected, concave):
        assert len(transparent_mesh_faces(mesh)) == 2


def test_tall_rigid_target_uses_side_view_without_changing_poses(authored):
    import matplotlib.pyplot as plt

    from generator.figures.topology_motion import draw_motion_detail

    actor = load_topology(authored[0])["actors"][-1]
    actor["meshes"] = [
        (mesh - actor["positions"][sample["index"]])[:, :, [0, 2, 1]] * [0.2, 0.2, 1]
        + actor["positions"][sample["index"]]
        for mesh, sample in zip(actor["meshes"], actor["mesh_samples"])
    ]
    before = [mesh.copy() for mesh in actor["meshes"]]
    fig = plt.figure()
    record = draw_motion_detail(fig, actor, (0, 0, 1, 1))
    assert record["projection_axes"] == ["Y", "Z"]
    assert record["displacement_m"] == pytest.approx(1)
    for actual, expected in zip(actor["meshes"], before):
        np.testing.assert_array_equal(actual, expected)
    plt.close(fig)


def test_top_view_rulers_match_projected_world_coordinates():
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import proj3d

    from generator.figures.topology import _fit_topology_view, _top_view_axes, _TopologyAxes

    fig = plt.figure(figsize=(10, 4))
    ax = fig.add_axes((0.2, 0.2, 0.7, 0.7), axes_class=_TopologyAxes)
    low, high = np.array([-40, -15, 0]), np.array([60, 35, 12])
    ax.set(xlim=(low[0], high[0]), ylim=(low[1], high[1]), zlim=(low[2], high[2]))
    ax.set_box_aspect(high - low)
    ax.set_proj_type("ortho")
    ax.view_init(elev=90, azim=-90)
    _fit_topology_view(ax, np.array([low, high]))
    _top_view_axes(ax, low, high)
    fig.canvas.draw()
    ruler = fig.axes[-1]
    for point in (low, high, np.array([7, 9, 6])):
        projected = proj3d.proj_transform(*point, ax.get_proj())[:2]
        np.testing.assert_allclose(
            ruler.transData.transform(point[:2]), ax.transData.transform(projected), atol=1e-8
        )
    assert ruler.get_xlabel() == "X (m)"
    assert ruler.get_ylabel() == "Y (m)"
    plt.close(fig)


def test_material_key_preserves_native_identity_and_only_background_loses_edges(authored):
    path, _ = authored
    (path.parent / "scene.xml").write_text(
        '<scene version="3.0.0"><bsdf type="itu-radio-material" id="mat-concrete">'
        '<string name="type" value="concrete"/></bsdf>'
        '<shape type="ply" id="ground"><string name="filename" value="target/tri.ply"/>'
        '<ref id="mat-concrete"/></shape>'
        '<shape type="ply" id="panel"><string name="filename" value="target/tri.ply"/>'
        '<ref id="mat-concrete"/></shape></scene>'
    )
    ground, panel = scene_file(path.parent / "scene.xml")[0]
    assert ground["material_id"] == panel["material_id"] == "mat-concrete"
    assert ground["material_label"] == "Ground (concrete)"
    assert panel["material_label"] == "Concrete"
    assert len(ground["edges"]) == 0
    # An actual object at z=0 keeps its edges; altitude alone is not a filter.
    assert len(panel["edges"]) == 3
    case = load_topology(path)
    render_topology(case, path.parent / "materials.png", dpi=40)
    assert {entry["label"] for entry in case["receipt"]["material_legend"]} == {
        "Concrete",
        "Ground (concrete)",
    }


def test_orientation_samples_use_entire_rotation_series_including_roll():
    fixed = np.repeat(rotation_matrix(Quaternion.from_euler_deg(20))[None], 5, axis=0)
    assert orientation_indices(fixed) == [0]
    for rotations in (
        [Quaternion.from_euler_deg(a) for a in [0, 90, 180, 270, 360]],
        [Quaternion.from_euler_deg(0, 0, a) for a in [0, 20, 40, 60, 80]],
    ):
        assert orientation_indices(np.array([rotation_matrix(q) for q in rotations])) == [0, 2, 4]


def test_stationary_marker_uses_full_positions_not_matching_endpoints():
    assert is_stationary(np.array([[1, 2, 3], [1, 2, 3]]))
    assert not is_stationary(np.array([[1, 2, 3], [2, 3, 4], [1, 2, 3]]))
    assert not is_stationary(np.array([[1, 2, 3], [1, 2, 4]]))


def test_stationary_look_at_and_moving_orientation_use_matching_native_samples(authored):
    path, cfg = authored
    cfg["actors"]["tx"][0]["orientation"] = {"type": "look_at", "actor": "Reflector"}
    cfg["actors"]["targets"][0]["mobility"] = {
        "type": "linear",
        "start_m": [1, 5, 1],
        "end_m": [3, 4, 1],
    }
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    tx, rx, target = case["actors"]
    assert tx["stationary"] and rx["stationary"]
    assert not target["stationary"]
    assert case["receipt"]["actors"][0]["stationary"]
    assert tx["orientation_indices"] == [0, 1, 2]
    assert rx["orientation_indices"] == [0]
    delta = target["positions"] - tx["positions"]
    expected = delta / np.linalg.norm(delta, axis=1, keepdims=True)
    np.testing.assert_allclose(tx["rotations"][:, :, 0], expected, atol=1e-12)
    assert [s["index"] for s in case["receipt"]["actors"][0]["orientation_samples"]] == [0, 1, 2]


@pytest.mark.parametrize(
    "switch,interval,end_behavior,expected_end",
    [
        (True, None, "loop", 2),
        (True, 1.0, "loop", 0),
        (True, 1.0, "hold_last", 2),
        (False, None, "loop", 1),
    ],
)
def test_animated_mesh_samples_and_asset_front(
    authored, switch, interval, end_behavior, expected_end
):
    path, cfg = authored
    directory = path.parent / "target"
    original = (directory / "tri.ply").read_text()
    for i in range(3):
        (directory / f"sequence_{i}.ply").write_text(
            original.replace("-1 -1 0", f"{-1 - i} -1 0").replace("1 -1 0\n0", f"{1 + i} -1 0\n0")
        )
    (directory / "target_metadata.json").write_text(json.dumps({"front": {"yaw_offset_deg": 180}}))
    target = cfg["actors"]["targets"][0]
    target["asset"].update(
        pattern="sequence_*.ply",
        switch_meshes=switch,
        start_index=1,
        frame_stride=2,
        mesh_end_behavior=end_behavior,
    )
    target["mobility"].update(start_m=[1, 5, 1], end_m=[3, 4, 1])
    target["orientation"] = dict(type="align_motion", allow_pitch=False)
    if interval is not None:
        cfg["raytracing"] = dict(mesh_update_interval_s=interval)
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    actor = case["actors"][-1]
    assert [s["mesh_index"] for s in actor["mesh_samples"]][::2] == [1, expected_end]
    assert [s["index"] for s in actor["mesh_samples"]] == [0, 1, 2]
    assert [s["opacity"] for s in actor["mesh_samples"]] == [1, 0.28, 1]
    assert actor["animated"] == switch
    # Native mesh rotation maps its declared -X front onto the travel direction.
    forward = np.array([2, -1, 0]) / np.sqrt(5)
    np.testing.assert_allclose(actor["rotations"][0] @ [-1, 0, 0], forward, atol=1e-12)
    np.testing.assert_allclose(actor["pointing_rotations"][0, :, 0], forward, atol=1e-12)
    for sample, mesh in zip(actor["mesh_samples"], actor["meshes"]):
        assert np.linalg.norm(mesh[0, 1] - mesh[0, 0]) == pytest.approx(
            2 * (1 + sample["mesh_index"])
        )
    receipt = case["receipt"]["actors"][-1]
    assert receipt["mesh_samples"][-1]["source"].endswith(f"sequence_{expected_end}.ply")


def test_colocated_communications_nodes_are_not_renamed(authored):
    path, cfg = authored
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    assert [a["label"] for a in case["actors"][:2]] == ["RadarTX", "RadarRX"]


def test_missing_mesh_is_not_omitted(authored):
    path, cfg = authored
    cfg["actors"]["targets"][0]["asset"]["pattern"] = "missing.ply"
    path.write_text(yaml.safe_dump(cfg))
    with pytest.raises(FileNotFoundError, match="No target mesh"):
        load_topology(path)


def test_topology_summary_uses_runtime_poses_and_staging(authored, tmp_path):
    from dataclasses import replace
    from types import SimpleNamespace

    from generator import build_simulation_config
    from generator.core.scenario_actors.runtime import prepare_actor_runtime
    from generator.figures.summary.generator import maybe_generate_generator_summary
    from shared.scenarios import load_scenario_configuration

    path, cfg = authored
    cfg["generator_summary"] = dict(
        enabled=True,
        create=["topology"],
        output=dict(format="svg"),
        visualization=dict(topology_motion_detail=True, topology_fixed_target_orientation=False),
    )
    path.write_text(yaml.safe_dump(cfg))
    context = load_scenario_configuration(path.parent)
    # A scripted override must be drawn instead of silently reloading authored YAML.
    runtime = prepare_actor_runtime(context)
    tx = replace(runtime.transmitters[0], name="Scripted transmitter")
    output = tmp_path / "staging"
    result = maybe_generate_generator_summary(
        [tx],
        list(runtime.receivers),
        [SimpleNamespace(config=t) for t in runtime.targets],
        build_simulation_config(context),
        context,
        output_root=output,
        strict=True,
    )
    assert set(result) == {output / "topology.svg", output / "topology.json"}
    receipt = json.loads((output / "topology.json").read_text())
    assert receipt["actors"][0]["name"] == "Scripted transmitter"
    assert receipt["actor_source"] == "prepared_runtime"
    assert receipt["resolved_yaml_sha256"] == context.summary_yaml_hash
    assert len(receipt["camera"]["panels"]) == 2
    assert receipt["topology_options"] == dict(
        pose_samples=3, motion_detail=True, fixed_target_orientation=False, actor_label_mode="role"
    )
    assert receipt["motion_details"][0]["displacement_m"] == pytest.approx(1.0)
    assert not (path.parent / "summary").exists()


def test_dense_stationary_nodes_share_a_key_without_losing_positions(authored):
    from generator.figures.topology import _display_actors

    case = load_topology(authored[0])
    base = case["actors"][1]
    actors = [
        dict(base, name=f"RX{i}", label=f"RX{i}", positions=base["positions"] + [i, 0, 0])
        for i in range(30)
    ]
    display, keys, numbers, grouped = _display_actors(actors)
    assert len(display) == 30 and keys == [0] and numbers == {0: 1}
    assert len(grouped) == 30
    for before, after in zip(actors, display):
        np.testing.assert_array_equal(before["positions"], after["positions"])
    assert actors[0]["label"] == "RX0"
    assert display[0]["label"] == "30 stationary receivers"


def test_missing_external_scene_never_triggers_acquisition(authored, monkeypatch):
    from types import SimpleNamespace

    import generator.figures.topology as module

    path, cfg = authored
    cfg["scene"] = dict(source="osm", id="missing.xml", osm=dict(center=[48.87, 2.33], size_m=100))
    path.write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(
        module, "load_scenario", lambda _: SimpleNamespace(model_dump=lambda **_: cfg)
    )
    monkeypatch.setattr(
        module, "parse_scene_config", lambda *a: pytest.fail("No scene acquisition")
    )
    with pytest.raises(FileNotFoundError, match="prepare the authored scene"):
        load_topology(path)


@pytest.mark.parametrize(
    "object_type,model_type,dimensions",
    [("human", 1, [0.5, 0.5, 1.75]), ("uav-large-size", 2, [1.6, 1.5, 0.7])],
)
def test_authored_native_rcs_topology_uses_proxy_and_prepared_motion(
    authored, object_type, model_type, dimensions
):
    path, cfg = authored
    target = cfg["actors"]["targets"][0]
    target.pop("asset")
    target.update(
        representation="tr38901",
        geometry={"type": "box"},
        tr38901={"object_type": object_type, "model_type": model_type},
    )
    path.write_text(yaml.safe_dump(cfg))
    case = load_topology(path)
    actor = case["actors"][-1]
    np.testing.assert_allclose(actor["positions"], [[0, 5, 1], [0, 4.5, 1], [0, 4, 1]])
    for mesh, position in zip(actor["meshes"], actor["positions"]):
        np.testing.assert_allclose(mesh.max((0, 1)) - mesh.min((0, 1)), dimensions)
        np.testing.assert_allclose((mesh.max((0, 1)) + mesh.min((0, 1))) / 2, position)
    assert case["receipt"]["actors"][-1]["representation"] == "tr38901"


def test_external_native_rcs_runtime_summary_hashes_installed_sources(authored, tmp_path):
    import hashlib
    from types import SimpleNamespace

    from generator import build_simulation_config
    from generator.core.scenario_actors.runtime import prepare_actor_runtime
    from generator.core.target import representation
    from generator.figures.summary.generator import maybe_generate_generator_summary
    from shared.scenarios import load_scenario_configuration

    path, cfg = authored
    target = cfg["actors"]["targets"][0]
    target.pop("asset")
    target.update(
        representation="tr38901",
        geometry={"type": "box"},
        tr38901={"object_type": "human", "model_type": 1},
    )
    cfg["generator_summary"] = dict(enabled=True, create=["topology"], output=dict(format="svg"))
    path.write_text(yaml.safe_dump(cfg))
    context = load_scenario_configuration(path.parent)
    assert not (Path(context.project_root) / "generator").exists()
    runtime = prepare_actor_runtime(context)
    output = tmp_path / "summary"
    result = maybe_generate_generator_summary(
        list(runtime.transmitters),
        list(runtime.receivers),
        [SimpleNamespace(config=t) for t in runtime.targets],
        build_simulation_config(context),
        context,
        output_root=output,
        strict=True,
    )
    assert set(result) == {output / "topology.svg", output / "topology.json"}
    receipt = json.loads((output / "topology.json").read_text())
    source = Path(representation.__file__).resolve()
    assert receipt["source_files"][str(source)] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert receipt["actors"][-1]["representation"] == "tr38901"
    assert receipt["actor_source"] == "prepared_runtime"


@pytest.mark.parametrize("value", [True, False, 0, 4, 6, 3.0, 5.0, "5", None, []])
def test_topology_pose_sample_option_rejects_invalid_configuration(value):
    from pydantic import ValidationError

    from shared.scenarios.model import VisualizationConfigModel

    with pytest.raises(ValidationError, match="topology_pose_samples"):
        VisualizationConfigModel(topology_pose_samples=value)


@pytest.mark.parametrize(
    "steps,three,five",
    [
        (1, [0], [0]),
        (2, [0, 1], [0, 1]),
        (3, [0, 1, 2], [0, 1, 2]),
        (4, [0, 2, 3], [0, 1, 2, 3]),
        (5, [0, 2, 4], [0, 1, 2, 3, 4]),
        (8, [0, 4, 7], [0, 2, 4, 6, 7]),
        (9, [0, 4, 8], [0, 2, 4, 6, 8]),
        (8192, [0, 4096, 8191], [0, 2048, 4096, 6144, 8191]),
    ],
)
def test_topology_pose_indices_preserve_native_samples_and_deduplicate(steps, three, five):
    from generator.figures.topology import _pose_sample_indices

    assert _pose_sample_indices(steps, 3) == three
    assert _pose_sample_indices(steps, 5) == five


@pytest.fixture
def animated_authored(authored, tmp_path):
    from shared.animated_target_workflow import _write_synthetic_asset

    path, cfg = authored
    manifest, _ = _write_synthetic_asset(tmp_path / "animation")
    cfg["timeline"]["steps"] = 9
    cfg["generator_summary"] = dict(visualization=dict(topology_pose_samples=5))
    cfg["actors"]["targets"][0] = dict(
        name="Reflector",
        mobility=dict(type="stationary", position_m=[0, 0, 0]),
        asset=dict(
            source="animated_asset",
            path=str(manifest),
            playback=dict(
                scenario_clock_id="scenario:topology",
                motion_clock_id="motion:topology",
                playback_rate=1.0,
                playback_offset_s=0.0,
            ),
        ),
    )
    path.write_text(yaml.safe_dump(cfg))
    return path, cfg, manifest


@pytest.mark.parametrize("stationary_root", [False, True])
def test_five_native_poses_and_gallery_use_each_authenticated_world_mesh(
    animated_authored, stationary_root, monkeypatch, tmp_path
):
    import matplotlib.pyplot as plt

    from generator.figures import topology
    from shared.animated_target import (
        AnimatedTargetBindingContext,
        AnimatedTargetBindingTime,
        animated_target_revision_sha256,
        load_animated_target,
    )

    path, _, manifest = animated_authored
    if stationary_root:
        raw = json.loads(manifest.read_text())
        for sample in raw["motion"]["root"]["samples"]:
            sample.update(translation=[0, 0, 0], rotation_xyzw=[0, 0, 0, 1])
        raw["asset"]["revision_sha256"] = animated_target_revision_sha256(raw)
        manifest.write_text(json.dumps(raw))
    case = load_topology(path)
    actor = case["actors"][-1]
    assert actor["stationary"] == stationary_root
    indices, times = [0, 2, 4, 6, 8], [0.0, 0.25, 0.5, 0.75, 1.0]
    assert [s["index"] for s in actor["mesh_samples"]] == indices
    assert [s["time_s"] for s in actor["mesh_samples"]] == times
    asset = load_animated_target(manifest)
    binding = asset.bind(
        AnimatedTargetBindingContext(
            "Reflector",
            "scenario:topology",
            "motion:topology",
            None,
            1.0,
            0.0,
            "affine_scenario_to_motion",
        ),
        [AnimatedTargetBindingTime(f"independent:{t.hex()}", t, t) for t in times],
    )
    for actual, expected in zip(actor["meshes"], binding.samples):
        np.testing.assert_array_equal(actual, expected.vertices_m[binding.faces])
    # Step interpolation legitimately repeats source poses at two sample times.
    assert [s["animated_sample"]["pose_sample_id"] for s in actor["mesh_samples"]] == [
        "pose:neutral_stance",
        "pose:neutral_stance",
        "pose:left_reach",
        "pose:left_reach",
        "pose:asymmetric_settle",
    ]
    inputs = []
    constructor = topology.Poly3DCollection

    def capture(vertices, *args, **kwargs):
        inputs.append(np.array(vertices, copy=True))
        return constructor(vertices, *args, **kwargs)

    monkeypatch.setattr(topology, "Poly3DCollection", capture)
    fig = plt.figure(figsize=(9, 3))
    topology._target_gallery(fig, [actor], 0.05)
    fig.canvas.draw()
    assert len(inputs) == 5
    for actual, sample in zip(inputs, binding.samples):
        np.testing.assert_array_equal(
            actual,
            sample.vertices_m[binding.faces] - sample.resolved_root_affine_4x4[:3, 3],
        )
    assert [text.get_text() for text in fig.texts[:2]] == ["Target motion", "1  Reflector"]
    assert [text.get_text() for text in fig.texts[2:]] == [
        "First · 0 s",
        "Quarter · 0.25 s",
        "Middle · 0.5 s",
        "Three-quarter · 0.75 s",
        "Final · 1 s",
    ]
    plt.close(fig)
    render_topology(case, tmp_path / "five-native.png", dpi=30)
    assert case["receipt"]["topology_options"]["pose_samples"] == 5
    recorded = case["receipt"]["actors"][-1]["mesh_samples"]
    assert [s["index"] for s in recorded] == indices
    assert [s["time_s"] for s in recorded] == times


@pytest.mark.parametrize("switch", [False, True])
@pytest.mark.parametrize("stationary", [False, True])
def test_five_pose_option_preserves_directory_mesh_timing_and_rigid_behavior(
    authored, switch, stationary
):
    path, cfg = authored
    directory = path.parent / "target"
    original = (directory / "tri.ply").read_text()
    for index in range(9):
        (directory / f"sequence_{index}.ply").write_text(
            original.replace("-1 -1 0", f"{-1 - index} -1 0").replace(
                "1 -1 0\n0", f"{1 + index} -1 0\n0"
            )
        )
    cfg["timeline"]["steps"] = 9
    cfg["generator_summary"] = dict(visualization=dict(topology_pose_samples=5))
    target = cfg["actors"]["targets"][0]
    target["asset"].update(
        pattern="sequence_*.ply",
        switch_meshes=switch,
        start_index=1,
        frame_stride=2,
    )
    if stationary:
        target["mobility"] = dict(type="stationary", position_m=[0, 5, 1])
    path.write_text(yaml.safe_dump(cfg))
    actor = load_topology(path)["actors"][-1]
    indices = [0, 2, 4, 6, 8] if switch else [0] if stationary else [0, 4, 8]
    expected_meshes = [1, 5, 0, 4, 8] if switch else [1] * len(indices)
    assert [s["index"] for s in actor["mesh_samples"]] == indices
    assert [s["mesh_index"] for s in actor["mesh_samples"]] == expected_meshes
    for mesh, index in zip(actor["meshes"], expected_meshes):
        assert np.linalg.norm(mesh[0, 1] - mesh[0, 0]) == pytest.approx(2 * (index + 1))


def test_five_pose_galleries_use_readable_full_width_rows(animated_authored):
    from copy import deepcopy

    import matplotlib.pyplot as plt

    from generator.figures import topology

    case = load_topology(animated_authored[0])
    first = case["actors"][-1]
    second = deepcopy(first)
    second.update(name="Second", label="Second")
    fig = plt.figure(figsize=(9, 5))
    upper = topology._target_gallery(fig, [first, second], 0.05)
    fig.canvas.draw()
    assert len(fig.axes) == 10
    assert upper == pytest.approx(0.05 + (2 * 1.8 + 0.52) / 5)
    assert len({round(ax.get_position().y0, 8) for ax in fig.axes}) == 2
    assert all(ax.get_position().width > 0.13 for ax in fig.axes)
    for ax in fig.axes:
        bounds = ax.get_position()
        assert bounds.x0 >= 0 and bounds.x1 <= 1
        assert bounds.y0 >= 0 and bounds.y1 <= 1
    plt.close(fig)


def test_default_three_pose_geometry_captions_and_pixels_are_unchanged(animated_authored, tmp_path):
    from PIL import Image

    path, cfg, _ = animated_authored
    cfg.pop("generator_summary")
    path.write_text(yaml.safe_dump(cfg))
    default = load_topology(path)
    explicit = load_topology(path, pose_samples=3)
    for actual, expected in zip(default["actors"][-1]["meshes"], explicit["actors"][-1]["meshes"]):
        np.testing.assert_array_equal(actual, expected)
    assert [s["index"] for s in default["actors"][-1]["mesh_samples"]] == [0, 4, 8]
    for name, case in [("default", default), ("three", explicit)]:
        render_topology(case, tmp_path / f"{name}.png", dpi=30)
    np.testing.assert_array_equal(
        np.asarray(Image.open(tmp_path / "default.png")),
        np.asarray(Image.open(tmp_path / "three.png")),
    )


def test_topology_pose_cli_overrides_yaml_before_sampling_and_records_receipt(
    animated_authored, tmp_path
):
    from generator.figures.topology import main

    path, cfg, _ = animated_authored
    cfg["generator_summary"]["visualization"]["topology_pose_samples"] = 3
    path.write_text(yaml.safe_dump(cfg))
    output = tmp_path / "cli.png"
    assert (
        main(
            [
                "--scenario",
                str(path),
                "--output",
                str(output),
                "--pose-samples",
                "5",
                "--view",
                "top",
                "--dpi",
                "30",
            ]
        )
        == 0
    )
    receipt = json.loads(output.with_suffix(".json").read_text())
    assert receipt["topology_options"]["pose_samples"] == 5
    assert [s["index"] for s in receipt["actors"][-1]["mesh_samples"]] == [0, 2, 4, 6, 8]
    assert receipt["command"][-2:] == ["--pose-samples", "5"]
    assert yaml.safe_load(path.read_text()) == cfg
    with pytest.raises(SystemExit) as invalid:
        main(["--scenario", str(path), "--pose-samples", "4"])
    assert invalid.value.code == 2
    with pytest.raises(ValueError, match="integer 3 or 5"):
        load_topology(path, pose_samples=5.0)


def test_native_generator_summary_honors_five_pose_yaml(animated_authored, tmp_path):
    from types import SimpleNamespace

    from generator import build_simulation_config
    from generator.core.scenario_actors.runtime import prepare_actor_runtime
    from generator.figures.summary.generator import maybe_generate_generator_summary
    from shared.scenarios import load_scenario_configuration

    path, cfg, _ = animated_authored
    cfg["generator_summary"].update(enabled=True, create=["topology"])
    path.write_text(yaml.safe_dump(cfg))
    context = load_scenario_configuration(path.parent)
    runtime = prepare_actor_runtime(context)
    output = tmp_path / "native-summary"
    result = maybe_generate_generator_summary(
        list(runtime.transmitters),
        list(runtime.receivers),
        [SimpleNamespace(config=t) for t in runtime.targets],
        build_simulation_config(context),
        context,
        output_root=output,
        strict=True,
    )
    assert output / "topology.png" in result
    receipt = json.loads((output / "topology.json").read_text())
    assert receipt["actor_source"] == "prepared_runtime"
    assert receipt["topology_options"]["pose_samples"] == 5
    samples = receipt["actors"][-1]["mesh_samples"]
    assert [s["index"] for s in samples] == [0, 2, 4, 6, 8]
    assert [s["time_s"] for s in samples] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert all("animated_sample" in s for s in samples)
