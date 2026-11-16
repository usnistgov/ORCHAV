"""Native representation selection and editable YAML round-trip contracts."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.scenarios.actors import (
    ConstantRCSSpec,
    NativeBoxGeometrySpec,
    NativeMeshGeometrySpec,
    TR38901RCSSpec,
)
from visualizer.src.authoring.assets import PreviewAssetCompiler, prepared_actor_pose
from visualizer.src.authoring.compiler import canonical_scenario_mapping
from visualizer.src.authoring.domain import (
    AuthoringActor,
    AuthoringScenario,
    SceneReference,
    TargetAsset,
)
from visualizer.src.authoring.persistence import scenario_from_mapping
from visualizer.src.authoring.target_representation_editor import TargetRepresentationEditor


def native_actor(representation="constant_rcs"):
    return AuthoringActor.create("target", "test-target", position=(1, 2, 3)).with_changes(
        representation=representation,
        geometry=NativeBoxGeometrySpec(),
        constant_rcs=(
            ConstantRCSSpec(sigma_m2=0.1, xpr_db=3) if representation == "constant_rcs" else None
        ),
        tr38901=(
            TR38901RCSSpec(object_type="human", model_type=1)
            if representation == "tr38901"
            else None
        ),
    )


@pytest.mark.parametrize("representation", ["constant_rcs", "tr38901"])
def test_native_schema_round_trip_keeps_pose_and_single_rf_model(tmp_path, representation):
    actor = native_actor(representation)
    scenario = AuthoringScenario(
        scene=SceneReference("sionna", "simple_street_canyon"), actors=(actor,)
    )
    mapping = canonical_scenario_mapping(scenario)
    record = mapping["actors"]["targets"][0]
    assert "asset" not in record
    assert set(record).intersection({"constant_rcs", "tr38901"}) == {representation}
    loaded = scenario_from_mapping(mapping, source_directory=tmp_path)
    actual = loaded.actors[0]
    assert actual.target is None
    assert actual.mobility == actor.mobility
    assert actual.orientation == actor.orientation
    assert actual.target_representation_mapping() == actor.target_representation_mapping()


def test_switch_from_imported_native_to_mesh_clears_rcs_blocks(tmp_path):
    actor = native_actor()
    original = AuthoringScenario(
        scene=SceneReference("sionna", "simple_street_canyon"), actors=(actor,)
    )
    loaded = scenario_from_mapping(canonical_scenario_mapping(original), source_directory=tmp_path)
    current = loaded.actors[0]
    mesh = current.with_changes(
        representation="mesh_material",
        target=TargetAsset.from_catalog_id("car"),
        geometry=None,
        constant_rcs=None,
    )
    record = canonical_scenario_mapping(replace(loaded, actors=(mesh,)))["actors"]["targets"][0]
    assert record["asset"]["id"] == "car"
    assert "constant_rcs" not in record
    assert "geometry" not in record


def test_editor_offers_all_stock_presets_and_selects_valid_models(qtbot):
    editor = TargetRepresentationEditor()
    qtbot.addWidget(editor)
    editor.representation.setCurrentIndex(editor.representation.findData("tr38901"))
    assert editor.object_type.count() == 7
    for index in range(editor.object_type.count()):
        editor.object_type.setCurrentIndex(index)
        changes = editor.actor_changes()
        model = changes["tr38901"]
        assert model.model_type == (1 if model.object_type == "uav-small-size" else 2)
        assert not any((model.random_sigma_s, model.random_xpr, model.random_phases))
        assert changes["target"] is None
        assert changes["constant_rcs"] is None


def test_editor_load_reset_and_mesh_proxy_round_trip(qtbot):
    editor = TargetRepresentationEditor()
    qtbot.addWidget(editor)
    actor = native_actor().with_changes(
        geometry=NativeMeshGeometrySpec(path="targets/car.obj", scale=2)
    )
    editor.set_actor(actor)
    assert editor.actor_changes()["geometry"] == actor.geometry
    assert editor.actor_changes()["constant_rcs"] == actor.constant_rcs
    editor.sigma.setValue(9)
    editor.set_actor(actor)
    assert editor.sigma.value() == 0.1
    editor.sigma.setValue(0)
    assert editor.actor_changes()["constant_rcs"].sigma_m2 == 0


def test_native_preview_uses_shared_points_and_prepared_motion(tmp_path):
    actor = native_actor("tr38901").with_changes(
        tr38901=TR38901RCSSpec(object_type="vehicle-multi-sp")
    )
    samples = SimpleNamespace(
        positions=((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)),
        orientations=((0.0, 0.0, 0.0), (90.0, 0.0, 0.0)),
    )
    frames = PreviewAssetCompiler(tmp_path).build_target_assets(actor, (), samples)
    assert len(frames) == 2
    assert frames[0].scattering_point_ids == (0, 1, 2, 3, 4)
    assert frames[0].metadata_source == "preset_preview"
    np.testing.assert_array_equal(frames[0].scattering_points_local_m[0], [0, 1, 0])
    world_point = prepared_actor_pose(samples.positions[1], samples.orientations[1]) @ np.array(
        [*frames[1].scattering_points_local_m[0], 1]
    )
    np.testing.assert_allclose(world_point[:3], [3, 5, 6])
    assert frames[0].representation == "tr38901"


def test_native_builder_proxy_keeps_transparent_backend_depth_policy(qtbot):
    from uuid import uuid4

    import pygfx
    from PySide6.QtWidgets import QWidget

    from tests.visualizer.unit.test_scenario_authoring_viewport import (
        _FakeRenderer,
        _triangle_payload,
    )
    from visualizer.src.authoring.viewport import PygfxScenarioAuthoringViewportPort
    from visualizer.src.authoring.viewport_port import (
        ActorOverlaySnapshot,
        OverlaySnapshot,
        TargetOverlayAsset,
        stable_renderer_id,
    )
    from visualizer.src.renderers.pygfx.geometry import PygfxGeometryMixin
    from visualizer.src.renderers.pygfx.materials import PygfxMaterialMixin
    from visualizer.src.types.render_payloads import MaterialPayload

    host = QWidget()
    qtbot.addWidget(host)
    port = PygfxScenarioAuthoringViewportPort(
        host, SimpleNamespace(), renderer_factory=_FakeRenderer
    )
    document_id, actor_id = uuid4(), uuid4()
    asset = TargetOverlayAsset(
        cache_key="native",
        payload=_triangle_payload(),
        material=MaterialPayload(base_color=(0.2, 0.4, 1, 0.25)),
        representation="tr38901",
        scattering_points_local_m=((0, 0, 0),),
        scattering_point_ids=(0,),
    )
    try:
        port.reconcile(
            OverlaySnapshot(
                document_id=document_id,
                revision=1,
                work_plane_visible=False,
                actors=(
                    ActorOverlaySnapshot(
                        actor_id=actor_id,
                        role="target",
                        name="Car",
                        positions=((0, 0, 0),),
                        current_position=(0, 0, 0),
                        target_asset=asset,
                    ),
                ),
            )
        )
        objects = port.renderer_objects()
        proxy = objects[stable_renderer_id(document_id, actor_id, "target")]
        material = pygfx.MeshBasicMaterial(color=(0.2, 0.4, 1, 0.25))
        backend = SimpleNamespace(
            _kinds={proxy.id: "mesh"},
            _objects={proxy.id: SimpleNamespace(material=material)},
            _material_requests_transparency=PygfxMaterialMixin._material_requests_transparency,
        )
        assert PygfxMaterialMixin._apply_material_alpha_state(
            backend, proxy.id, material, proxy.material_payload
        )
        assert not material.depth_write
        PygfxGeometryMixin._sync_render_object_metadata(backend, proxy)
        assert not material.depth_write
        assert material.alpha_mode == "weighted_blend"
        point = objects[stable_renderer_id(document_id, actor_id, "scattering_point_0")]
        assert point.visible and point.material_payload.base_color[3] == 1.0
        assert point.metadata["pickable"] is False
    finally:
        port.close()


@pytest.mark.parametrize("case", ("08_human_model1", "06_vehicle_multi"))
def test_actual_native_example_copies_saves_reopens_and_compiles(tmp_path, case):
    from pathlib import Path

    from shared.scenarios import load_scenario_yaml
    from visualizer.src.authoring.compiler import ScenarioCompiler
    from visualizer.src.authoring.persistence import (
        LoadDisposition,
        create_scenario_copy,
        load_for_authoring,
        save_document,
    )

    root = Path(__file__).resolve().parents[3]
    source = root / "scenarios/sensing/foundations/RCS" / case / "scenario.yaml"
    original = source.read_bytes()
    document = create_scenario_copy(source, tmp_path / "editable_copy")
    compiler = ScenarioCompiler(root)
    saved = save_document(document, compiler=compiler)
    assert saved.is_file()
    mapping = load_scenario_yaml(saved)
    assert (saved.parent / mapping["scene"]["id"]).is_file()
    assert not (saved.parent / "frames").exists()
    reopened = load_for_authoring(saved)
    assert reopened.disposition is LoadDisposition.OWNED_EDITABLE
    assert reopened.document is not None
    compiled = compiler.compile(reopened.document, scenario_directory=saved.parent)
    assert compiled.valid, compiled.issues
    assert [
        (actor.name, actor.role, actor.mobility, actor.orientation)
        for actor in reopened.scenario.actors
    ] == [
        (actor.name, actor.role, actor.mobility, actor.orientation)
        for actor in document.scenario.actors
    ]
    assert [actor.target_representation_mapping() for actor in reopened.scenario.actors] == [
        actor.target_representation_mapping() for actor in document.scenario.actors
    ]
    assert source.read_bytes() == original
