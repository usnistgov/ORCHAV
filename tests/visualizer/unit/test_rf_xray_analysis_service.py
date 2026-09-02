from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from shared.scenarios.actors import TimelineSpec
from visualizer.src.io.frame_sources import FileSource, make_frame_source
from visualizer.src.io.scenario_config import Scenario
from visualizer.src.metrics.mpc_canon import CanonicalStepData
from visualizer.src.pipeline.core import FrameRenderPacket, ViewModel
from visualizer.src.services import rf_xray_analysis_service as rf_xray_module
from visualizer.src.services.rf_xray_analysis_service import (
    RFXRAY_MODE_MATERIAL_MAP,
    RFXRAY_MODE_MATERIAL_PROPERTIES,
    RFXRAY_MODE_MPC_USAGE,
    RFXRayAnalysisService,
)
from visualizer.src.state import MpcVisibility, create_initial_state


def _canonical_material_frame() -> CanonicalStepData:
    points = np.array(
        [
            [0.0, 0.0, 0.0],  # path 0 TX
            [1.0, 0.0, 0.0],  # path 0 concrete bounce
            [2.0, 0.0, 0.0],  # path 0 RX
            [0.0, 1.0, 0.0],  # path 1 TX
            [1.0, 1.0, 0.0],  # path 1 glass bounce
            [2.0, 1.0, 0.0],  # path 1 RX
        ],
        dtype=np.float32,
    )
    lines = np.array([[0, 1], [1, 2], [3, 4], [4, 5]], dtype=np.int32)
    material_ids = np.array([0, 1, 0, 0, 2, 0], dtype=np.int16)
    return CanonicalStepData(
        points=points,
        lines=lines,
        order=np.array([0, 1, 0, 0, 1, 0], dtype=np.uint8),
        itype=np.array([0, 1, 0, 0, 2, 0], dtype=np.uint8),
        delay=np.zeros(6, dtype=np.float32),
        loss=np.zeros(6, dtype=np.float32),
        path_id=np.array([0, 0, 0, 1, 1, 1], dtype=np.int32),
        path_start_indices=np.array([0, 3], dtype=np.int32),
        path_orders=np.array([1, 1], dtype=np.uint8),
        path_losses=np.array([40.0, 50.0], dtype=np.float32),
        segment_start_indices=lines[:, 0].copy(),
        segment_end_indices=lines[:, 1].copy(),
        segment_order=np.array([0, 1, 0, 1], dtype=np.uint8),
        segment_itype=np.array([1, 1, 2, 2], dtype=np.uint8),
        segment_path_id=np.array([0, 0, 1, 1], dtype=np.int32),
        segment_material_ids=np.array([0, 1, 0, 2], dtype=np.int16),
        material_ids=material_ids,
        material_id_to_name={0: "", 1: "mat-itu_concrete", 2: "mat-itu_glass"},
        material_id_to_itu={0: "", 1: "concrete", 2: "glass"},
        material_id_to_bare={0: "", 1: "concrete", 2: "glass"},
    )


def _frame_packet(canon: CanonicalStepData, segment_mask: np.ndarray) -> FrameRenderPacket:
    view_model = ViewModel(
        tx_positions=np.empty((0, 3), dtype=np.float32),
        rx_positions=np.empty((0, 3), dtype=np.float32),
        tx_orientations=np.empty((0, 3), dtype=np.float32),
        rx_orientations=np.empty((0, 3), dtype=np.float32),
        mpc_points=canon.points,
        mpc_lines=canon.lines,
        mpc_colors=np.ones((len(canon.lines), 3), dtype=np.float32),
        colorbar=None,
        stats_text="",
        mpc_visibility=MpcVisibility(),
        target_positions=np.empty((0, 3), dtype=np.float32),
        target_orientations=np.empty((0, 3), dtype=np.float32),
        target_mesh_files=[],
        target_use_ply_positions=[],
        target_metadata=[],
        canonical_data=canon,
        segment_mask=segment_mask,
    )
    return view_model.to_render_packet()


def _visualizer(mode: str = RFXRAY_MODE_MPC_USAGE, **state_overrides):
    state = create_initial_state(show_rf_xray=True, rf_xray_mode=mode, **state_overrides)
    return SimpleNamespace(
        app_state=state,
        scenario_config=SimpleNamespace(raytracing=SimpleNamespace(materials=None)),
        mesh_entries=[
            {
                "name": "wall",
                "stable_mesh_id": "wall",
                "material_type": "concrete",
                "pbr_properties": {"material_type": "concrete", "color": [0.5, 0.5, 0.5]},
            },
            {
                "name": "window",
                "stable_mesh_id": "window",
                "material_type": "glass",
                "pbr_properties": {"material_type": "glass", "color": [0.1, 0.2, 0.7]},
            },
        ],
        target_entries=[],
        material_pbr_service=None,
        scene_service=SimpleNamespace(_mesh_id_to_group={}, _merged_meshes={}),
    )


def _file_scenario(root, directory: str | None = None) -> Scenario:
    data_spec: dict = {"mode": "files"}
    if directory is not None:
        data_spec["files"] = {"directory": directory}
    return Scenario(
        root=root,
        scene_spec={},
        data_mode="files",
        data_spec=data_spec,
        view_defaults={},
        timeline=TimelineSpec(steps=1, duration_s=0.0),
    )


def test_material_map_snapshot_colors_scene_materials_by_geometry_name():
    viz = _visualizer(mode=RFXRAY_MODE_MATERIAL_MAP)
    service = RFXRayAnalysisService(viz)

    snapshot = service.build_snapshot(_frame_packet(_canonical_material_frame(), np.ones(4, bool)))

    assert snapshot.enabled is True
    assert snapshot.mode == RFXRAY_MODE_MATERIAL_MAP
    assert "scene:wall::mesh" in snapshot.geometry_colors
    assert "scene:window::mesh" in snapshot.geometry_colors
    assert {row.material_key for row in snapshot.usage} == {"concrete", "glass"}
    assert [(row.material_key, row.display_name) for row in snapshot.legend_entries] == [
        ("concrete", "Concrete"),
        ("glass", "Glass"),
    ]
    assert "2 scene materials" in snapshot.summary


def test_mpc_usage_snapshot_aggregates_visible_material_segments_with_linear_path_loss():
    viz = _visualizer(mode=RFXRAY_MODE_MPC_USAGE)
    service = RFXRayAnalysisService(viz)
    canon = _canonical_material_frame()

    snapshot = service.build_snapshot(_frame_packet(canon, np.array([False, True, False, True])))

    usage = {row.material_key: row for row in snapshot.usage}
    assert usage["concrete"].bounce_count == 1
    assert usage["concrete"].path_count == 1
    assert usage["concrete"].weight == pytest.approx(1e-4)
    assert usage["glass"].weight == pytest.approx(1e-5)
    assert usage["concrete"].normalized_score == pytest.approx(1.0)
    assert usage["glass"].normalized_score == pytest.approx(0.1)
    assert snapshot.bounce_points is None
    assert snapshot.bounce_colors is None


def test_mpc_usage_snapshot_matches_ground_scene_alias_to_frame_material():
    canon = CanonicalStepData(
        points=np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        lines=np.array([[0, 1], [1, 2]], dtype=np.int32),
        order=np.array([0, 1, 0], dtype=np.uint8),
        itype=np.array([0, 1, 0], dtype=np.uint8),
        delay=np.zeros(3, dtype=np.float32),
        loss=np.zeros(3, dtype=np.float32),
        path_id=np.array([0, 0, 0], dtype=np.int32),
        path_start_indices=np.array([0], dtype=np.int32),
        path_orders=np.array([1], dtype=np.uint8),
        path_losses=np.array([50.0], dtype=np.float32),
        segment_start_indices=np.array([0, 1], dtype=np.int32),
        segment_end_indices=np.array([1, 2], dtype=np.int32),
        segment_order=np.array([0, 1], dtype=np.uint8),
        segment_itype=np.array([1, 1], dtype=np.uint8),
        segment_path_id=np.array([0, 0], dtype=np.int32),
        segment_material_ids=np.array([0, 1], dtype=np.int16),
        material_ids=np.array([0, 1, 0], dtype=np.int16),
        material_id_to_name={0: "", 1: "ground_asphalt"},
        material_id_to_itu={0: "", 1: "concrete"},
        material_id_to_bare={0: "", 1: "asphalt"},
    )
    viz = _visualizer(mode=RFXRAY_MODE_MPC_USAGE)
    viz.mesh_entries = [
        {
            "name": "ground",
            "stable_mesh_id": "ground",
            "material_type": "ground_asphalt",
            "pbr_properties": {
                "material_type": "ground_asphalt",
                "color": [0.18, 0.18, 0.18],
            },
        }
    ]

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(canon, np.array([False, True]))
    )

    assert [row.material_key for row in snapshot.usage] == ["asphalt"]
    assert snapshot.usage[0].weight == pytest.approx(1e-5)
    assert snapshot.usage[0].unknown_material is False
    assert snapshot.geometry_colors["scene:ground::mesh"] == pytest.approx(snapshot.usage[0].color)


def test_mpc_usage_snapshot_respects_segment_mask_and_excludes_hidden_materials():
    viz = _visualizer(mode=RFXRAY_MODE_MPC_USAGE)
    service = RFXRayAnalysisService(viz)

    snapshot = service.build_snapshot(
        _frame_packet(_canonical_material_frame(), np.array([False, True, False, False]))
    )

    assert [row.material_key for row in snapshot.usage] == ["concrete"]
    assert snapshot.bounce_points is None
    assert snapshot.bounce_colors is None


def test_mpc_usage_snapshot_builds_top_path_line_payload_for_strongest_paths():
    viz = _visualizer(
        mode=RFXRAY_MODE_MPC_USAGE,
        rf_xray_show_top_paths=True,
        rf_xray_max_top_paths=1,
    )
    service = RFXRayAnalysisService(viz)

    snapshot = service.build_snapshot(
        _frame_packet(_canonical_material_frame(), np.array([True, True, True, True]))
    )

    assert [path.path_id for path in snapshot.top_paths] == [0]
    assert snapshot.top_path_points is not None
    assert snapshot.top_path_lines is not None
    assert snapshot.top_path_colors is not None
    assert snapshot.top_path_lines.shape == (2, 2)


def test_mpc_usage_color_uses_standard_continuous_lut(monkeypatch):
    lut = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.2, 0.5, 0.8],
            [1.0, 0.9, 0.1],
        ],
        dtype=np.float32,
    )
    monkeypatch.setattr(rf_xray_module, "ensure_continuous_lut", lambda: lut)

    assert RFXRayAnalysisService._usage_color(1.0) == pytest.approx((1.0, 0.9, 0.1, 0.86))
    assert RFXRayAnalysisService._usage_color(0.0) == pytest.approx((0.18, 0.20, 0.24, 0.28))


def test_snapshot_opacity_scales_overlay_alpha_and_signature():
    viz = _visualizer(mode=RFXRAY_MODE_MATERIAL_MAP, rf_xray_opacity=0.5)
    service = RFXRayAnalysisService(viz)

    snapshot = service.build_snapshot(_frame_packet(_canonical_material_frame(), np.ones(4, bool)))

    assert snapshot.overlay_opacity == pytest.approx(0.5)
    assert snapshot.geometry_colors["scene:wall::mesh"][3] == pytest.approx(0.5)
    assert snapshot.signature[3] == pytest.approx(0.5)


def test_snapshot_opacity_is_absolute_for_material_map_and_mpc_usage():
    packet = _frame_packet(_canonical_material_frame(), np.ones(4, bool))

    material_snapshot = RFXRayAnalysisService(
        _visualizer(mode=RFXRAY_MODE_MATERIAL_MAP, rf_xray_opacity=1.0)
    ).build_snapshot(packet)
    mpc_snapshot = RFXRayAnalysisService(
        _visualizer(mode=RFXRAY_MODE_MPC_USAGE, rf_xray_opacity=1.0)
    ).build_snapshot(packet)
    partial_mpc_snapshot = RFXRayAnalysisService(
        _visualizer(mode=RFXRAY_MODE_MPC_USAGE, rf_xray_opacity=0.35)
    ).build_snapshot(packet)

    assert material_snapshot.geometry_colors["scene:wall::mesh"][3] == pytest.approx(1.0)
    assert mpc_snapshot.geometry_colors["scene:wall::mesh"][3] == pytest.approx(1.0)
    assert partial_mpc_snapshot.geometry_colors["scene:wall::mesh"][3] == pytest.approx(0.35)


def test_material_properties_snapshot_uses_scenario_material_overrides():
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="conductivity",
    )
    viz.scenario_config = SimpleNamespace(
        raytracing=SimpleNamespace(
            materials={
                "itu_concrete": {"conductivity": 0.25},
                "mat-itu_glass": {"conductivity": 0.05},
            },
            scene_materials=SimpleNamespace(scattering_coefficient_preset="none"),
        )
    )

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert snapshot.mode == RFXRAY_MODE_MATERIAL_PROPERTIES
    assert snapshot.scalar_property == "conductivity"
    assert snapshot.scalar_range == pytest.approx((0.05, 0.25))
    assert values == {"concrete": 0.25, "glass": 0.05}
    assert snapshot.missing_material_keys == ()


def test_material_properties_snapshot_uses_default_file_directory_when_omitted(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frames_manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "material_properties": {
                        "schema_version": 1,
                        "source": "sionna.rt.Scene.radio_materials",
                        "properties": {
                            "itu_concrete": {"relative_permittivity": 5.5},
                            "mat-itu_glass": {"relative_permittivity": 6.8},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="relative_permittivity",
    )
    viz.scenario = _file_scenario(tmp_path)
    viz.frame_source = make_frame_source(viz.scenario)
    assert isinstance(viz.frame_source, FileSource)
    assert viz.frame_source.directory == "frames"
    viz.scenario_config = SimpleNamespace(
        raytracing=SimpleNamespace(
            materials=None,
            scene_materials=SimpleNamespace(scattering_coefficient_preset="none"),
        ),
    )

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert values == {"concrete": 5.5, "glass": 6.8}
    assert snapshot.missing_material_keys == ()


def test_material_properties_snapshot_uses_custom_file_source_directory(tmp_path):
    frames_dir = tmp_path / "selected" / "frames"
    frames_dir.mkdir(parents=True)
    (frames_dir / "frames_manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "material_properties": {
                        "schema_version": 1,
                        "properties": {
                            "itu_concrete": {"conductivity": 0.31},
                            "mat-itu_glass": {"conductivity": 0.07},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="conductivity",
    )
    viz.scenario = _file_scenario(tmp_path, "selected/frames")
    viz.frame_source = make_frame_source(viz.scenario)
    viz.scenario_config = SimpleNamespace(
        raytracing=SimpleNamespace(
            materials=None,
            scene_materials=SimpleNamespace(scattering_coefficient_preset="none"),
        ),
    )

    service = RFXRayAnalysisService(viz)
    snapshot = service.build_snapshot(_frame_packet(_canonical_material_frame(), np.ones(4, bool)))
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert isinstance(viz.frame_source, FileSource)
    assert viz.frame_source.directory == "selected/frames"
    assert service._frames_manifest_path() == frames_dir / "frames_manifest.json"
    assert values == {"concrete": 0.31, "glass": 0.07}
    assert snapshot.missing_material_keys == ()


def test_material_properties_snapshot_uses_remote_frame_source_material_properties():
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="conductivity",
    )
    viz.frame_source = SimpleNamespace(
        metadata={
            "material_properties": {
                "schema_version": 1,
                "properties": {
                    "itu_concrete": {"conductivity": 0.21},
                    "mat-itu_glass": {"conductivity": 0.02},
                },
            }
        }
    )

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert values == {"concrete": 0.21, "glass": 0.02}
    assert snapshot.missing_material_keys == ()


def test_material_properties_scenario_override_wins_over_manifest(tmp_path):
    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    (frames_dir / "frames_manifest.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "material_properties": {
                        "schema_version": 1,
                        "properties": {
                            "itu_concrete": {"conductivity": 0.1},
                            "mat-itu_glass": {"conductivity": 0.05},
                        },
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="conductivity",
    )
    viz.scenario_config = {
        "root": tmp_path,
        "raytracing": {
            "materials": {"concrete": {"conductivity": 0.25}},
            "scene_materials": {"scattering_coefficient_preset": "none"},
        },
    }

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert values == {"concrete": 0.25, "glass": 0.05}


def test_material_properties_snapshot_uses_live_sionna_scene_materials():
    class TensorScalar:
        def __init__(self, value):
            self.value = value

        def numpy(self):
            return np.array([self.value], dtype=np.float32)

    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="relative_permittivity",
    )
    viz.sionna_scene = SimpleNamespace(
        radio_materials={
            "itu_concrete": SimpleNamespace(
                name="itu_concrete",
                relative_permittivity=TensorScalar(5.2),
            ),
            "mat-itu_glass": SimpleNamespace(
                name="mat-itu_glass",
                relative_permittivity=6.4,
            ),
        }
    )

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert values == pytest.approx({"concrete": 5.2, "glass": 6.4})
    assert snapshot.missing_material_keys == ()


def test_material_properties_missing_values_are_neutral_and_reported():
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="relative_permittivity",
        rf_xray_opacity=0.6,
    )
    viz.scenario_config = {
        "raytracing": {
            "materials": {"concrete": {"relative_permittivity": 5.5}},
            "scene_materials": {"scattering_coefficient_preset": "none"},
        }
    }

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )

    assert snapshot.missing_material_keys == ("glass",)
    assert "missing data: Glass" in snapshot.summary
    assert snapshot.geometry_colors["scene:window::mesh"] == pytest.approx((0.18, 0.20, 0.24, 0.6))


def test_material_properties_scattering_uses_itu_preset_fallback():
    viz = _visualizer(
        mode=RFXRAY_MODE_MATERIAL_PROPERTIES,
        rf_xray_property="scattering_coefficient",
    )
    viz.scenario_config = {
        "raytracing": {
            "materials": {"glass": {"scattering_coefficient": 0.0}},
            "scene_materials": {"scattering_coefficient_preset": "itu"},
        }
    }

    snapshot = RFXRayAnalysisService(viz).build_snapshot(
        _frame_packet(_canonical_material_frame(), np.ones(4, bool))
    )
    values = {row.material_key: row.property_value for row in snapshot.usage}

    assert values["concrete"] == pytest.approx(0.40)
    assert values["glass"] == pytest.approx(0.0)
    assert snapshot.missing_material_keys == ()


def test_mpc_usage_signature_tracks_canonical_usage_arrays_when_render_revision_is_same():
    viz = _visualizer(mode=RFXRAY_MODE_MPC_USAGE)
    service = RFXRayAnalysisService(viz)
    scene_materials = service._collect_scene_materials()
    segment_mask = np.array([False, True, False, True])
    first_canon = _canonical_material_frame()
    second_canon = replace(
        first_canon,
        path_losses=np.array([45.0, 55.0], dtype=np.float32),
    )
    first_view = SimpleNamespace(
        mpc_line_revision=("same-rendered-lines",),
        segment_mask=segment_mask,
        canonical_data=first_canon,
    )
    second_view = SimpleNamespace(
        mpc_line_revision=("same-rendered-lines",),
        segment_mask=segment_mask,
        canonical_data=second_canon,
    )

    first_signature = service._signature(
        first_view,
        viz.app_state,
        scene_materials,
        {},
        "none",
        True,
        RFXRAY_MODE_MPC_USAGE,
        "scattering_coefficient",
        viz.app_state.rf_xray_opacity,
    )
    second_signature = service._signature(
        second_view,
        viz.app_state,
        scene_materials,
        {},
        "none",
        True,
        RFXRAY_MODE_MPC_USAGE,
        "scattering_coefficient",
        viz.app_state.rf_xray_opacity,
    )

    assert first_signature != second_signature
