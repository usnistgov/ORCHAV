"""Tests for bounded, server-owned live scene XML staging."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from generator.io.grpc.cache import GeneratorFrameCache
from generator.io.grpc.live_scene_update import LiveSceneXmlStager
from generator.io.grpc.live_server import GeneratorService


def _write_source_scene(root: Path) -> Path:
    source = root / "scene.xml"
    source.write_text('<scene version="3.0.0"/>', encoding="utf-8")
    return source


def test_stager_publishes_a_server_named_sibling_and_cleans_it(tmp_path: Path) -> None:
    source = _write_source_scene(tmp_path)
    stager = LiveSceneXmlStager()
    payload = '<scene version="3.0.0"><shape type="rectangle"/></scene>'

    candidate = stager.stage(payload, source_scene_path=source)
    stager.accept(candidate)

    assert candidate.parent == source.parent
    assert candidate.name.startswith(".orchav-live-")
    assert candidate.suffix == ".xml"
    assert candidate.read_text(encoding="utf-8") == payload
    assert source.read_text(encoding="utf-8") == '<scene version="3.0.0"/>'
    assert stager.active_path == candidate.resolve()

    stager.close()

    assert not candidate.exists()
    assert source.exists()


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "<scene>",
        "<shape/>",
    ],
)
def test_invalid_payload_does_not_create_a_candidate(
    tmp_path: Path,
    payload: str,
) -> None:
    source = _write_source_scene(tmp_path)
    stager = LiveSceneXmlStager()

    with pytest.raises(ValueError):
        stager.stage(payload, source_scene_path=source)

    assert list(tmp_path.iterdir()) == [source]
    assert stager.active_path is None


def test_oversized_payload_does_not_create_a_candidate(tmp_path: Path) -> None:
    source = _write_source_scene(tmp_path)
    stager = LiveSceneXmlStager(max_payload_bytes=16)

    with pytest.raises(ValueError, match="too large"):
        stager.stage("<scene>payload</scene>", source_scene_path=source)

    assert list(tmp_path.iterdir()) == [source]


def test_accepting_a_second_candidate_removes_the_previous_one(tmp_path: Path) -> None:
    source = _write_source_scene(tmp_path)
    stager = LiveSceneXmlStager()
    first = stager.stage("<scene><shape/></scene>", source_scene_path=source)
    stager.accept(first)
    second = stager.stage("<scene><sensor/></scene>", source_scene_path=source)

    stager.accept(second)

    assert not first.exists()
    assert second.exists()
    assert stager.active_path == second.resolve()


def _live_service(source: Path, *, reject_candidate: bool = False):
    simulation_config = SimpleNamespace(
        scene_name=str(source),
        num_steps=1,
        duration=1.0,
    )
    scenario_config = SimpleNamespace(scene_xml=source)

    def build_scene(*_args) -> None:
        if reject_candidate and ".orchav-live-" in simulation_config.scene_name:
            raise RuntimeError("candidate scene rejected")

    scene_service = SimpleNamespace(
        simulation_config=simulation_config,
        target_managers=[],
        build_scene=MagicMock(side_effect=build_scene),
    )
    actor_state_service = SimpleNamespace(
        simulation_config=simulation_config,
        prepare_actor_state=MagicMock(return_value=(None, object())),
    )
    raytracing_service = SimpleNamespace(
        simulation_config=simulation_config,
        simulation_objects=SimpleNamespace(
            simulation_config=simulation_config,
            settings={},
        ),
    )

    def prepare_simulation(*_args, **_kwargs) -> None:
        raytracing_service.simulation_objects = SimpleNamespace(
            simulation_config=simulation_config,
            settings={},
        )

    raytracing_service.prepare_simulation = MagicMock(side_effect=prepare_simulation)
    service = GeneratorService(
        GeneratorFrameCache(max_frames=2, ttl_seconds=0, max_size_bytes=0),
        {
            "scenario_configuration": scenario_config,
            "simulation_config": simulation_config,
            "services": {
                "scene_service": scene_service,
                "actor_state_service": actor_state_service,
                "raytracing_service": raytracing_service,
            },
            "configs": {
                "tx_configs": [],
                "rx_configs": [],
                "target_configs": [],
                "simulation_config": simulation_config,
            },
        },
    )
    return service, scenario_config, simulation_config, scene_service


def test_service_activates_a_staged_scene_without_overwriting_source(tmp_path: Path) -> None:
    source = _write_source_scene(tmp_path)
    service, scenario_config, simulation_config, _scene_service = _live_service(source)

    simulation = service._activate_live_scene_xml("<scene><shape/></scene>")

    active = service._scene_xml_stager.active_path
    assert active is not None and active.exists()
    assert scenario_config.scene_xml == active
    assert simulation_config.scene_name == str(active)
    assert simulation.simulation_config is simulation_config
    assert source.read_text(encoding="utf-8") == '<scene version="3.0.0"/>'

    service.close()

    assert not active.exists()
    assert source.exists()


def test_failed_activation_rebuilds_the_previous_scene_and_discards_candidate(
    tmp_path: Path,
) -> None:
    source = _write_source_scene(tmp_path)
    service, scenario_config, simulation_config, scene_service = _live_service(
        source,
        reject_candidate=True,
    )

    with pytest.raises(RuntimeError, match="previous scene was restored"):
        service._activate_live_scene_xml("<scene><shape/></scene>")

    assert scenario_config.scene_xml == source
    assert simulation_config.scene_name == str(source)
    assert scene_service.build_scene.call_count == 2
    assert not list(tmp_path.glob(".orchav-live-*"))
    assert service.generation_epoch == 0


def test_invalid_service_payload_leaves_active_scene_untouched(tmp_path: Path) -> None:
    source = _write_source_scene(tmp_path)
    service, scenario_config, simulation_config, scene_service = _live_service(source)

    with pytest.raises(ValueError, match="not well formed"):
        service._activate_live_scene_xml("<scene>")

    assert scenario_config.scene_xml == source
    assert simulation_config.scene_name == str(source)
    scene_service.build_scene.assert_not_called()
    assert service.generation_epoch == 0
    assert list(tmp_path.iterdir()) == [source]
