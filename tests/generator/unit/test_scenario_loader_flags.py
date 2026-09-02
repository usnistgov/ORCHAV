import textwrap
from pathlib import Path

import pytest

from generator.core.configuration import build_simulation_config
from shared.scenarios import load_scenario_configuration
from shared.scenarios.yaml import validate_scenario_data


def write_yaml(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "scenario.yaml"
    p.write_text(textwrap.dedent(content))
    return p


def test_scenario_loader_debug_and_scene_material_defaults(tmp_path: Path):
    # Minimal scenario without debug_level/scene_materials uses defaults
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline: {steps: 1, duration_s: 0.0}
        scene:
          id: lobby/lobby.xml
          source: library
        data:
          files:
            format: hdf5
            directory: frames
            pattern: mpc_frames_*.h5
        raytracing:
          enabled: false
        coverage:
          enabled: false
        """,
    )
    scx = load_scenario_configuration(yaml, project_root=Path.cwd())
    sim = build_simulation_config(scx)
    assert sim.scene_material_scattering_coefficient_preset == "none"
    # debug_level is absent unless explicitly set in YAML
    assert getattr(scx, "debug_level", None) is None


def test_scenario_loader_debug_and_scene_material_preset_applied(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline: {steps: 1, duration_s: 0.0}
        debug_level: DEBUG
        scene:
          id: lobby/lobby.xml
          source: library
        data:
          files:
            format: hdf5
            directory: frames
            pattern: mpc_frames_*.h5
        raytracing:
          enabled: false
          scene_materials:
            scattering_coefficient_preset: itu
        coverage:
          enabled: false
        """,
    )
    scx = load_scenario_configuration(yaml, project_root=Path.cwd())
    # debug_level is uppercased in loader
    assert getattr(scx, "debug_level", "").upper() == "DEBUG"

    # Build SimulationConfig from scenario and ensure flags propagate
    sc = build_simulation_config(scx)
    assert sc.scene_material_scattering_coefficient_preset == "itu"
    assert sc.debug_level.upper() == "DEBUG"


def test_unsupported_enable_diffusion_key_is_rejected_as_unknown_key():
    with pytest.raises(ValueError, match="Unknown key 'enable_diffusion'"):
        validate_scenario_data(
            {
                "schema_version": 2,
                "enable_diffusion": True,
                "scene": {"id": "box/box.xml", "source": "library"},
            }
        )


def test_unsupported_scene_material_scattering_key_reports_exact_path():
    with pytest.raises(
        ValueError,
        match="raytracing.scene_materials.default_scattering_coefficients",
    ):
        validate_scenario_data(
            {
                "schema_version": 2,
                "scene": {"id": "box/box.xml", "source": "library"},
                "raytracing": {"scene_materials": {"default_scattering_coefficients": "itu"}},
            }
        )


def test_scenario_loader_applies_live_grpc_endpoint_overrides(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline: {steps: 1, duration_s: 0.0}
        scene:
          id: box/box.xml
          source: library
        data:
          mode: live_grpc
          live_grpc:
            endpoint: grpc://localhost:60000
            buffer_size: 25
        raytracing:
          enabled: false
        coverage:
          enabled: false
        """,
    )
    scx = load_scenario_configuration(yaml, project_root=Path.cwd())
    sim = build_simulation_config(scx)

    assert scx.data_mode == "live_grpc"
    assert scx.live_grpc_endpoints["sionna"] == "grpc://localhost:60000"
    assert sim.grpc_config == {
        "endpoint": "grpc://localhost:60000",
        "advertised_host": "localhost",
        "port": 60000,
    }


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [
        ("http://localhost:60000", "grpc:// scheme"),
        ("grpc://localhost", "include a port"),
        ("grpc://localhost:not-a-port", "invalid port"),
        ("grpc://localhost:0", "between 1 and 65535"),
    ],
)
def test_simulation_config_rejects_invalid_live_grpc_endpoint(
    tmp_path: Path,
    endpoint: str,
    message: str,
) -> None:
    yaml = write_yaml(
        tmp_path,
        f"""
        schema_version: 2
        timeline: {{steps: 1, duration_s: 0.0}}
        scene:
          id: box/box.xml
          source: library
        data:
          mode: live_grpc
          live_grpc:
            endpoint: {endpoint}
        raytracing:
          enabled: false
        coverage:
          enabled: false
        """,
    )
    with pytest.raises(ValueError, match=message):
        scenario = load_scenario_configuration(yaml, project_root=Path.cwd())
        build_simulation_config(scenario)


def test_scenario_loader_preserves_raytracing_start_step(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline: {steps: 10, duration_s: 9.0}
        scene:
          id: box/box.xml
          source: library
        data:
          files:
            format: hdf5
            directory: frames
            pattern: mpc_frames_*.h5
        raytracing:
          enabled: true
          start_step: 3
        coverage:
          enabled: false
        """,
    )
    scx = load_scenario_configuration(yaml, project_root=Path.cwd())
    sim = build_simulation_config(scx)

    assert scx.raytracing["start_step"] == 3
    assert sim.start_step == 3


def test_scenario_loader_accepts_v2_actors(tmp_path: Path):
    yaml = write_yaml(
        tmp_path,
        """
        schema_version: 2
        timeline:
          steps: 12
          duration_s: 12.0
        scene:
          id: etoile
          source: sionna
        raytracing:
          enabled: true
        actors:
          tx:
            - name: TX1
              mobility:
                type: stationary
                position_m: [-12.3, -56.0, 15.0]
          rx:
            - name: RX1
              mobility:
                type: linear
                start_m: [57.0, -49.0, 1.5]
                end_m: [72.0, -38.0, 1.5]
        """,
    )

    scx = load_scenario_configuration(yaml, project_root=Path.cwd())

    assert scx.actors.tx[0].name == "TX1"
    assert scx.actors.rx[0].mobility.type == "linear"
    assert scx.timeline.steps == 12
