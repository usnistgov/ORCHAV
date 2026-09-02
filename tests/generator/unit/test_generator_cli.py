import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]


def _load_generator_cli_module():
    module_path = ROOT / "generator" / "__main__.py"
    spec = importlib.util.spec_from_file_location("generator_cli_test", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


generator_cli = _load_generator_cli_module()


def test_help_uses_argparse(capsys):
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage: orchav-generator" in out
    assert "With no scenario, list curated producer entry" in out
    assert "--geometry-only" in out
    assert "--list" not in out
    assert "--progress-format {text,jsonl}" in out
    assert "--data-mode {files,live_grpc}" in out
    assert "--grpc-port GRPC_PORT" in out
    assert "--bind-host GRPC_BIND_HOST" in out
    assert "authoring-snapshot" not in out


def test_no_argument_catalog_includes_yaml_and_scripted_scenarios(tmp_path, monkeypatch, capsys):
    yaml_only = tmp_path / "scenarios" / "getting_started" / "hello_world"
    yaml_only.mkdir(parents=True)
    (yaml_only / "scenario.yaml").write_text("scene: {}\n", encoding="utf-8")

    yaml_with_script = tmp_path / "scenarios" / "getting_started" / "hello_world_scripted"
    yaml_with_script.mkdir(parents=True)
    (yaml_with_script / "scenario.yaml").write_text("scene: {}\n", encoding="utf-8")
    (yaml_with_script / "generate.py").write_text("# script\n", encoding="utf-8")

    script_only = tmp_path / "scenarios" / "generator" / "custom_driver"
    script_only.mkdir(parents=True)
    (script_only / "generate.py").write_text("# script\n", encoding="utf-8")

    remote_client = tmp_path / "scenarios" / "visualizer" / "remote_client"
    remote_client.mkdir(parents=True)
    (remote_client / "scenario.yaml").write_text(
        "data:\n  mode: ' remote_hdf5 '\n",
        encoding="utf-8",
    )

    remote_generation = remote_client / "generation"
    remote_generation.mkdir()
    (remote_generation / "scenario.yaml").write_text("scene: {}\n", encoding="utf-8")

    live_generator = tmp_path / "scenarios" / "visualizer" / "live_generator"
    live_generator.mkdir(parents=True)
    (live_generator / "scenario.yaml").write_text(
        "data:\n  mode: live_grpc\n",
        encoding="utf-8",
    )

    hidden = tmp_path / "scenarios" / "_hidden" / "not_listed"
    hidden.mkdir(parents=True)
    (hidden / "generate.py").write_text("# hidden\n", encoding="utf-8")

    private = tmp_path / "scenarios" / "generator" / "private" / "not_listed"
    private.mkdir(parents=True)
    (private / "scenario.yaml").write_text("scene: {}\n", encoding="utf-8")

    monkeypatch.setattr(generator_cli, "_project_root", lambda: tmp_path)

    generator_cli.main([])

    out = capsys.readouterr().out
    lines = out.splitlines()
    assert "[YAML] scenarios/getting_started/hello_world/" in out
    assert "orchav-generator scenarios/getting_started/hello_world/" in out
    assert "[Python-scripted] scenarios/getting_started/hello_world_scripted/" in out
    assert "python scenarios/getting_started/hello_world_scripted/generate.py" in out
    assert "orchav-generator scenarios/getting_started/hello_world_scripted/" not in out
    assert "optional scripted driver" not in out
    assert "[Python-scripted] scenarios/generator/custom_driver/" in out
    assert "python scenarios/generator/custom_driver/generate.py" in out
    assert "  [YAML] scenarios/visualizer/remote_client/" not in lines
    assert "    orchav-generator scenarios/visualizer/remote_client/" not in lines
    assert "[YAML] scenarios/visualizer/remote_client/generation/" in out
    assert "orchav-generator scenarios/visualizer/remote_client/generation/" in out
    assert "[YAML] scenarios/visualizer/live_generator/" in out
    assert "orchav-generator scenarios/visualizer/live_generator/" in out
    assert "_hidden" not in out
    assert "scenarios/generator/private/not_listed" not in out


def test_removed_list_option_reports_cli_error(capsys):
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["--list"])

    assert exc.value.code == 2
    assert "unrecognized arguments: --list" in capsys.readouterr().err


def test_repository_catalog_advertises_only_supported_primary_commands():
    out = generator_cli.format_scenario_catalog()
    lines = out.splitlines()

    for path in (
        "scenarios/getting_started/hello_world_scripted",
        "scenarios/visualizer/multi_device_trajectory",
        "scenarios/visualizer/synthetic_mpc_benchmark",
    ):
        assert f"[Python-scripted] {path}/" in out
        assert f"python {path}/generate.py" in out
        assert f"orchav-generator {path}/" not in out

    remote_client = "scenarios/visualizer/data_modes/remote_hdf5"
    assert f"  [YAML] {remote_client}/" not in lines
    assert f"    orchav-generator {remote_client}/" not in lines
    assert f"[YAML] {remote_client}/generation/" in out
    assert f"orchav-generator {remote_client}/generation/" in out

    live_generator = "scenarios/visualizer/data_modes/live_grpc"
    assert f"[YAML] {live_generator}/" in out
    assert f"orchav-generator {live_generator}/" in out


def test_scenario_path_runs_yaml_pipeline(tmp_path, monkeypatch):
    calls = []
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )

    def fake_run_from_yaml(
        path: Path,
        geometry_only: bool = False,
        progress_format: str = "text",
    ):
        calls.append((path, geometry_only, progress_format))

    monkeypatch.setattr(generator_cli, "_run_from_yaml", fake_run_from_yaml)

    generator_cli.main(["--geometry-only", str(scenario_dir)])

    assert calls == [(scenario_dir, True, "text")]


def test_jsonl_progress_format_is_forwarded_to_yaml_runner(tmp_path, monkeypatch):
    calls = []
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )

    def fake_run_from_yaml(
        path: Path,
        geometry_only: bool = False,
        progress_format: str = "text",
    ):
        calls.append((path, geometry_only, progress_format))

    monkeypatch.setattr(generator_cli, "_run_from_yaml", fake_run_from_yaml)

    generator_cli.main(["--progress-format", "jsonl", str(scenario_dir)])

    assert calls == [(scenario_dir, False, "jsonl")]


def test_data_mode_and_live_transport_overrides_are_forwarded_to_yaml_runner(
    tmp_path,
    monkeypatch,
):
    calls = []
    scenario_dir = tmp_path / "scenario"
    scenario_dir.mkdir()
    (scenario_dir / "scenario.yaml").write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )

    def fake_run_from_yaml(
        path: Path,
        geometry_only: bool = False,
        progress_format: str = "text",
        *,
        data_mode: str | None = None,
        grpc_port: int | None = None,
        grpc_bind_host: str | None = None,
    ):
        calls.append((path, geometry_only, progress_format, data_mode, grpc_port, grpc_bind_host))

    monkeypatch.setattr(generator_cli, "_run_from_yaml", fake_run_from_yaml)

    generator_cli.main(
        [
            str(scenario_dir),
            "--data-mode",
            "live_grpc",
            "--grpc-port",
            "61000",
            "--bind-host",
            "0.0.0.0",
        ]
    )

    assert calls == [(scenario_dir, False, "text", "live_grpc", 61000, "0.0.0.0")]


def test_live_transport_overrides_require_scenario(capsys):
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["--grpc-port", "61000"])

    assert exc.value.code == 2
    assert "require a scenario path" in capsys.readouterr().err


def test_yaml_runner_applies_data_mode_before_building_simulation(tmp_path, monkeypatch):
    import generator
    import shared.scenarios
    import shared.scenarios.paths

    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )
    scenario = SimpleNamespace(data_mode="files", raytracing={"enabled": False})
    simulation = SimpleNamespace(start_step=0, num_steps=0)
    built_modes = []

    monkeypatch.setattr(
        shared.scenarios,
        "load_scenario_configuration",
        lambda _path: scenario,
    )
    monkeypatch.setattr(shared.scenarios.paths, "find_project_root", lambda _path: tmp_path)
    monkeypatch.setitem(
        generator._LOADED,
        "build_simulation_config",
        lambda loaded: built_modes.append(loaded.data_mode) or simulation,
    )
    monkeypatch.setitem(generator._LOADED, "perform_pipeline", lambda **_kwargs: None)

    generator_cli._run_from_yaml(scenario_yaml, data_mode="live_grpc")

    assert built_modes == ["live_grpc"]


def test_yaml_runner_rejects_grpc_options_for_effective_files_mode(tmp_path, monkeypatch):
    import shared.scenarios
    import shared.scenarios.paths

    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )
    scenario = SimpleNamespace(data_mode="files", raytracing={"enabled": False})
    monkeypatch.setattr(
        shared.scenarios,
        "load_scenario_configuration",
        lambda _path: scenario,
    )
    monkeypatch.setattr(shared.scenarios.paths, "find_project_root", lambda _path: tmp_path)

    with pytest.raises(ValueError, match="effective data mode to be live_grpc"):
        generator_cli._run_from_yaml(scenario_yaml, grpc_port=50052)


def test_private_authoring_snapshot_keeps_scenario_root_authoritative(
    tmp_path,
    monkeypatch,
):
    from shared.source_identity import EXPECTED_SOURCE_IDENTITY_ENV

    calls = []
    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    snapshot = scenario_root / ".orchav-aq-test.yaml"
    snapshot.write_text(
        "schema_version: 2\ndata: {files: {directory: redirected}}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(EXPECTED_SOURCE_IDENTITY_ENV, "builder-handshake-present")

    def fake_run_from_yaml(
        path: Path,
        geometry_only: bool = False,
        progress_format: str = "text",
        *,
        _authoring_snapshot_yaml: Path | None = None,
    ):
        calls.append((path, geometry_only, progress_format, _authoring_snapshot_yaml))

    monkeypatch.setattr(generator_cli, "_run_from_yaml", fake_run_from_yaml)

    generator_cli.main(
        [
            str(scenario_root),
            "--progress-format",
            "jsonl",
            "--_authoring-snapshot-yaml",
            str(snapshot),
        ]
    )

    assert calls == [(scenario_root, False, "jsonl", snapshot.resolve())]


def test_private_authoring_snapshot_is_not_a_public_untrusted_option(
    tmp_path,
    monkeypatch,
    capsys,
):
    from shared.source_identity import EXPECTED_SOURCE_IDENTITY_ENV

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    snapshot = scenario_root / ".orchav-aq-test.yaml"
    snapshot.write_text("schema_version: 2\n", encoding="utf-8")
    monkeypatch.delenv(EXPECTED_SOURCE_IDENTITY_ENV, raising=False)

    with pytest.raises(SystemExit) as exc:
        generator_cli.main(
            [
                str(scenario_root),
                "--_authoring-snapshot-yaml",
                str(snapshot),
            ]
        )

    assert exc.value.code == 2
    assert "reserved for Scenario Builder" in capsys.readouterr().err


def test_private_authoring_snapshot_must_be_a_real_direct_child(
    tmp_path,
    monkeypatch,
    capsys,
):
    from shared.source_identity import EXPECTED_SOURCE_IDENTITY_ENV

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    outside = tmp_path / ".orchav-aq-outside.yaml"
    outside.write_text("schema_version: 2\n", encoding="utf-8")
    monkeypatch.setenv(EXPECTED_SOURCE_IDENTITY_ENV, "builder-handshake-present")

    with pytest.raises(SystemExit) as exc:
        generator_cli.main(
            [
                str(scenario_root),
                "--_authoring-snapshot-yaml",
                str(outside),
            ]
        )

    assert exc.value.code == 2
    assert "directly below the authoritative scenario root" in capsys.readouterr().err


def test_private_authoring_snapshot_rejects_exact_file_symlink(
    tmp_path,
    monkeypatch,
    capsys,
):
    from shared.source_identity import EXPECTED_SOURCE_IDENTITY_ENV

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    target = scenario_root / ".orchav-aq-target.yaml"
    target.write_text("schema_version: 2\n", encoding="utf-8")
    link = scenario_root / ".orchav-aq-link.yaml"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"file symlink creation is unavailable: {exc}")
    monkeypatch.setenv(EXPECTED_SOURCE_IDENTITY_ENV, "builder-handshake-present")

    with pytest.raises(SystemExit) as exc:
        generator_cli.main(
            [
                str(scenario_root),
                "--_authoring-snapshot-yaml",
                str(link),
            ]
        )

    assert exc.value.code == 2
    assert "real private" in capsys.readouterr().err


def test_jsonl_progress_format_requires_scenario():
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["--progress-format", "jsonl"])

    assert exc.value.code == 2


def test_geometry_only_jsonl_run_can_complete_without_step_callbacks(tmp_path, monkeypatch, capsys):
    import generator
    import shared.scenarios
    import shared.scenarios.paths
    from shared.source_identity import (
        EXPECTED_SOURCE_IDENTITY_ENV,
        loaded_source_identity,
    )

    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )
    scenario = SimpleNamespace(raytracing={"enabled": True})
    simulation = SimpleNamespace(start_step=0, num_steps=3)
    calls = []

    monkeypatch.setattr(
        shared.scenarios,
        "load_scenario_configuration",
        lambda _path: scenario,
    )
    monkeypatch.setattr(shared.scenarios.paths, "find_project_root", lambda _path: tmp_path)
    monkeypatch.setitem(generator._LOADED, "build_simulation_config", lambda _scenario: simulation)
    actual_identity = loaded_source_identity("generator")
    expected_payload = actual_identity.to_dict()
    expected_payload["parent_only_marker"] = "must-not-be-echoed"
    monkeypatch.setenv(EXPECTED_SOURCE_IDENTITY_ENV, json.dumps(expected_payload))

    def fake_perform_pipeline(**kwargs):
        calls.append(kwargs)
        return "geometry output"

    monkeypatch.setitem(generator._LOADED, "perform_pipeline", fake_perform_pipeline)

    result = generator_cli._run_from_yaml(
        scenario_yaml,
        geometry_only=True,
        progress_format="jsonl",
    )

    records = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert result == "geometry output"
    assert [record["event"] for record in records] == ["run_started", "run_completed"]
    source_identity = records[0]["source_identity"]
    assert source_identity == actual_identity.to_dict()
    assert "parent_only_marker" not in source_identity
    assert Path(source_identity["source_root"]).samefile(Path(__file__).resolve().parents[3])
    assert source_identity["version"] == "0.1.0"
    assert records[0]["first_step"] == 0
    assert records[0]["frame_set_mode"] == "fresh_full"
    assert records[-1]["completed_steps"] == 0
    assert calls[0]["geometry_only"] is True
    assert calls[0]["show_progress"] is False


def test_yaml_runner_loads_private_snapshot_against_authoritative_root(
    tmp_path,
    monkeypatch,
):
    import generator
    import shared.scenarios
    import shared.scenarios.paths

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    snapshot = scenario_root / ".orchav-aq-test.yaml"
    snapshot.write_text("schema_version: 2\n", encoding="utf-8")
    scenario = SimpleNamespace(
        root=scenario_root.resolve(),
        frames_directory="frames",
        frames_dir=(scenario_root / "frames").resolve(),
        raytracing={"enabled": False},
    )
    simulation = SimpleNamespace(start_step=0, num_steps=0)
    load_calls = []

    def fake_load(path, *, yaml_path=None):
        load_calls.append((path, yaml_path))
        return scenario

    monkeypatch.setattr(shared.scenarios, "load_scenario_configuration", fake_load)
    monkeypatch.setattr(shared.scenarios.paths, "find_project_root", lambda _path: tmp_path)
    monkeypatch.setitem(generator._LOADED, "build_simulation_config", lambda _scenario: simulation)
    monkeypatch.setitem(generator._LOADED, "perform_pipeline", lambda **_kwargs: None)

    generator_cli._run_from_yaml(
        scenario_root,
        _authoring_snapshot_yaml=snapshot,
    )

    assert load_calls == [(scenario_root.resolve(), snapshot.resolve())]


def test_yaml_runner_rejects_snapshot_output_redirection(
    tmp_path,
    monkeypatch,
):
    import generator
    import shared.scenarios
    import shared.scenarios.paths

    scenario_root = tmp_path / "scenario"
    scenario_root.mkdir()
    (scenario_root / "scenario.yaml").write_text("schema_version: 2\n", encoding="utf-8")
    snapshot = scenario_root / ".orchav-aq-test.yaml"
    snapshot.write_text("schema_version: 2\n", encoding="utf-8")
    redirected = SimpleNamespace(
        root=scenario_root.resolve(),
        frames_directory="redirected",
        frames_dir=(scenario_root / "redirected").resolve(),
        raytracing={"enabled": False},
    )
    pipeline_calls = []

    monkeypatch.setattr(
        shared.scenarios,
        "load_scenario_configuration",
        lambda _root, *, yaml_path=None: redirected,
    )
    monkeypatch.setattr(shared.scenarios.paths, "find_project_root", lambda _path: tmp_path)
    monkeypatch.setitem(generator._LOADED, "build_simulation_config", lambda _scenario: None)
    monkeypatch.setitem(
        generator._LOADED,
        "perform_pipeline",
        lambda **kwargs: pipeline_calls.append(kwargs),
    )

    with pytest.raises(ValueError, match="outputs remain fixed"):
        generator_cli._run_from_yaml(
            scenario_root,
            _authoring_snapshot_yaml=snapshot,
        )

    assert pipeline_calls == []


def test_expected_source_mismatch_fails_before_scenario_loading(
    tmp_path,
    monkeypatch,
):
    import shared.scenarios
    from shared.source_identity import (
        EXPECTED_SOURCE_IDENTITY_ENV,
        SourceIdentityError,
        loaded_source_identity,
    )

    scenario_yaml = tmp_path / "scenario.yaml"
    scenario_yaml.write_text(
        "schema_version: 2\ntimeline: {steps: 1, duration_s: 0.0}\n",
        encoding="utf-8",
    )
    expected_payload = loaded_source_identity("generator").to_dict()
    expected_payload["source_root"] = str(tmp_path / "different-checkout")
    monkeypatch.setenv(EXPECTED_SOURCE_IDENTITY_ENV, json.dumps(expected_payload))
    load_calls = []
    monkeypatch.setattr(
        shared.scenarios,
        "load_scenario_configuration",
        lambda path: load_calls.append(path),
    )

    with pytest.raises(SourceIdentityError, match="does not match the launching process"):
        generator_cli._run_from_yaml(
            scenario_yaml,
            progress_format="jsonl",
        )

    assert load_calls == []


def test_missing_scenario_path_reports_cli_error(capsys):
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["path/to/scenario_folder/"])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "scenario path does not exist: path/to/scenario_folder" in err
    assert "orchav-generator` with no arguments" in err
    assert "Traceback" not in err


def test_directory_without_scenario_yaml_reports_cli_error(tmp_path, capsys):
    scenario_dir = tmp_path / "empty_scenario"
    scenario_dir.mkdir()

    with pytest.raises(SystemExit) as exc:
        generator_cli.main([str(scenario_dir)])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "scenario directory does not contain scenario.yaml" in err
    assert "Pass a scenario directory with scenario.yaml" in err
    assert "Traceback" not in err


def test_normal_cli_rejects_alternate_yaml_filename(tmp_path, capsys):
    alternate = tmp_path / "scenario_variant.yaml"
    alternate.write_text("schema_version: 2\n", encoding="utf-8")

    with pytest.raises(SystemExit) as exc:
        generator_cli.main([str(alternate)])

    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "accepts only a scenario directory or its canonical scenario.yaml" in err
    assert "Traceback" not in err


def test_geometry_only_requires_scenario():
    with pytest.raises(SystemExit) as exc:
        generator_cli.main(["--geometry-only"])

    assert exc.value.code == 2
