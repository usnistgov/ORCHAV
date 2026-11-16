"""Create a local inspection entry point beside a derived frame set.

Geometry remains in the source checkout. This small YAML is a viewer binding,
not a portable asset bundle or a new propagation acquisition.
"""

from pathlib import Path

import yaml

from shared.scenarios.paths import create_path_policy


def write_result_scenario(source, output):
    """Write a viewer scenario beside derived frames without copying source assets.

    ``source`` is the scenario directory or its frame directory; ``output``
    must already exist. Returns the new YAML path, or ``None`` if no source
    scenario is available. An existing output ``scenario.yaml`` is never
    overwritten.
    """
    source, output = Path(source).resolve(), Path(output).resolve()
    source_yaml = source / "scenario.yaml"
    if not source_yaml.is_file():
        source_yaml = source.parent / "scenario.yaml"
    if not source_yaml.is_file():
        return None
    config = yaml.safe_load(source_yaml.read_text())
    policy = create_path_policy(source_yaml)
    scene = config["scene"]
    # Compact scenario YAML can omit SceneModel's default library source.
    scene_source = scene.get("source", "library")
    if scene_source == "library":
        scene["id"] = str(policy.project_root / "libraries/scenes" / scene["id"])
        scene["source"] = "local"
    elif scene_source == "local":
        scene["id"] = str(policy.resolve_path(scene["id"]))
    for targets in config.get("actors", {}).values():
        for target in targets:
            asset = target.get("asset", {})
            if "path" in asset:
                path = Path(asset["path"])
                if not path.is_absolute():
                    path = policy.resolve_path(
                        str(path),
                        base=(
                            policy.project_root
                            if str(path).startswith("libraries/")
                            else source_yaml.parent
                        ),
                    )
                asset["path"] = str(path.resolve())
    config["data"] = {"mode": "files", "files": {"format": "hdf5", "directory": "frames"}}
    config.setdefault("raytracing", {})["enabled"] = False
    config["generator_summary"] = {"enabled": False}
    path = output / "scenario.yaml"
    with path.open("x") as handle:
        handle.write("# Local inspection of derived results; source assets are not copied.\n")
        yaml.safe_dump(config, handle, sort_keys=False)
    return path
