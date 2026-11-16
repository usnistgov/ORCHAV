"""Rewrite authored sensing YAML with grouped names, proving resolved equivalence.

Only supplied source YAMLs are changed. Generated recordings are not traversed.
The receipt records old/new source identities and complete configuration hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import yaml

from generator.core.configuration.loader import load_simulation_config
from generator.core.exceptions import ConfigurationError
from shared.scenarios.model import ScenarioModel
from shared.scenarios.sensing_groups import COHERENT_HANN, GROUPS, normalize_scenario_sensing


def fingerprint(data):
    normalized = normalize_scenario_sensing(data)
    runtime = asdict(load_simulation_config(SimpleNamespace(**normalized)))
    # Optional storage-profile fields do not alter legacy configuration identity
    # when the scenario did not select a profile.
    for key in ("recording_profile", "recording_display"):
        if runtime["sensing"].get(key) is None:
            runtime["sensing"].pop(key, None)
    schema = ScenarioModel.model_validate(normalized).model_dump(mode="json", exclude_none=True)
    # Comparing runtime handles omitted-vs-explicit defaults. Compare geometry,
    # clocks and output controls independently of the normalized sensing aliases.
    for key in ("sensing", "raytracing", "view_defaults", "generator_summary"):
        schema.pop(key, None)
    return {"runtime": runtime, "scene_and_actors": schema}


def effective_fingerprint(data):
    """Numerical/actor identity with equivalent output aliases and panel defaults.

    Keep ``fingerprint`` unchanged for the original migration receipts. This
    comparison additionally recognizes the HDF5 spelling alias, the runtime
    debug default and the existing sensing-panel inference rule.
    """
    result = fingerprint(data)
    # Optional axis hints affect presentation only, not processing or saved
    # numerical products. Record intentional changes to them separately.
    for key in ("display_range_xlim", "display_velocity_ylim"):
        result["runtime"]["sensing"].pop(key, None)
    # Solver custom options are consumed directly by RayTracingService, outside
    # SimulationConfig. Include their complete validated representation too.
    result["raytracing"] = ScenarioModel.model_validate(
        normalize_scenario_sensing(data)
    ).model_dump(mode="json")["raytracing"]
    schema = result["scene_and_actors"]
    schema.pop("debug_level", None)  # Already included in the complete runtime.
    files = schema.get("data", {}).get("files", {})
    if files.get("format") == "hdf5":
        files["format"] = "h5"
    visualizer = schema.get("visualizer") or {}
    panels = visualizer.get("panels") or {}
    configured = panels.get("sensing")
    inferred = bool(data.get("sensing", {}).get("enabled", False))
    if configured == {"enabled": inferred} or configured is inferred:
        panels.pop("sensing", None)
    if not panels:
        visualizer.pop("panels", None)
    schema["visualizer"] = visualizer
    return result


def omit_redundant_authoring(data):
    """Omit proven output/UI/solver defaults, keeping physical inputs explicit."""
    from shared.scenarios.parsers import parse_generator_summary_config

    def identity(doc):
        return (
            effective_fingerprint(doc),
            parse_generator_summary_config(normalize_scenario_sensing(doc), Path(".")),
        )

    working = deepcopy(data)
    baseline = identity(working)
    paths = [("debug_level",), ("data", "files", "format"), ("visualizer", "panels", "sensing")]
    summary = working.get("generator_summary") or {}
    paths += [("generator_summary", key) for key in summary if key not in {"enabled", "create"}]
    for path in paths:
        candidate = deepcopy(working)
        parent = candidate
        parents = []
        for key in path[:-1]:
            if not isinstance(parent, dict) or key not in parent:
                break
            parents.append((parent, key))
            parent = parent[key]
        else:
            if path[-1] not in parent:
                continue
            del parent[path[-1]]
            for container, key in reversed(parents):
                if container[key] == {}:
                    del container[key]
            try:
                if identity(candidate) == baseline:
                    working = candidate
            except (ValueError, TypeError, KeyError, ConfigurationError):
                pass
    return working


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str).encode()).hexdigest()


def leaves(spec, prefix=()):
    for key, value in spec.items():
        if isinstance(value, dict):
            yield from leaves(value, (*prefix, key))
        else:
            yield (*prefix, key), value


def grouped(values):
    out = {key: values[key] for key in ("enabled", "preset") if key in values}
    used = set()
    for path, destination in leaves(GROUPS):
        if destination in used:
            continue
        if (
            path == ("acquisition", "samples_per_cpi")
            and values.get("processing_mode") != "coherent_cpi"
        ):
            continue
        value = values
        for part in destination.split("."):
            if not isinstance(value, dict) or part not in value:
                break
            value = value[part]
        else:
            parent = out
            for part in path[:-1]:
                parent = parent.setdefault(part, {})
            parent[path[-1]] = (
                "unweighted_mean"
                if destination == "coherent_clutter_mode" and value == "legacy"
                else deepcopy(value)
            )
            used.add(destination)
    return out


def simplify(data):
    source = normalize_scenario_sensing(data)
    baseline = fingerprint(source)
    working = deepcopy(source)
    settings = working["sensing"]
    # Preset is explicitly named only for existing coherent finite-bandwidth
    # examples; all differing values are retained as overrides.
    if (
        settings.get("processing_mode") == "coherent_cpi"
        and settings.get("range_response", {}).get("model") == "cfr_ifft"
    ):
        for key in COHERENT_HANN:
            # Populate nested defaults too: a partial old range_response can
            # implicitly request boxcar, which the new Hann preset must not replace.
            settings[key] = deepcopy(baseline["runtime"]["sensing"][key])
        # Keep the actual acquisition explicit: it is not part of the processing preset.
        for key in ("bandwidth", "prf_hz", "fft_size_doppler", "range_response"):
            settings[key] = deepcopy(baseline["runtime"]["sensing"][key])
        settings["preset"] = "coherent_hann"
    else:
        settings.setdefault("range_response", {"model": "point_bin_floor"})
        settings.setdefault("coherent_clutter_mode", "legacy")
    acquisition_keys = (
        {"bandwidth", "prf_hz", "fft_size_doppler"} if settings.get("preset") else set()
    )
    for key in list(settings):
        if key in acquisition_keys:
            continue
        if key in {
            "enabled",
            "preset",
            "range_response",
            "processing_mode",
            "coherent_clutter_mode",
        }:
            continue
        candidate = deepcopy(working)
        del candidate["sensing"][key]
        try:
            equivalent = fingerprint(candidate) == baseline
        except (ValueError, TypeError, KeyError, ConfigurationError):
            equivalent = False
        if equivalent:
            working = candidate
    if working["sensing"].get("preset"):
        for key in ("processing_mode", "coherent_clutter_mode"):
            candidate = deepcopy(working)
            candidate["sensing"].pop(key, None)
            if fingerprint(candidate) == baseline:
                working = candidate
    response = working["sensing"].get("range_response", {})
    for key in list(response):
        if key == "num_frequency_bins" and working["sensing"].get("preset"):
            continue
        if key == "model" and not working["sensing"].get("preset"):
            continue
        candidate = deepcopy(working)
        del candidate["sensing"]["range_response"][key]
        if not candidate["sensing"]["range_response"]:
            candidate["sensing"].pop("range_response")
        try:
            equivalent = fingerprint(candidate) == baseline
        except (ValueError, TypeError, KeyError, ConfigurationError):
            equivalent = False
        if equivalent:
            working = candidate
    # Expose optional view hints outside the scientific settings.
    for key, name in [
        ("display_range_xlim", "range_xlim"),
        ("display_velocity_ylim", "velocity_ylim"),
    ]:
        value = working["sensing"].pop(key, None)
        if value is not None:
            working.setdefault("view_defaults", {}).setdefault("sensing", {})[name] = value
    # Only explicit coherent rate/count fields supply acquisition clocks;
    # the processing preset never supplies physical acquisition parameters.
    before_group = deepcopy(working["sensing"])
    if before_group.get("preset"):
        before_group["processing_mode"] = "coherent_cpi"
    working["sensing"] = grouped(before_group)
    if before_group.get("preset"):
        working["sensing"]["acquisition"].pop("mode", None)
    if before_group.get("preset") or before_group.get("processing_mode") == "coherent_cpi":
        for ray, old in [
            ("cir_time_steps", "fft_size_doppler"),
            ("cir_sampling_frequency_hz", "prf_hz"),
        ]:
            if old in before_group:
                working.get("raytracing", {}).pop(ray, None)
    summary = working.get("generator_summary", {})
    if "inspection_figures" in summary:
        summary.setdefault("sensing", {})["inspection_figures"] = summary.pop("inspection_figures")
    if fingerprint(working) != baseline:
        after = fingerprint(working)
        before = baseline["runtime"]["sensing"]
        raise ValueError(
            "Grouping changed resolved configuration: "
            + str(
                {
                    k: (v, after["runtime"]["sensing"][k])
                    for k, v in before.items()
                    if v != after["runtime"]["sensing"][k]
                }
            )
        )
    return working, baseline


def omit_schema_defaults(data):
    """Remove explicitly authored schema defaults without changing effective data."""
    from pydantic import BaseModel

    from shared.scenarios.parsers import parse_generator_summary_config

    def identity(doc):
        normalized = normalize_scenario_sensing(doc)
        model = ScenarioModel.model_validate(normalized).model_dump(mode="json")
        # The runtime owns scientific defaults; the summary parser owns output
        # defaults. Preserve all other normalized schema values strictly.
        for key in ("sensing", "generator_summary"):
            model.pop(key, None)
        return fingerprint(doc), model, parse_generator_summary_config(normalized, Path("."))

    result = deepcopy(data)
    baseline = identity(data)
    model = ScenarioModel.model_validate(data)
    candidates = []

    def inspect(raw, parsed, path=()):
        if isinstance(raw, dict) and isinstance(parsed, BaseModel):
            kept = parsed.model_dump(exclude_defaults=True)
            for key, value in raw.items():
                if key == "sensing" or key not in type(parsed).model_fields:
                    continue
                if key not in kept:
                    candidates.append((*path, key))
                else:
                    inspect(value, getattr(parsed, key), (*path, key))
        elif isinstance(raw, list) and isinstance(parsed, (list, tuple)):
            for index, (value, child) in enumerate(zip(raw, parsed)):
                inspect(value, child, (*path, index))

    inspect(result, model)
    for path in candidates:
        candidate = deepcopy(result)
        parent = candidate
        for key in path[:-1]:
            parent = parent[key]
        del parent[path[-1]]
        try:
            if identity(candidate) == baseline:
                result = candidate
        except (ValueError, TypeError, KeyError, ConfigurationError):
            pass
    return result


class _CompactYamlDumper(yaml.SafeDumper):
    """Keep short scalar lists (coordinates, bounds, selections) on one line."""


def _compact_sequence(dumper, value):
    node = dumper.represent_sequence("tag:yaml.org,2002:seq", value)
    node.flow_style = len(value) <= 4 and all(not isinstance(v, (dict, list)) for v in value)
    return node


_CompactYamlDumper.add_representer(list, _compact_sequence)


def replace_blocks(original, data, *, compact=False):
    """Preserve all unrelated bytes, comments, actor paths and long trajectories."""
    import re

    for name in (
        "sensing",
        "raytracing",
        "view_defaults",
        "generator_summary",
        "actors",
        "data",
        "scene",
        "debug_level",
        "visualizer",
    ):
        if name not in data:
            original = re.sub(
                rf"(?m)^{name}:.*?(?=^[A-Za-z_][A-Za-z_0-9]*:|\Z)", "", original, flags=re.S
            )
            continue
        if not compact and yaml.safe_load(original).get(name) == data[name]:
            continue
        replacement = yaml.dump(
            {name: data[name]}, Dumper=_CompactYamlDumper, sort_keys=False, width=1000
        )
        pattern = rf"(?m)^{name}:.*?(?=^[A-Za-z_][A-Za-z_0-9]*:|\Z)"
        match = re.search(pattern, original, flags=re.S)
        if match:
            original = original[: match.start()] + replacement + original[match.end() :]
        else:
            original += "\n" + replacement
    return original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="+")
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(args.receipt)
    rows = []
    prepared = []
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for path in args.paths:
        text = path.read_text()
        data, baseline = simplify(yaml.safe_load(text))
        data = omit_redundant_authoring(omit_schema_defaults(data))
        new_text = replace_blocks(text, data)
        assert effective_fingerprint(yaml.safe_load(new_text)) == effective_fingerprint(
            yaml.safe_load(text)
        )
        rows.append(
            {
                "path": path.as_posix(),
                "before_sha256": hashlib.sha256(text.encode()).hexdigest(),
                "after_sha256": hashlib.sha256(new_text.encode()).hexdigest(),
                "resolved_sha256": digest(baseline),
                "effective_before_sha256": digest(effective_fingerprint(yaml.safe_load(text))),
                "effective_final_sha256": digest(effective_fingerprint(yaml.safe_load(new_text))),
                "runtime_sensing": baseline["runtime"]["sensing"],
                "before_lines": len(text.splitlines()),
                "after_lines": len(new_text.splitlines()),
            }
        )
        prepared.append((path, new_text))
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(
        json.dumps({"source_revision": revision, "scenarios": rows}, indent=2, default=str) + "\n"
    )
    for path, text in prepared:
        path.write_text(text)
    print(f"{len(prepared)} YAMLs simplified with exact resolved equivalence")


if __name__ == "__main__":
    main()
