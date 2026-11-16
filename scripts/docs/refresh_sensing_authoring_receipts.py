from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]

RECORDING_BINDINGS = "docs/assets/sensing_recording/authoring_bindings.json"

METRIC_DEFAULT_BINDINGS = "docs/assets/sensing_recording/metric_default_bindings.json"

SOURCE_RELOCATION_BINDINGS = "docs/assets/sensing_recording/source_relocation_bindings.json"

SOURCE_RELOCATION_MODULES = {
    "shared.sensing_loader": "shared.sensing.loader",
    "shared.sensing_range_response": "shared.sensing.range_response",
    "shared.sensing_viz": "shared.sensing.plotting.viz",
}

SOURCE_RELOCATION_PATHS = {
    "shared/sensing_viz.py": "shared/sensing/plotting/viz.py",
    "shared/sensing_figure_style.py": "shared/sensing/plotting/style.py",
}


def _metric_default_document(source: bytes, default: bool) -> dict:
    """Resolve only the metric flag, preserving every other authored value."""
    document = yaml.safe_load(source)
    if not isinstance(document, dict):
        raise ValueError("Metric default binding requires a YAML document mapping")
    raytracing = document.get("raytracing", {})
    if not isinstance(raytracing, dict):
        raise ValueError("Metric default binding requires a raytracing mapping")
    value = raytracing.get("export_path_metrics", default)
    if type(value) is not bool:
        raise ValueError("Metric default binding requires a boolean export_path_metrics")
    document["raytracing"] = {**raytracing, "export_path_metrics": value}
    return document


def _bound_retention_profile_bytes(current: bytes, name: str, root: Path) -> bytes:
    """Authenticate reprocessing-to-maps retention without relabelling old results."""
    inventory = root / RECORDING_BINDINGS
    if not inventory.is_file():
        return current
    document = json.loads(inventory.read_text())
    if document.get("schema") != "orchav.recording-authoring-bindings.v1":
        raise ValueError("Unsupported recording authoring binding")
    row = document.get("retention_profiles", {}).get(name)
    if row is None:
        return current
    if hashlib.sha256(current).hexdigest() != row["current_authored_sha256"]:
        raise ValueError(f"Current retention-profile input changed: {name}")
    current_block = row["current_recording_block"].encode()
    previous_block = row["previous_recording_block"].encode()
    if not current_block or current.count(current_block) != 1:
        raise ValueError(f"Retention profile block is not uniquely identified: {name}")
    previous = current.replace(current_block, previous_block, 1)
    if hashlib.sha256(previous).hexdigest() != row["previous_authored_sha256"]:
        raise ValueError(f"Previous retention-profile input identity mismatch: {name}")
    old, new = yaml.safe_load(previous), yaml.safe_load(current)
    for resolved, profile in ((old, "reprocessing"), (new, "maps")):
        sensing = resolved.get("sensing") if isinstance(resolved, dict) else None
        recording = sensing.get("recording") if isinstance(sensing, dict) else None
        if not isinstance(recording, dict) or recording.pop("profile", None) != profile:
            raise ValueError(f"Unsupported retention profile transition: {name}")
    encoded_old = json.dumps(old, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    encoded_new = json.dumps(new, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (
        encoded_old != encoded_new
        or hashlib.sha256(encoded_new).hexdigest() != row["non_profile_sha256"]
    ):
        raise ValueError(f"Configuration changed outside the retention profile: {name}")
    return previous


def _bound_metric_default_bytes(current: bytes, name: str, root: Path) -> bytes:
    """Authenticate an exact metric-default spelling edit before storage bindings."""
    inventory = root / METRIC_DEFAULT_BINDINGS
    if not inventory.is_file():
        return current
    document = json.loads(inventory.read_text())
    if document.get("schema") != "orchav.metric-default-authoring-bindings.v1":
        raise ValueError("Unsupported metric default authoring binding")
    row = document["scenarios"].get(name)
    if row is None:
        return current
    if hashlib.sha256(current).hexdigest() != row["current_authored_sha256"]:
        raise ValueError(f"Current metric-default input changed: {name}")
    current_block = row["current_block"].encode()
    previous_block = row["previous_block"].encode()
    if not current_block or current.count(current_block) != 1:
        raise ValueError(f"Metric default block is not uniquely identified: {name}")
    previous = current.replace(current_block, previous_block, 1)
    if hashlib.sha256(previous).hexdigest() != row["previous_authored_sha256"]:
        raise ValueError(f"Previous metric-default input identity mismatch: {name}")
    old = _metric_default_document(previous, False)
    new = _metric_default_document(current, True)
    old_authored = yaml.safe_load(previous).get("raytracing", {})
    new_authored = yaml.safe_load(current).get("raytracing", {})
    if not (
        (
            old_authored.get("export_path_metrics") is True
            and "export_path_metrics" not in new_authored
        )
        or (
            "export_path_metrics" not in old_authored
            and new_authored.get("export_path_metrics") is False
        )
    ):
        raise ValueError(f"Unsupported metric default spelling change: {name}")
    encoded_old = json.dumps(old, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    encoded_new = json.dumps(new, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if (
        encoded_old != encoded_new
        or hashlib.sha256(encoded_new).hexdigest() != row["resolved_document_sha256"]
    ):
        raise ValueError(f"Scientific configuration changed in metric default binding: {name}")
    return previous


def _bound_source_relocation_bytes(current: bytes, name: str, root: Path) -> bytes:
    """Recover exact prior bytes only for declared imports and source locators."""
    inventory = root / SOURCE_RELOCATION_BINDINGS
    if not inventory.is_file():
        return current
    document = json.loads(inventory.read_text())
    if document.get("schema") != "orchav.sensing-source-relocation-bindings.v1":
        raise ValueError("Unsupported source relocation binding")
    row = document["sources"].get(name)
    if row is None:
        return current
    if hashlib.sha256(current).hexdigest() != row["current_authored_sha256"]:
        raise ValueError(f"Current source input changed: {name}")
    previous = current
    for replacement in row["replacements"]:
        current_block = replacement["current"].encode()
        previous_block = replacement["previous"].encode()
        if not current_block or previous.count(current_block) != 1:
            raise ValueError(f"Source relocation block is not uniquely identified: {name}")
        previous = previous.replace(current_block, previous_block, 1)
    if hashlib.sha256(previous).hexdigest() != row["previous_authored_sha256"]:
        raise ValueError(f"Previous source input identity mismatch: {name}")
    old, new = ast.parse(previous, filename=name), ast.parse(current, filename=name)
    for node in ast.walk(old):
        if isinstance(node, ast.ImportFrom) and node.level == 0:
            node.module = SOURCE_RELOCATION_MODULES.get(node.module, node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in SOURCE_RELOCATION_MODULES:
                    if alias.asname is None:
                        raise ValueError(f"Unaliased module binding cannot be relocated: {name}")
                    alias.name = SOURCE_RELOCATION_MODULES[alias.name]
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            node.value = SOURCE_RELOCATION_PATHS.get(node.value, node.value)
    if ast.dump(old) != ast.dump(new):
        raise ValueError(f"Source relocation changes scientific statements: {name}")
    return previous


def _bound_recording_bytes(current: bytes, name: str, root: Path) -> bytes:
    """Verify the existing storage-only binding against its exact authored input."""
    inventory = root / RECORDING_BINDINGS
    if not inventory.is_file():
        return current
    document = json.loads(inventory.read_text())
    if document.get("schema") != "orchav.recording-authoring-bindings.v1":
        raise ValueError("Unsupported recording authoring binding")
    row = document["scenarios"].get(name)
    if row is None:
        return current
    if hashlib.sha256(current).hexdigest() != row["current_authored_sha256"]:
        raise ValueError(f"Current authored input changed: {name}")
    current_block = row["current_recording_block"].encode()
    previous_block = row["previous_recording_block"].encode()
    if current.count(current_block) != 1:
        raise ValueError(f"Recording block is not uniquely identified: {name}")
    previous = current.replace(current_block, previous_block, 1)
    if hashlib.sha256(previous).hexdigest() != row["previous_authored_sha256"]:
        raise ValueError(f"Previous authored input identity mismatch: {name}")
    old, new = yaml.safe_load(previous), yaml.safe_load(current)
    old["sensing"].pop("recording")
    new["sensing"].pop("recording")
    encoded = json.dumps(new, sort_keys=True, separators=(",", ":")).encode()
    if old != new or hashlib.sha256(encoded).hexdigest() != row["non_recording_sha256"]:
        raise ValueError(f"Scientific configuration changed: {name}")
    return previous


def authenticated_authoring_inputs(path: Path, root: Path = ROOT) -> tuple[bytes, ...]:
    """Return exact authored stages, each verified by the declared binding chain.

    Receipts may identify any stage of authenticated retention-profile changes,
    equivalent metric-default spelling, storage-only YAML edits, or explicit
    Python import/source relocations.
    No image, acquisition, scientific statement, or configuration is relabelled
    as a fresh result.
    """
    current = path.read_bytes()
    try:
        name = path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return (current,)
    if path.suffix == ".py":
        stages = (current, _bound_source_relocation_bytes(current, name, root))
    else:
        retention_input = _bound_retention_profile_bytes(current, name, root)
        metric_input = _bound_metric_default_bytes(retention_input, name, root)
        stages = (
            current,
            retention_input,
            metric_input,
            _bound_recording_bytes(metric_input, name, root),
        )
    return tuple(dict.fromkeys(stages))


def bound_authoring_bytes(path: Path, root: Path = ROOT) -> bytes:
    """Read the earliest exact authored input authenticated by declared bindings."""
    return authenticated_authoring_inputs(path, root)[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify equivalent authoring and source-relocation bindings (read-only)."
    )
    parser.add_argument("--check-recording-bindings", action="store_true")
    parser.add_argument("--check-source-bindings", action="store_true")
    parser.add_argument("--check-metric-default-bindings", action="store_true")
    parser.parse_args()
    count = 0
    for inventory, key in (
        (RECORDING_BINDINGS, "scenarios"),
        (RECORDING_BINDINGS, "retention_profiles"),
        (SOURCE_RELOCATION_BINDINGS, "sources"),
        (METRIC_DEFAULT_BINDINGS, "scenarios"),
    ):
        entries = json.loads((ROOT / inventory).read_text()).get(key, {})
        for name in entries:
            bound_authoring_bytes(ROOT / name)
            count += 1
    print(f"{count} non-scientific authoring bindings verified")
