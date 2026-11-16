#!/usr/bin/env python3
"""Prepare a new, portable wall/open inspection subset from saved acquisitions.

No RT or detector runs. Every retained compact report is re-read from its
verified native source. Only path strings in copied native JSON metadata change;
all source archives and all numerical datasets remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from shared.sensing.fusion.experiment import fingerprints, load_sensor
from shared.sensing.volume.evaluation import VolumeEvaluation, saved_reports
from shared.sensing.volume.inspection import PairedVolume, context_snapshot, save_bookmark
from shared.sensing.volume.reader import SensingVolume


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def portable_native(source, destination, assets, condition):
    """Copy evidence; make only declared asset references relative in the copy."""
    import h5py

    shutil.copytree(source, destination)
    config = yaml.safe_load((source / "scenario.yaml").read_text())
    scene_original = config["scene"]["id"]
    target_original = config["actors"]["targets"][0]["asset"]["path"]
    scene_relative = os.path.relpath(assets / "scene" / f"{condition}.xml", destination)
    target_relative = os.path.relpath(assets / "cube", destination)
    replacements = {scene_original: scene_relative, target_original: target_relative}
    config["scene"]["id"] = scene_relative
    config["actors"]["targets"][0]["asset"]["path"] = target_relative
    (destination / "scenario.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    # Asset paths participate in the content-addressed target catalog. Rebind
    # its revision and every reference in copied metadata, keeping semantic IDs.
    from shared.sensing.targets import build_target_catalog

    for path in (destination / "frames").glob("*.h5"):
        with h5py.File(path) as f:
            if "static/sensing_config_json" in f:
                payload = json.loads(f["static/sensing_config_json"][()].decode())
                catalog = payload.get("target_catalog") if isinstance(payload, dict) else None
                if catalog:
                    encoded = json.dumps(catalog["targets"])
                    for old, new in replacements.items():
                        encoded = encoded.replace(old, new)
                    rebound = build_target_catalog(json.loads(encoded))
                    replacements[catalog["catalog_revision"]] = rebound["catalog_revision"]
    changed = []
    for path in (destination / "frames").glob("*.h5"):
        with h5py.File(path, "r+") as f:

            def rewrite(name, obj):
                if not isinstance(obj, h5py.Dataset) or obj.dtype.kind not in ("O", "S", "U"):
                    return
                before = obj[()]

                def convert(value):
                    text = value.decode() if isinstance(value, bytes) else str(value)
                    for old, new in replacements.items():
                        text = text.replace(old, new)
                    if "/mnt/" in text or "/home/" in text:
                        raise ValueError(f"Unmapped private path in {name}")
                    return text

                after = (
                    np.vectorize(convert, otypes=[object])(before)
                    if np.ndim(before)
                    else convert(before)
                )
                # These are existing variable-length JSON datasets. Numeric
                # arrays and native path evidence are never rewritten.
                old_text = str(before)
                new_text = str(after)
                if old_text != new_text and any(old in old_text for old in replacements):
                    obj[...] = after
                    changed.append(path.name + ":" + name)

            f.visititems(rewrite)
    # The frame manifest repeats producer configuration for orientation and
    # provenance; its scene reference must relocate with the copied recording.
    manifest = destination / "frames" / "frames_manifest.json"
    if manifest.is_file():
        text = manifest.read_text()
        for old, new in replacements.items():
            text = text.replace(old, new)
        if "/mnt/" in text or "/home/" in text:
            raise ValueError("Unmapped private path in frame manifest")
        manifest.write_text(text)
        changed.append("frames_manifest.json:asset_references")
    return changed


def package(wall, opened, destination):
    destination = destination.resolve()
    if destination.exists():
        raise ValueError("Destination must be new; originals are never pruned")
    staging = destination.with_name(destination.name + ".partial")
    staging.mkdir(parents=True, exist_ok=False)
    assets = staging / "assets"
    assets.mkdir()
    shutil.copytree(
        ROOT / "scenarios/sensing/applications/wall_detection_volume/scene", assets / "scene"
    )
    shutil.copytree(ROOT / "libraries/targets/cube", assets / "cube")
    for name in ("LICENSE", "NOTICE", "NIST_SOFTWARE_DISCLAIMER.md"):
        shutil.copy2(ROOT / name, staging / name)
    readers = [SensingVolume(wall), SensingVolume(opened)]
    PairedVolume(*readers)
    inventory = []
    bookmarks = []
    native_positions = {(-2.0, 0.0, 2.5), (2.0, 0.0, 2.5), (2.0, 2.0, 4.5)}
    for reader in readers:
        condition = reader.metadata["condition"]
        out = staging / condition
        out.mkdir()
        data = deepcopy(reader.metadata)
        data["axes_m"] = {n: list(map(float, reader.axes[n][::2])) for n in ("x", "y", "z")}
        data["derived_from"] = {
            "volume_sha256": reader.identity,
            "selection": "Every second original centre on each axis; no interpolation or new acquisition",
        }
        cfg = yaml.safe_load(reader.scene_path.read_text())
        cfg["scene"]["id"] = f"../assets/scene/{condition}.xml"
        (out / "scenario.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        data["scene_configuration_sha256"] = sha(out / "scenario.yaml")
        data["scene_scenario"] = "scenario.yaml"
        data["local_scene_assets"] = {
            "root": "../assets",
            "hashes": {
                str(p.relative_to(assets)): sha(p) for p in assets.rglob("*") if p.is_file()
            },
        }
        trials = []
        rows = {}
        target_contract = None
        selected = [(key, t) for key, t in reader.cells.items() if all(i % 2 == 0 for i in key[1:])]
        for index, (key, original) in enumerate(selected):
            trial = deepcopy(original)
            trial["index_zyx"] = [i // 2 for i in key[1:]]
            if trial["status"] == "measured":
                source = reader.path.parent / trial["native"]
                sensor = load_sensor(
                    source,
                    reader.metadata.get("sensor_id", "Volume"),
                    result_frame_ids=(reader.metadata["cpi_samples"] - 1,),
                )
                if (
                    sensor["source"]["hashes"] != trial["source_hashes"]
                    or sha(source / "scenario.yaml") != trial["configuration_sha256"]
                ):
                    raise ValueError("Original native hash changed")
                truth = sensor["truth"]
                if (
                    len(truth) != 1
                    or len(truth[0]["targets"]) != 1
                    or not np.allclose(
                        truth[0]["targets"][0]["position_m"], trial["position_m"], atol=1e-5
                    )
                ):
                    raise ValueError("Native truth binding changed")
                trial["reports"] = saved_reports(sensor)
                rows[trial["id"]] = trial["reports"]
                target = yaml.safe_load((source / "scenario.yaml").read_text())["actors"][
                    "targets"
                ][0]
                target_contract = {k: v for k, v in target.items() if k != "mobility"}
                target_contract["asset"]["path"] = "../assets/cube"
                include = tuple(trial["position_m"]) in native_positions
                trial["native_evidence"] = "included" if include else "not_included"
                trial["original_source"] = {
                    "configuration_sha256": trial["configuration_sha256"],
                    "source_hashes": trial["source_hashes"],
                }
                trial["native"] = "native/" + trial["id"]
                changed = []
                if include:
                    changed = portable_native(source, out / trial["native"], assets, condition)
                    trial["source_hashes"] = fingerprints(out / trial["native"] / "frames")
                    trial["configuration_sha256"] = sha(out / trial["native"] / "scenario.yaml")
                inventory.append(
                    dict(
                        condition=condition,
                        trial_id=trial["id"],
                        native_evidence=trial["native_evidence"],
                        rewritten_path_metadata=changed,
                    )
                )
            trials.append(trial)
            if index % 50 == 0:
                print(condition, index, len(selected), flush=True)
        data["trials"] = trials
        data["inspection_target"] = target_contract
        (out / "sensing_volume.json").write_text(
            json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n"
        )
        packaged = SensingVolume(out)
        packaged.verify_scene()
        VolumeEvaluation(packaged, rows)
        # Authored cases are exact existing sample coordinates; titles do not
        # assert a detection outcome that has not been evaluated.
        locations = [
            ("before_wall", [-2, 0, 2.5]),
            ("behind_wall", [2, 0, 2.5]),
            ("beside_edge", [2, 3, 2.5]),
            ("above_wall", [2, 0, 5.5]),
        ]
        extra = next((t for t in trials if t.get("fp", 0) > 0), None)
        if extra:
            locations.append(("extra_reports", extra["position_m"]))
        for name, position in locations:
            found = next(
                (key for key, t in packaged.cells.items() if t["position_m"] == position), None
            )
            if found is None:
                continue
            view = dict(
                quantity="details",
                mode="slice",
                altitude=position[2],
                x=position[0],
                y=position[1],
                opacity=0.6,
                visible=True,
                inspect=True,
                inspection={"vectors": True, "orientation": "xy"},
                evaluation=dict(
                    basis="position",
                    position_gate=0.75,
                    range_gate=0.75,
                    velocity_gate=0.5,
                    error_max=2,
                    expanded=False,
                ),
            )
            state = context_snapshot(packaged, 0, found, view)
            directory = staging / "bookmarks"
            directory.mkdir(exist_ok=True)
            filename = directory / f"{condition}_{name}.json"
            row = packaged.cells[found]
            save_bookmark(
                filename,
                state,
                title=condition + ": " + name.replace("_", " "),
                description=f"Recorded tested position {position}; baseline TP {row.get('tp',0)}, FP {row.get('fp',0)}, FN {row.get('fn',0)}.",
            )
            bookmarks.append(str(filename.relative_to(staging)))
    PairedVolume(SensingVolume(staging / "wall"), SensingVolume(staging / "open"))
    (staging / "evidence_inventory.json").write_text(
        json.dumps(
            dict(
                schema_version=1,
                derivation="Original numerical HDF5 evidence preserved; asset-path strings and their target-catalog identity references in copied JSON metadata/configurations rebound. Original and packaged hashes are separately retained.",
                trials=inventory,
                bookmarks=bookmarks,
            ),
            indent=2,
        )
        + "\n"
    )
    (staging / "README.md").write_text("""# Saved wall/open inspection example

Open `wall` with `python -m visualizer --renderer pygfx --sensing-volume <package>/wall`.
No generation is required. Interactive 3D requires the supported graphics environment.

This package retains every second original grid centre (1 m spacing), with all
saved detector reports at those centres embedded in each manifest. It is a subset
of the original experiment, not a new or interpolated campaign. There is one
acquisition per position, not an empirical detection probability.

Use Colour by → Outcome detail or Paired change. Enable Inspect a sampled
position to see actual reports, then Open trial recording for included evidence.
Return to volume preserves the investigation. Focus / bookmarks → Open bookmark
loads the authored exact before/behind/edge/above-wall examples.

Six representative native recordings are included (three per condition).
Other measured trials remain browsable and rescorable; their native action says
“not included”. See evidence_inventory.json for exact availability. No original
archive was pruned. Copied native numerical datasets are unchanged; asset paths
in copied configurations and JSON metadata are made relative, with their
content-addressed target-catalog revision references updated. Separate original
and derived source hashes record this relocation.

Assets: ORCHAV-authored cube, planar wall and scene XML from the source checkout;
no OSM, Paris, third-party UAV mesh or external data. See LICENSE, NOTICE and
NIST_SOFTWARE_DISCLAIMER.md. This locally prepared example is not a publication.
""")
    sizes = {
        "files": sum(p.is_file() for p in staging.rglob("*")),
        "bytes": sum(p.stat().st_size for p in staging.rglob("*") if p.is_file()),
        "included_native": sum(t["native_evidence"] == "included" for t in inventory),
        "omitted_native": sum(t["native_evidence"] == "not_included" for t in inventory),
    }
    (staging / "package_summary.json").write_text(json.dumps(sizes, indent=2) + "\n")
    # Include this small summary itself in the as-packaged size and file count.
    summary = staging / "package_summary.json"
    for _ in range(3):
        sizes["files"] = sum(p.is_file() for p in staging.rglob("*"))
        sizes["bytes"] = sum(p.stat().st_size for p in staging.rglob("*") if p.is_file())
        summary.write_text(json.dumps(sizes, indent=2) + "\n")
    staging.rename(destination)
    print(json.dumps(sizes), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("wall", type=Path)
    p.add_argument("open", type=Path)
    p.add_argument("destination", type=Path)
    args = p.parse_args()
    package(args.wall, args.open, args.destination)
