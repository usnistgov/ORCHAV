"""CPU-only reader for conditional detection outcomes on an authored world grid.

One noiseless trial per cell is a detection outcome, not an estimate of detection
probability. Unmeasured and excluded cells remain unknown, never misses.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
from math import prod
from pathlib import Path

import numpy as np

MAX_VOLUME_CELLS = 2_000_000
MAX_MANIFEST_BYTES = 64 * 1024 * 1024


class SensingVolume:
    """Validate compact outcomes; native recordings are opened only on request."""

    def __init__(self, path):
        self.path = Path(path).resolve()
        if self.path.is_dir():
            self.path /= "sensing_volume.json"
        if self.path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Sensing volume manifest exceeds 64 MiB")
        raw = self.path.read_bytes()
        self.identity = hashlib.sha256(raw).hexdigest()
        self.metadata = json.loads(raw)
        data = self.metadata
        assets = data.get("scene_asset_hashes")
        if assets is not None and (
            not isinstance(assets, dict)
            or not assets
            or any(
                not isinstance(name, str)
                or not name
                or Path(name).is_absolute()
                or ".." in Path(name).parts
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in assets.items()
            )
        ):
            raise ValueError("Scene asset bindings require relative paths and SHA-256 digests")
        self.is_fusion = data.get("kind") == "conditional_fusion_outcomes"
        if data.get("schema_version") != 1 or data.get("kind") not in {
            "conditional_detection_outcomes",
            "conditional_fusion_outcomes",
        }:
            raise ValueError("Unsupported sensing volume contract")
        if self.is_fusion:
            from shared.sensing.volume.fusion import validate_fusion_metadata

            validate_fusion_metadata(data)
        frame = data.get("world_frame", {})
        if (
            not frame.get("id")
            or not frame.get("origin")
            or frame.get("axes") != ["x", "y", "z"]
            or frame.get("handedness") != "right"
            or frame.get("length_unit") != "m"
        ):
            raise ValueError("Declare a common right-handed world frame in metres")
        self.axes = {}
        for name in ("x", "y", "z"):
            axis = np.asarray(data["axes_m"][name], dtype=float)
            if axis.ndim != 1 or len(axis) < 2 or not np.all(np.isfinite(axis)):
                raise ValueError("Volume axes require at least two finite sample centres")
            steps = np.diff(axis)
            if np.any(steps <= 0) or not np.allclose(steps, steps[0], rtol=1e-9, atol=1e-9):
                raise ValueError("Volume rendering requires an increasing regular grid")
            axis.setflags(write=False)
            self.axes[name] = axis
        self.shape = tuple(len(self.axes[name]) for name in ("z", "y", "x"))
        if prod(self.shape) * max(1, len(data.get("headings", []))) > MAX_VOLUME_CELLS:
            raise ValueError("Sensing volume exceeds the two-million-cell display budget")
        self.headings = data["headings"]
        if not self.headings:
            raise ValueError("Declare at least one flight heading")
        for heading in self.headings:
            velocity = np.asarray(heading["velocity_m_s"], dtype=float)
            if (
                not heading.get("name")
                or velocity.shape != (3,)
                or not np.all(np.isfinite(velocity))
            ):
                raise ValueError("Invalid heading")
        sensor = np.asarray(data["sensor_position_m"], dtype=float)
        if sensor.shape != (3,) or not np.all(np.isfinite(sensor)):
            raise ValueError("Invalid sensor position")
        gate = data["position_gate_m"]
        if isinstance(gate, bool) or not np.isfinite(gate) or gate <= 0:
            raise ValueError("Declare a positive spatial matching gate")
        if "cpi_samples" in data and (
            type(data["cpi_samples"]) is not int or data["cpi_samples"] < 1
        ):
            raise ValueError("CPI sample count must be a positive integer")
        self.cells = {}
        trial_ids = set()
        for trial in data["trials"]:
            trial_id = trial.get("id")
            if not isinstance(trial_id, str) or not trial_id or trial_id in trial_ids:
                raise ValueError("Trial IDs must be nonempty and unique across the volume")
            trial_ids.add(trial_id)
            index = trial["index_zyx"]
            heading = trial["heading"]
            if (
                type(heading) is not int
                or not 0 <= heading < len(self.headings)
                or len(index) != 3
                or any(
                    type(i) is not int or not 0 <= i < n
                    for i, n in zip(index, self.shape, strict=True)
                )
            ):
                raise ValueError("Trial index is outside the authored volume")
            key = (heading, *index)
            if key in self.cells:
                raise ValueError("Duplicate position/heading trial")
            position = np.array(
                [self.axes[name][i] for name, i in zip(("x", "y", "z"), index[::-1], strict=True)]
            )
            velocity = self.headings[heading]["velocity_m_s"]
            if not np.array_equal(trial["position_m"], position) or not np.array_equal(
                trial["velocity_m_s"], velocity
            ):
                raise ValueError("Trial position or motion disagrees with its grid binding")
            status = trial["status"]
            if status not in {"pending", "excluded_building", "excluded_geometry", "measured"}:
                raise ValueError("Unknown trial status")
            if "direct_path_blocked" in trial and type(trial["direct_path_blocked"]) is not bool:
                raise ValueError("Direct-path geometry must be a boolean, not an outcome score")
            if status == "measured":
                if type(trial.get("hit")) is not bool:
                    raise ValueError("A deterministic outcome must be a boolean, not a probability")
                if (trial.get("tp"), trial.get("fn")) != (int(trial["hit"]), int(not trial["hit"])):
                    raise ValueError("Outcome and TP/FN disagree")
                if type(trial.get("fp")) is not int or trial["fp"] < 0:
                    raise ValueError("Invalid false-output count")
                if self.is_fusion:
                    if (
                        trial.get("radial_closing_velocity_m_s") is not None
                        or "reports" not in trial
                    ):
                        raise ValueError(
                            "Fusion cells need saved estimates and no single-sensor radial velocity"
                        )
                    if any(
                        r.get("range_m") is not None or r.get("closing_velocity_m_s") is not None
                        for r in trial["reports"]
                    ):
                        raise ValueError("Fused estimates cannot carry single-sensor measurements")
                else:
                    delta = position - sensor
                    if np.linalg.norm(delta) == 0:
                        raise ValueError("Undefined radial projection at the sensor")
                    radial = -float(np.dot(velocity, delta / np.linalg.norm(delta)))
                    if not np.isclose(
                        trial["radial_closing_velocity_m_s"], radial, rtol=1e-9, atol=1e-9
                    ):
                        raise ValueError(
                            "Recorded radial velocity disagrees with declared geometry"
                        )
                error = trial.get("matched_error_m")
                if trial["hit"] and (
                    error is None or not np.isfinite(error) or not 0 <= error <= gate
                ):
                    raise ValueError(
                        "A hit requires a finite matched error within the declared gate"
                    )
                if not trial.get("native") or not trial.get("source_hashes"):
                    raise ValueError("Measured cells require native evidence and hashes")
            availability = trial.get("native_evidence")
            if availability not in {None, "included", "not_included"}:
                raise ValueError("Unsupported native-evidence availability")
            if availability == "not_included" and "reports" not in trial:
                raise ValueError("Omitted native evidence requires authoritative embedded reports")
            self.cells[key] = trial

    @property
    def scene_path(self):
        return (self.path.parent / self.metadata["scene_scenario"]).resolve()

    def verify_scene(self):
        raw = self.scene_path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if digest != self.metadata.get("scene_configuration_sha256"):
            raise ValueError("Volume scene configuration is not the declared world binding")
        local = self.metadata.get("local_scene_assets")
        if local:
            import yaml

            root = (self.path.parent / local["root"]).resolve()
            config = yaml.safe_load(raw)
            scene = config.get("scene", {})
            scene_path = (self.scene_path.parent / scene.get("id", "")).resolve()
            if scene.get("source") != "local" or not scene_path.is_relative_to(root):
                raise ValueError("Local scene is not inside the bound asset directory")
            if str(scene_path.relative_to(root)) not in local["hashes"]:
                raise ValueError("Local scene XML is missing its asset hash")
            for relative, expected in local["hashes"].items():
                asset = (root / relative).resolve()
                if not asset.is_relative_to(root):
                    raise ValueError("Local asset escapes its declared directory")
                with asset.open("rb") as stream:
                    if hashlib.file_digest(stream, "sha256").hexdigest() != expected:
                        raise ValueError(f"Local scene asset changed: {relative}")
        assets = self.metadata.get("scene_asset_hashes")
        if assets:
            import yaml

            config = yaml.safe_load(raw)
            scene = config.get("scene") if isinstance(config, dict) else None
            name = scene.get("id") if isinstance(scene, dict) else None
            if (
                not isinstance(name, str)
                or not name.isidentifier()
                or scene.get("source") != "sionna"
            ):
                raise ValueError("Installed scene asset hashes require a named Sionna scene")
            package = importlib.util.find_spec("sionna")
            if package is None or not package.submodule_search_locations:
                raise ValueError("Install the bound Sionna scene to inspect this volume")
            directory = Path(next(iter(package.submodule_search_locations))) / "rt/scenes" / name
            for relative, expected in assets.items():
                path = (directory / relative).resolve()
                if not path.is_relative_to(directory.resolve()):
                    raise ValueError("Scene asset paths must remain inside the installed scene")
                with path.open("rb") as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"Installed scene asset changed: {relative}")
        return self.scene_path

    def outcome_grid(self, heading):
        """Return NaN for unknown/excluded, zero for miss and one for hit."""
        if not 0 <= heading < len(self.headings):
            raise IndexError("Unknown flight heading")
        values = np.full(self.shape, np.nan, dtype=np.float32)
        for (ih, iz, iy, ix), trial in self.cells.items():
            if ih == heading and trial["status"] == "measured":
                values[iz, iy, ix] = float(trial["hit"])
        values.setflags(write=False)
        return values

    def native_availability(self, key):
        """Availability is distinct from the retained trial's measurement status."""
        trial = self.cells.get(key)
        if trial is None or trial["status"] != "measured":
            return "No measured native evidence"
        if trial.get("native_evidence") == "not_included":
            return "Native evidence not included; compact saved reports remain available"
        source = self.path.parent / trial["native"]
        if self.is_fusion:
            from shared.sensing.volume.fusion import BUNDLE_FILES

            return (
                "available"
                if all((source / name).is_file() for name in BUNDLE_FILES)
                else "Fusion replay unavailable at its declared path"
            )
        if not (source / "scenario.yaml").is_file():
            return "Native recording unavailable at its declared path"
        # Check the declared inventory only for this cell. Do not read HDF5 or
        # hash the entire campaign just to decide whether opening is available.
        from shared.sensing.loader import _resolve_frames_dir

        try:
            frames = _resolve_frames_dir(source)
        except FileNotFoundError:
            return "Native frame manifest missing; saved volume reports remain available"
        if any(not (frames / name).is_file() for name in trial["source_hashes"]):
            return "Native frame files missing; saved volume reports remain available"
        return "available"

    def verify_native(self, key):
        """Verify the selected native recording before opening its evidence."""
        from shared.sensing.fusion.experiment import fingerprints
        from shared.sensing.loader import _resolve_frames_dir

        trial = self.cells[key]
        if trial["status"] != "measured":
            raise ValueError("This cell has no measured native evidence")
        if trial.get("native_evidence") == "not_included":
            raise ValueError("Native evidence not included in this compact package")
        if self.is_fusion:
            from shared.sensing.volume.fusion import verify_fusion_trial

            return verify_fusion_trial(self, key)
        source = (self.path.parent / trial["native"]).resolve()
        if fingerprints(_resolve_frames_dir(source)) != trial["source_hashes"]:
            raise ValueError("Native cell source hash mismatch")
        config = hashlib.sha256((source / "scenario.yaml").read_bytes()).hexdigest()
        if config != trial["configuration_sha256"]:
            raise ValueError("Native trial configuration changed")
        return source

    def geometry_grid(self, heading):
        """Return declared centre-ray blockage independently of detection outcomes."""
        if not 0 <= heading < len(self.headings):
            raise IndexError("Unknown flight heading")
        values = np.full(self.shape, np.nan, dtype=np.float32)
        for (ih, iz, iy, ix), trial in self.cells.items():
            if ih == heading and trial["status"] in {"pending", "measured"}:
                if "direct_path_blocked" in trial:
                    values[iz, iy, ix] = float(trial["direct_path_blocked"])
        values.setflags(write=False)
        return values
