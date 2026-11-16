"""Validated CPU-only fusion reader and indexed, atomic JSON result bundles.

Compact states/observations are loaded once. Candidate diagnostics are fetched
per epoch from a separately indexed JSONL file, including integrity checks.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Mapping
from copy import deepcopy
from functools import cached_property
from pathlib import Path

import numpy as np

KIND = "synchronized_monostatic_detection_fusion"
NOT_RECORDED = "not recorded"
DECISION_KEYS = ("decisions", "geometry_rejections", "suppressed_births", "unsupported_births")


def validate_alignment(recipe):
    """Explicit common-frame declaration; units/truth alone never prove alignment."""
    frame = recipe.get("world_frame", {})
    if (
        not isinstance(frame.get("id"), str)
        or not frame["id"].strip()
        or not isinstance(frame.get("origin"), str)
        or not frame["origin"].strip()
        or frame.get("axes") != ["x", "y", "z"]
        or frame.get("handedness") != "right"
        or frame.get("length_unit") != "m"
        or frame.get("angle_unit") != "rad"
    ):
        raise ValueError(
            "Declare world_frame identity, origin, xyz axes, right handedness and m/rad units"
        )
    for spec in recipe["sensors"].values():
        if (
            spec.get("world_frame_id") != frame["id"]
            or spec.get("alignment") != "explicit_shared_world"
        ):
            raise ValueError(
                "Every source needs explicit_shared_world alignment and world_frame_id"
            )
    return deepcopy(frame)


def code_identity(root):
    """Fingerprint local implementation, including edits not represented by HEAD."""
    root = Path(root)
    digest = hashlib.sha256()
    for directory in ("shared", "visualizer", "generator"):
        for path in sorted((root / directory).rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    try:
        head = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        head = NOT_RECORDED
    return {
        "head": head,
        "implementation_sha256": digest.hexdigest(),
        "fingerprint_scope": "all Python files in shared, visualizer, generator; includes uncommitted files",
    }


class FusionResult(Mapping):
    """One interpretation shared by evaluation, figures and native inspection."""

    def __init__(self, source):
        self.path = None
        if isinstance(source, (str, Path)):
            path = Path(source).resolve()
            self.path = path / "fusion_results.json" if path.is_dir() else path
            self._data = json.loads(self.path.read_text())
        elif isinstance(source, FusionResult):
            self.path, self._data = source.path, source._data
            self.version, self.times = source.version, source.times
            return
        else:
            self._data = deepcopy(source)
        self.version = self._data.get("schema_version", 1)
        if self.version not in (1, 2):
            raise ValueError("Unsupported fusion result schema")
        self._validate()

    def _validate(self):
        data = self._data
        if self.version == 2:
            if data.get("kind") != KIND or not data.get("run_id"):
                raise ValueError("Invalid fusion result identity")
            validate_alignment({"world_frame": data["world_frame"], "sensors": data["sources"]})
            if data.get("state_order") != ["x", "y", "z", "vx", "vy", "vz"]:
                raise ValueError("Unsupported fusion state order")
        reference = None
        for history in data["histories"].values():
            times = np.asarray([e["time_s"] for e in history], dtype=float)
            if not np.all(np.isfinite(times)) or not np.all(np.diff(times) > 0):
                raise ValueError("Result epochs must be finite and strictly increasing")
            if reference is not None and not np.array_equal(reference, times):
                raise ValueError("Result histories need the same epoch index")
            reference = times
            for epoch in history:
                ids = [t["track_id"] for t in epoch["tracks"]]
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate track identity in epoch")
                for track in epoch["tracks"]:
                    for key in ("position_m", "velocity_m_s"):
                        array = np.asarray(track[key])
                        if array.shape != (3,) or not np.all(np.isfinite(array)):
                            raise ValueError("Invalid saved state")
                    if "covariance" in track:
                        cov = np.asarray(track["covariance"])
                        if (
                            cov.shape != (6, 6)
                            or not np.all(np.isfinite(cov))
                            or not np.allclose(cov, cov.T, atol=1e-10, rtol=0)
                            or np.linalg.eigvalsh(cov).min() < -1e-9
                        ):
                            raise ValueError("Invalid saved covariance")
        self.times = np.array([]) if reference is None else reference
        self.times.setflags(write=False)
        if self.version == 2:
            for mode, entries in data.get("decision_index", {}).items():
                if mode not in data["histories"] or len(entries) != len(self.times):
                    raise ValueError("Invalid decision index")
                for entry in entries:
                    if (
                        type(entry["offset"]) is not int
                        or entry["offset"] < 0
                        or type(entry["length"]) is not int
                        or entry["length"] <= 0
                    ):
                        raise ValueError("Invalid decision offsets")

    def __getitem__(self, key):
        return self._data[key]

    def __iter__(self):
        return iter(self._data)

    def __len__(self):
        return len(self._data)

    @cached_property
    def inspection_id(self):
        """Stable selection scope even when a v1 result did not record a run ID."""
        return (
            self._data.get("run_id")
            or "legacy:"
            + hashlib.sha256(json.dumps(self._data, sort_keys=True).encode()).hexdigest()
        )

    @property
    def metadata(self):
        return {
            key: self._data.get(key, NOT_RECORDED)
            for key in (
                "run_id",
                "world_frame",
                "clock",
                "code_identity",
                "config",
                "config_sha256",
            )
        }

    def epoch(self, index, mode="fused"):
        return self._data["histories"][mode][index]

    def epoch_index(self, time_s):
        """Nearest saved epoch, clamped at either end; never fabricate a result."""
        if not len(self.times) or not np.isfinite(time_s):
            raise ValueError("No finite epoch to select")
        return int(np.abs(self.times - time_s).argmin())

    def decisions(self, index, mode="fused"):
        if self.version == 1:
            epoch = self.epoch(index, mode)
            return {key: epoch.get(key, NOT_RECORDED) for key in DECISION_KEYS}
        entry = self._data["decision_index"][mode][index]
        if self.path is None:
            return {key: self.epoch(index, mode).get(key, NOT_RECORDED) for key in DECISION_KEYS}
        # The bundle name is fixed. Results cannot request arbitrary file reads.
        with (self.path.parent / "fusion_decisions.jsonl").open("rb") as stream:
            stream.seek(entry["offset"])
            raw = stream.read(entry["length"])
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("Fusion decision record integrity failure")
        record = json.loads(raw)
        if record["mode"] != mode or record["epoch_index"] != index:
            raise ValueError("Fusion decision record binding mismatch")
        return record["decisions"]

    def source_path(self, sensor_id):
        relative = self._data.get("sources", {}).get(sensor_id, {}).get("source")
        if relative is None and self.path is not None and self.version == 1:
            recipe = self.path.parent / "experiment.json"
            if recipe.is_file():
                relative = (
                    json.loads(recipe.read_text())
                    .get("sensors", {})
                    .get(sensor_id, {})
                    .get("source")
                )
        if relative is None or self.path is None:
            return None
        return (self.path.parent / relative).resolve()

    def verify_sources(self):
        from shared.sensing.fusion.experiment import fingerprints

        result = {}
        for name, source in self._data.get("sources", {}).items():
            path = self.source_path(name)
            if path is not None:
                from shared.sensing.loader import _resolve_frames_dir

                try:
                    path = _resolve_frames_dir(path)
                except FileNotFoundError:
                    path = None
            result[name] = bool(
                path and path.is_dir() and fingerprints(path) == source.get("hashes")
            )
        return result


def write_result(result, directory):
    """Write validated compact data plus lazy decisions inside an unpublished directory."""
    directory = Path(directory)
    data = deepcopy(result)
    index = {}
    with (directory / "fusion_decisions.jsonl").open("wb") as stream:
        for mode, history in data["histories"].items():
            index[mode] = []
            for i, epoch in enumerate(history):
                decisions = {key: epoch.pop(key, []) for key in DECISION_KEYS}
                decisions["updates"] = [
                    {"track_id": track["track_id"], **deepcopy(source)}
                    for track in epoch["tracks"]
                    for source in track["sources"]
                ]
                for track in epoch["tracks"]:
                    for source in track["sources"]:
                        for key in ("residual", "measurement_covariance", "innovation_covariance"):
                            source.pop(key, None)
                raw = (
                    json.dumps(
                        {"mode": mode, "epoch_index": i, "decisions": decisions},
                        allow_nan=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode()
                index[mode].append(
                    {
                        "offset": stream.tell(),
                        "length": len(raw),
                        "sha256": hashlib.sha256(raw).hexdigest(),
                    }
                )
                stream.write(raw)
    data["decision_index"] = index
    FusionResult(data)
    (directory / "fusion_results.json").write_text(
        json.dumps(data, allow_nan=False, separators=(",", ":")) + "\n"
    )
    return FusionResult(directory)
