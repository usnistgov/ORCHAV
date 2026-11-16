"""Regenerate standard sensing summaries from saved observations and reports.

Run ``python -m generator.figures.summary.saved_sensing SOURCE -o OUTPUT``.
This reader does not run acquisition, signal processing, detection, or tracking.
It retains the final full product and compact histories instead of every RD map.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from contextlib import ExitStack, closing
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from generator.figures.summary.sensing import maybe_generate_sensing_summary
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.sensing.evaluation_truth import extract_recorded_evaluation_truth
from shared.sensing.loader import _open_provider, iter_frames_sensing
from shared.sensing.plotting.doppler_summary import (
    doppler_processing_matches,
    doppler_summary_event,
)
from shared.sensing.plotting.figures import build_tracking_history

FIGURE_CHOICES = (
    "range_profile",
    "range_doppler",
    "doppler",
    "truth",
    "detection_history",
    "tracking",
    "tracking_xy",
    "tracking_xyz",
    "tracking_cartesian",
    "detection",
    "angles",
    "acquisition",
    "detection_stages",
    "tracking_decisions",
)


def _source_file_hashes(manifest_path: Path) -> dict[str, str]:
    """Hash the authoritative manifest and each listed frame chunk in order."""
    manifest = json.loads(manifest_path.read_text())
    files = [
        manifest_path,
        *(manifest_path.parent / chunk["file"] for chunk in manifest.get("chunks", [])),
    ]
    hashes = {}
    for path in files:
        with path.open("rb") as stream:
            hashes[path.relative_to(manifest_path.parent).as_posix()] = hashlib.file_digest(
                stream, "sha256"
            ).hexdigest()
    return hashes


class _CartesianSceneContexts:
    """Load only exact bound device poses for optional Cartesian truth overlays.

    Coherent products may be stored after their geometry epoch. The explicit
    display binding selects the scene; truth never chooses a nearest frame or
    supplies an estimated device pose. One lazy provider is closed by the caller.
    """

    def __init__(self, source: Path, stack: ExitStack):
        self.source, self.stack = source, stack
        self.provider = self.manifest = None

    def for_frame(self, frame: dict) -> dict | None:
        sensing = frame["_typed_sensing"]
        temporal = sensing["temporal"]
        truth_time = temporal.get("truth", {}).get("reference_time_s")
        if truth_time is None or frame.get("_frame_timestamp_s") == truth_time:
            return frame
        binding = temporal.get("display_binding", {})
        if binding.get("policy") not in {"source_scene_exact", "exact"}:
            return None
        scene_id = binding.get("scene_frame_id")
        if not isinstance(scene_id, int) or isinstance(scene_id, bool):
            return None
        if self.provider is None:
            _, self.manifest, self.provider = _open_provider(self.source)
            self.stack.callback(self.provider.close)
        if (
            self.manifest.generation_id != frame["_source_generation_id"]
            or self.manifest.frame_set_id != frame["_source_frame_set_id"]
        ):
            raise ValueError("Bound Cartesian scene belongs to a different recording")
        if scene_id not in self.manifest.frame_ids:
            return None
        scene = self.provider.load_frame_projection(
            scene_id,
            FrameReadRequest(
                components=frozenset({FrameComponent.DEVICES, FrameComponent.PROVENANCE})
            ),
        ).frame
        if scene.timestamp_s != truth_time:
            return None
        acquisition = temporal["acquisition"]
        pairs = sensing.get("pair_axis", [])
        if len(pairs) != 1:
            return None
        for device in ("tx", "rx"):
            names = getattr(scene, f"{device}_names")
            index = pairs[0].get(f"{device}_device_index")
            if (
                not isinstance(index, int)
                or isinstance(index, bool)
                or not 0 <= index < len(names)
                or names[index] != acquisition[f"{device}_sensor_id"]
            ):
                return None
        return {
            "scene_time_s": scene.timestamp_s,
            "tx_positions_snapshot": scene.tx_positions,
            "rx_positions_snapshot": scene.rx_positions,
        }


def generate_saved_sensing_summary(
    source: Path,
    output_root: Path,
    *,
    figures: Sequence[str] | None = None,
    frame_index: int | None = None,
    output_format: str = "png",
    range_limits_m: tuple[float, float] | None = None,
    velocity_limits_m_s: tuple[float, float] | None = None,
    power_limits_db: tuple[float, float] | None = None,
) -> list[Path]:
    """Render the standard summary, preserving authored products and result times.

    ``frame_index`` stops at that exact saved frame; it must carry a complete
    sensing result. Histories include preceding complete results. Incomplete
    acquisitions are counted and retained only for the acquisition diagram.
    Outputs must be fresh so existing generated evidence cannot be overwritten.
    """
    source, output_root = Path(source), Path(output_root)
    if (output_root / "sensing").exists():
        raise FileExistsError(f"Choose a fresh summary output: {output_root / 'sensing'}")
    selected = set(figures) if figures is not None else None
    if selected is not None and (unknown := selected - set(FIGURE_CHOICES)):
        raise ValueError(f"Unknown sensing figures: {', '.join(sorted(unknown))}")
    if output_format not in {"png", "svg", "pdf"}:
        raise ValueError("Summary format must be png, svg, or pdf")
    display_limits = {
        "range_limits_m": range_limits_m,
        "velocity_limits_m_s": velocity_limits_m_s,
        "power_limits_db": power_limits_db,
    }
    for name, bounds in display_limits.items():
        if bounds is not None:
            values = np.asarray(bounds, dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all() or values[0] >= values[1]:
                raise ValueError(f"{name} must contain two finite, increasing values")

    history: list[dict[str, Any]] = []
    detections: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    windows: list[dict[str, Any]] = []
    receipts: list[dict[str, Any]] = []
    last_sensing: dict[str, Any] | None = None
    incomplete_count = 0
    identity = None
    found_frame = False
    source_identity = None
    with ExitStack() as stack:
        frames = stack.enter_context(closing(iter_frames_sensing(source)))
        scene_contexts = _CartesianSceneContexts(source, stack)
        for frame in frames:
            step = int(frame["_frame_index"])
            if frame_index is not None and step > frame_index:
                break
            found_frame = step == frame_index
            sensing = frame["_typed_sensing"]
            temporal = sensing["temporal"]
            result = temporal.get("result")
            complete = isinstance(result, dict)
            same_kind = sum(
                isinstance(w["temporal"].get("result"), dict) == complete for w in windows
            )
            if same_kind < (2 if complete else 1):
                windows.append(
                    {
                        key: deepcopy(sensing[key])
                        for key in (
                            "schema_version",
                            "processing_strategy",
                            "config_resolved",
                            "temporal",
                            "products",
                        )
                        if key in sensing
                    }
                )
            if not complete:
                incomplete_count += 1
                if found_frame:
                    raise ValueError(f"Frame {frame_index} has no complete sensing result")
                continue
            cfg = sensing["config_resolved"]
            candidate_identity = (
                temporal["acquisition"]["pair_id"],
                cfg["range_mode"],
                cfg["carrier_freq_hz"],
            )
            if identity is not None and candidate_identity != identity:
                raise ValueError(
                    "Saved summary requires one sensor pair and range/velocity convention"
                )
            identity = candidate_identity
            reference_time_s = float(result["reference_time_s"])
            if not np.isfinite(reference_time_s) or (
                history and reference_time_s <= history[-1]["reference_time_s"]
            ):
                raise ValueError(
                    "Saved result reference times must be finite and strictly increasing"
                )
            stored_truth = extract_recorded_evaluation_truth(sensing)
            if stored_truth is not None:
                truth_time = temporal.get("truth", {}).get("reference_time_s")
                if truth_time != reference_time_s or stored_truth.get(
                    "_evaluation_unavailable_reason"
                ):
                    raise ValueError(
                        "Truth and result reference times must match for a summary overlay"
                    )
                if "velocity_reference_policy" in stored_truth:
                    stored_truth["evaluation_reference_valid"] = True
            entry = {
                "step": step,
                "reference_time_s": reference_time_s,
                "target_ids": [],
                "target_ranges": [],
                "target_speeds": [],
                "target_velocity_valid": [],
                **(stored_truth or {}),
                "det_ranges": [d["range_m"] for d in sensing.get("detections") or []],
                "det_velocities": [d["velocity_m_s"] for d in sensing.get("detections") or []],
            }
            if selected is not None and "angles" in selected:
                from shared.sensing.angular import angular_event

                entry["angular_summary"] = angular_event(sensing)
            if selected is None or selected & {
                "tracking_xy",
                "tracking_xyz",
                "tracking_cartesian",
            }:
                from shared.sensing.plotting.cartesian import cartesian_tracking_event

                entry["cartesian_summary"] = cartesian_tracking_event(
                    sensing, scene_context=scene_contexts.for_frame(frame)
                )
            history.append(entry)
            for detection in sensing.get("detections") or []:
                detections.append({**deepcopy(detection), "frame": step})
            if selected is None or "doppler" in selected:
                event = doppler_summary_event(sensing)
                if event is not None:
                    spectra.append(event)
            if selected is not None and "detection" in selected and stored_truth is not None:
                from generator.core.sensing.hooks import compute_frame_association

                association = compute_frame_association(
                    entry,
                    range_gate_m=cfg["eval_association_range_gate_m"],
                    velocity_gate_m_s=cfg["eval_association_velocity_gate_m_s"],
                )
                if association.get("evaluated", True):
                    associations.append(association)
            last_sensing = sensing
            source_identity = {
                "generation_id": frame["_source_generation_id"],
                "frame_set_id": frame["_source_frame_set_id"],
            }
            receipts.append(
                {
                    "frame_index": step,
                    "reference_time_s": reference_time_s,
                    "acquisition_id": temporal["acquisition"]["acquisition_id"],
                    "processing_recipe_id": result["processing_recipe_id"],
                    "detection_count": len(sensing.get("detections") or []),
                }
            )
            if found_frame:
                break
    if frame_index is not None and not found_frame:
        raise ValueError(f"Saved frame {frame_index} was not found")
    if last_sensing is None:
        raise ValueError("The recording has no complete sensing result")
    tracks = build_tracking_history(detections)
    if selected is None:
        selected = {"detection_history"}
        available_kinds = {p.get("kind") for p in last_sensing["products"].values()}
        if "range_doppler" in available_kinds:
            selected.add("range_doppler")
        # Derived recipes may bind distinct source observations while retaining
        # independently verified equal processing settings. Keep every original
        # recipe identity in the receipt; unknown differences remain incompatible.
        if spectra and all(doppler_processing_matches(spectra[0], event) for event in spectra):
            selected.add("doppler")
        if "range_profile" in available_kinds:
            selected.add("range_profile")
        if tracks:
            selected.add("tracking")
        cartesian_states = [
            track["position_m"]
            for entry in history
            for track in entry["cartesian_summary"]["tracks"]
        ]
        if cartesian_states:
            selected.add("tracking_xy")
            if np.isfinite(cartesian_states).all(axis=1).any():
                selected.update({"tracking_xyz", "tracking_cartesian"})
    inspections = sorted(selected & {"detection_stages", "tracking_decisions"})
    scenario = SimpleNamespace(
        generator_summary={
            "enabled": True,
            "create": ["sensing"],
            "sensing": {
                "figures": sorted(selected),
                "inspection_figures": inspections,
                "history_time_axis": "time",
            },
            "output": {"dir": output_root, "format": output_format},
        }
    )
    generated = (
        maybe_generate_sensing_summary(
            last_sensing,
            history,
            gt_state=history[-1],
            scenario_context=scenario,
            association_history=associations,
            sensing_event_history=windows,
            doppler_history=spectra,
            tracking_history=tracks,
            result_label="selected result" if frame_index is not None else "last frame",
            output_root=output_root,
            strict=True,
            **display_limits,
        )
        or []
    )
    manifest_path = source / "frames_manifest.json"
    if not manifest_path.is_file():
        manifest_path = source / "frames" / "frames_manifest.json"
    source_hashes = _source_file_hashes(manifest_path)
    receipt = {
        "kind": "standard_sensing_summary_from_saved_products",
        "source": str(source.resolve()),
        "source_identity": source_identity,
        "manifest_sha256": source_hashes[manifest_path.name],
        "source_files_sha256": source_hashes,
        "figures": sorted(selected),
        "incomplete_acquisition_count": incomplete_count,
        "result_count": len(receipts),
        "results": receipts,
        "outputs": [str(p.relative_to(output_root)) for p in generated],
        "processing": "Saved products, detections, and track identities; no detector or tracker rerun.",
        "evaluation": (
            "Requested detection metrics use persisted evaluation gates and bound truth."
            if associations
            else None
        ),
    }
    if any(value is not None for value in display_limits.values()):
        receipt["display_limits"] = display_limits
    receipt_path = output_root / "sensing" / "summary_provenance.json"
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return [*generated, receipt_path]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Saved frame directory or recording root")
    parser.add_argument(
        "-o", "--output", required=True, type=Path, help="Fresh summary output root"
    )
    parser.add_argument(
        "--figure",
        choices=FIGURE_CHOICES,
        action="append",
        help="Select a standard figure; repeat for several",
    )
    parser.add_argument(
        "--frame-index", type=int, help="Stop at this exact complete result's saved frame"
    )
    parser.add_argument("--format", choices=("png", "svg", "pdf"), default="png")
    parser.add_argument(
        "--range-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Range bounds in metres for zoomed profile/RD views; full views stay complete",
    )
    parser.add_argument(
        "--velocity-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Velocity bounds in m/s for the zoomed RD view",
    )
    parser.add_argument(
        "--power-limits",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        help="Common RD color limits in channel-power dB; does not normalize samples",
    )
    args = parser.parse_args(argv)
    generate_saved_sensing_summary(
        args.source,
        args.output,
        figures=args.figure,
        frame_index=args.frame_index,
        output_format=args.format,
        range_limits_m=args.range_limits,
        velocity_limits_m_s=args.velocity_limits,
        power_limits_db=args.power_limits,
    )


if __name__ == "__main__":
    main()
