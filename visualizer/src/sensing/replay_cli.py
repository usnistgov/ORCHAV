"""Installed, renderer-free CLI for exporting one persisted sensing replay.

The command reads native sensing schema-v4 data through the shared packed-frame
provider, constructs the same immutable replay event used by the interactive
visualizer, and delegates publication to the deterministic replay exporter.
It does not load a scene, initialize Qt, select a renderer, or require a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from shared.extensions.sensing import SENSING_SCHEMA_VERSION, assert_sensing_contract
from shared.frames.contracts import FrameComponent, FrameReadRequest
from shared.frames.manifest import FRAMES_MANIFEST_FILENAME, load_frame_manifest
from shared.frames.providers import Hdf5Provider
from shared.sensing.replay import ReplayEvent, ReplaySceneContext, build_replay_event

_REPLAY_REQUEST = FrameReadRequest(
    components=frozenset(
        {
            FrameComponent.DEVICES,
            FrameComponent.PROVENANCE,
            FrameComponent.TARGETS,
            FrameComponent.SENSING,
        }
    )
)
_REPLAY_PROBE_REQUEST = FrameReadRequest(
    components=frozenset({FrameComponent.PROVENANCE, FrameComponent.SENSING}),
    sensing_products=frozenset({"products"}),
)
_SCENE_TIME_REQUEST = FrameReadRequest(components=frozenset({FrameComponent.PROVENANCE}))
_REPLAY_EXPORT_DEPENDENCY_MODULES = frozenset({"matplotlib", "PIL"})


class ReplaySourceError(ValueError):
    """Raised when persisted frames cannot form a canonical replay event."""


def _is_missing_replay_export_dependency(exc: ModuleNotFoundError) -> bool:
    """Return whether ``exc`` names one absent replay-export dependency.

    Only top-level plotting dependencies are translated into
    installation guidance.  Missing ORCHAV modules or submodules from a broken
    dependency installation must retain their original traceback.
    """

    return exc.name in _REPLAY_EXPORT_DEPENDENCY_MODULES


def _resolve_frames_dir(source: Path) -> Path:
    """Resolve a packed-frame directory from a scenario/output or frame path."""

    candidate = Path(source)
    if (candidate / FRAMES_MANIFEST_FILENAME).is_file():
        return candidate
    nested = candidate / "frames"
    if (nested / FRAMES_MANIFEST_FILENAME).is_file():
        return nested
    raise FileNotFoundError(f"{FRAMES_MANIFEST_FILENAME} not found in {candidate} or {nested}")


def _finite_time(value: Any) -> float | None:
    """Return one finite real timestamp without accepting booleans."""

    if isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _bound_scene_identity(sensing: Mapping[str, Any]) -> int | str:
    """Return the exact persisted scene frame bound to a sensing result."""

    temporal = sensing.get("temporal")
    display = temporal.get("display_binding") if isinstance(temporal, Mapping) else None
    bound_frame_id = display.get("scene_frame_id") if isinstance(display, Mapping) else None
    if isinstance(bound_frame_id, bool) or not isinstance(bound_frame_id, (int, str)):
        raise ReplaySourceError(
            "sensing.temporal.display_binding.scene_frame_id is missing or invalid"
        )
    if isinstance(bound_frame_id, str) and not bound_frame_id.strip():
        raise ReplaySourceError("sensing.temporal.display_binding.scene_frame_id must not be empty")
    return bound_frame_id


def _sample_bound_scene_time(
    sensing: Mapping[str, Any],
    bound_frame_id: int | str,
) -> float | None:
    """Recover scene time from exact acquisition membership when needed."""

    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    samples = acquisition.get("samples") if isinstance(acquisition, Mapping) else None
    if isinstance(samples, list):
        matching_times: list[float] = []
        for sample in samples:
            if not isinstance(sample, Mapping):
                continue
            source = sample.get("source")
            if not isinstance(source, Mapping) or source.get("scene_frame_id") != bound_frame_id:
                continue
            timestamp = _finite_time(sample.get("sample_time_s"))
            if timestamp is not None:
                matching_times.append(timestamp)
        if matching_times:
            return min(matching_times)

    truth = temporal.get("truth") if isinstance(temporal, Mapping) else None
    if isinstance(truth, Mapping) and truth.get("scene_frame_id") == bound_frame_id:
        timestamp = _finite_time(truth.get("reference_time_s"))
        if timestamp is not None:
            return timestamp
    return None


def _bound_scene_time(
    provider: Hdf5Provider,
    sensing: Mapping[str, Any],
    bound_frame_id: int | str,
) -> float:
    """Resolve the bound scene timestamp from its row or acquisition lineage."""

    if isinstance(bound_frame_id, int) and provider.has_frame(bound_frame_id):
        projection = provider.load_frame_projection(bound_frame_id, _SCENE_TIME_REQUEST)
        timestamp = _finite_time(projection.frame.timestamp_s)
        if timestamp is not None:
            return timestamp

    timestamp = _sample_bound_scene_time(sensing, bound_frame_id)
    if timestamp is not None:
        return timestamp
    raise ReplaySourceError(
        "could not resolve the display-bound scene timestamp from the frame set "
        "or acquisition membership"
    )


def _event_from_frame(provider: Hdf5Provider, frame_id: int) -> ReplayEvent:
    """Load and validate one frame as a fresh, exact replay selection."""

    projection = provider.load_frame_projection(frame_id, _REPLAY_REQUEST)
    frame = projection.frame
    sensing = frame.sensing
    if not isinstance(sensing, Mapping) or sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        raise ReplaySourceError(
            f"frame {frame_id} does not contain native sensing schema-v{SENSING_SCHEMA_VERSION}"
        )
    bound_frame_id = _bound_scene_identity(sensing)
    bound_time_s = _bound_scene_time(provider, sensing, bound_frame_id)
    context = ReplaySceneContext(
        storage_frame_id=int(frame.frame_index),
        bound_scene_frame_id=bound_frame_id,
        current_scene_frame_id=bound_frame_id,
        bound_scene_time_s=bound_time_s,
        current_scene_time_s=bound_time_s,
        pinned=False,
    )
    return build_replay_event(frame, context, require_product_data=True)


def _probe_selection_key(
    provider: Hdf5Provider,
    frame_id: int,
) -> tuple[float, str, str, str] | None:
    """Return the production-equivalent ordering key for an eligible result.

    Rows without native sensing or with an incomplete acquisition are not replay
    candidates.  Once a row declares a complete schema-v4 acquisition, malformed
    metadata is an error: the CLI must not hide a corrupt newest result by falling
    back to an older event.
    """

    projection = provider.load_frame_projection(frame_id, _REPLAY_PROBE_REQUEST)
    sensing = projection.frame.sensing
    if not isinstance(sensing, Mapping) or sensing.get("schema_version") != SENSING_SCHEMA_VERSION:
        return None
    temporal = sensing.get("temporal")
    acquisition = temporal.get("acquisition") if isinstance(temporal, Mapping) else None
    if not isinstance(acquisition, Mapping) or acquisition.get("completeness") != "complete":
        return None

    try:
        assert_sensing_contract(
            {"sensing": sensing},
            require_schema_version=True,
            require_described_products=False,
        )
        result = temporal.get("result")
        if not isinstance(result, Mapping):
            raise ReplaySourceError("sensing.temporal.result must be a mapping")
        reference_time_s = _finite_time(result.get("reference_time_s"))
        processing_run_id = result.get("processing_run_id")
        acquisition_id = acquisition.get("acquisition_id")
        if reference_time_s is None:
            raise ReplaySourceError("result.reference_time_s must be finite")
        if not isinstance(processing_run_id, str) or not processing_run_id:
            raise ReplaySourceError("result.processing_run_id must be a non-empty string")
        if not isinstance(acquisition_id, str) or not acquisition_id:
            raise ReplaySourceError("acquisition.acquisition_id must be a non-empty string")
    except (TypeError, ValueError) as exc:
        raise ReplaySourceError(
            f"complete native sensing frame {frame_id} has invalid replay metadata: {exc}"
        ) from exc
    return (
        reference_time_s,
        processing_run_id,
        acquisition_id,
        f"{type(frame_id).__name__}:{frame_id!r}",
    )


def load_replay_event(source: str | Path, *, frame_id: int | None = None) -> ReplayEvent:
    """Load one canonical replay, defaulting to the latest eligible result.

    An explicit frame fails with its exact validation error.  Without
    ``frame_id``, metadata-only and incomplete rows are skipped and complete
    results use the same reference-time/run/acquisition/storage ordering as the
    production visualizer.  A malformed complete native result fails closed.
    """

    frames_dir = _resolve_frames_dir(Path(source))
    manifest = load_frame_manifest(frames_dir)
    available = tuple(int(candidate) for candidate in manifest.frame_ids)
    provider = Hdf5Provider(frames_dir.parent, frames_subdir=frames_dir.name)
    try:
        if frame_id is not None:
            if frame_id not in available:
                raise ReplaySourceError(f"frame {frame_id} is not present in {frames_dir}")
            try:
                return _event_from_frame(provider, frame_id)
            except (KeyError, TypeError, ValueError) as exc:
                raise ReplaySourceError(
                    f"frame {frame_id} cannot form a canonical replay: {exc}"
                ) from exc

        eligible: list[tuple[tuple[float, str, str, str], int]] = []
        for candidate in available:
            key = _probe_selection_key(provider, candidate)
            if key is not None:
                eligible.append((key, candidate))
        if not eligible:
            raise ReplaySourceError(f"no canonical sensing replay found in {frames_dir}")
        _key, selected = max(eligible, key=lambda item: item[0])
        try:
            return _event_from_frame(provider, selected)
        except (KeyError, TypeError, ValueError) as exc:
            raise ReplaySourceError(
                f"selected frame {selected} cannot form a canonical replay: {exc}"
            ) from exc
    finally:
        provider.close()


def _build_parser() -> argparse.ArgumentParser:
    """Build the installed replay-export parser without loading plotting libraries."""

    parser = argparse.ArgumentParser(
        prog="orchav-sensing-replay",
        description=(
            "Export a deterministic, no-GPU sensing replay summary from native "
            "schema-v4 HDF5 frames."
        ),
    )
    parser.add_argument(
        "source",
        type=Path,
        help="Scenario/output root or directory containing frames_manifest.json.",
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=None,
        help="Exact storage frame ID (default: latest canonical sensing result).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        required=True,
        help="New directory to publish atomically; it must not already exist.",
    )
    parser.add_argument(
        "--allow-blocked",
        action="store_true",
        help="Explicitly export a replay whose credibility preflight is blocked.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the installed no-GPU sensing replay export command."""

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        event = load_replay_event(args.source, frame_id=args.frame)
        try:
            from visualizer.src.sensing.replay_export import export_replay_session
        except ModuleNotFoundError as exc:
            if not _is_missing_replay_export_dependency(exc):
                raise
            raise ReplaySourceError(
                "sensing replay export dependencies are unavailable. "
                "Install ORCHAV with: python -m pip install orchav"
            ) from exc
        result = export_replay_session(
            event,
            args.output_dir,
            allow_blocked=args.allow_blocked,
        )
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    summary = {
        "artifact_count": len(result.artifacts),
        "blocked_override_used": result.blocked_override_used,
        "event_manifest_id": result.event_manifest_id,
        "event_view_manifest_id": result.event_view_manifest_id,
        "output_directory": str(result.output_directory),
        "preflight_disposition": event.preflight.disposition,
        "reference_time_s": event.key.reference_time_s,
        "storage_frame_id": event.scene_binding.storage_frame_id,
    }
    sys.stdout.write(json.dumps(summary, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - installed command owns this path
    raise SystemExit(main())


__all__ = ["ReplaySourceError", "load_replay_event", "main"]
