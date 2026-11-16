"""Prepare compact 3D sensing and saved tracks without loading dashboard histories."""

from concurrent.futures import CancelledError
from dataclasses import dataclass
from threading import Event

from .saved_tracking import SavedTrackingLoader
from .scene_recording import prepare_scene_recording, provider_token


@dataclass(frozen=True)
class PreparedSensing:
    """One recording's ready-to-use 3D records and separately saved tracks."""

    scene: object
    tracking: object
    tracking_error: str | None


def prepare_sensing(
    provider, cancelled: Event, progress=lambda *_: None, scene_ready=lambda _: None
):
    """Prepare scene overlays; interactive dashboard histories load when requested."""
    scene = prepare_scene_recording(provider, cancelled, progress)
    scene_ready(scene)
    progress("Preparing saved tracking", 0, 1)
    tracking, tracking_error = SavedTrackingLoader(provider.source).load(cancelled)
    if tracking is not None:
        _ = tracking.history.tracking_plot_data
    if cancelled.is_set():
        raise CancelledError()
    if provider_token(provider) != scene.token:
        raise RuntimeError("Recording changed during sensing preparation")
    progress("Sensing ready", len(scene.records), len(scene.records))
    return PreparedSensing(scene, tracking, tracking_error)
