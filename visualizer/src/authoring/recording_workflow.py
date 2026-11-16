"""Create a new authoring document from a recording's acquisition inputs."""

import threading
from pathlib import Path

from PySide6.QtCore import QObject, Signal

from shared.scenarios.paths import resolve_actor_resource

from .compiler import ScenarioCompiler
from .persistence import create_scenario_copy, save_document


def acquisition_scenario_path(document) -> Path | None:
    """Find the input YAML beside the explicitly selected recording directory.

    This is a candidate acquisition input, never a conversion of recorded poses
    into authored motion. The ordinary authoring loader still checks it.
    """
    if document is None or document.path is None or not document.scenario.recorded_poses:
        return None
    mapping = document.scenario.source_snapshot.to_mapping()
    directory = mapping.get("data", {}).get("files", {}).get("directory", "frames")
    frames = resolve_actor_resource(str(directory), scenario_root=document.path.parent)
    candidate = frames.parent / "scenario.yaml"
    return (
        candidate
        if candidate.is_file() and candidate.resolve() != document.path.resolve()
        else None
    )


class RecordingVariantJob(QObject):
    """Copy and validate inputs off the GUI thread; never copy generated frames."""

    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = threading.Event()

    def start(self, source, destination):
        def prepare():
            try:
                source_path = Path(source).resolve()
                output = Path(destination).resolve()
                if output.is_relative_to(source_path.parent):
                    raise ValueError(
                        "Choose a new directory outside the original acquisition directory."
                    )
                if output.exists() and (not output.is_dir() or any(output.iterdir())):
                    raise ValueError("Choose a new or empty directory for the editable variant.")
                document = create_scenario_copy(source_path, output)
                if self._cancelled.is_set():
                    return
                compiler = ScenarioCompiler()
                try:
                    save_document(document, compiler=compiler)
                finally:
                    compiler.close()
                if not self._cancelled.is_set():
                    self.succeeded.emit(document)
            except (OSError, ValueError, TypeError, RuntimeError) as exc:
                if not self._cancelled.is_set():
                    self.failed.emit(str(exc))

        threading.Thread(target=prepare, name="orchav-recording-variant", daemon=True).start()

    def cancel(self):
        """Suppress late UI publication; an in-flight atomic save may finish."""
        self._cancelled.set()
