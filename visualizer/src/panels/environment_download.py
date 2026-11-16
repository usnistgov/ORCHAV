"""Nonblocking optional-sky download controls for the Rendering panel."""

from __future__ import annotations

from queue import Empty, Queue
from threading import Event, Thread

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..lighting.assets import (
    DownloadCancelled,
    cached_environment,
    download_environment,
    environment_catalog,
)


def _download(asset, cancel: Event, messages: Queue) -> None:
    """Worker owns only plain Python state; never access Qt after window closure."""
    try:
        download_environment(
            asset,
            cancelled=cancel,
            progress=lambda done, total: messages.put(("progress", done * 100 // total)),
        )
        messages.put(("done", None))
    except DownloadCancelled:
        messages.put(("cancelled", None))
    except Exception as exc:  # Worker errors must become actionable UI status.
        messages.put(("error", str(exc)))


class EnvironmentDownload(QWidget):
    """One selected catalog entry with download/retry/cancel and recipe preference."""

    ready = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.selected = None
        self._cancel = None
        self._messages = None
        self._closed = False
        self._timer = QTimer(self)
        self._timer.setInterval(80)
        self._timer.timeout.connect(self._poll)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setOpenExternalLinks(True)
        layout.addWidget(self.status)
        self.match_lighting = QCheckBox("Use matching lighting when selecting a sky")
        self.match_lighting.setChecked(True)
        self.match_lighting.setToolTip(
            "Apply the sky's light intensities and sun direction. Uncheck to keep the current light rig when changing maps."
        )
        layout.addWidget(self.match_lighting)
        row = QHBoxLayout()
        self.download = QPushButton()
        self.download.clicked.connect(self.start_download)
        row.addWidget(self.download)
        self.cancel = QPushButton("Cancel")
        self.cancel.clicked.connect(self._cancel_download)
        self.cancel.hide()
        row.addWidget(self.cancel)
        self.progress = QProgressBar()
        self.progress.hide()
        row.addWidget(self.progress, 1)
        layout.addLayout(row)
        self.hide()

    @property
    def awaiting_download(self) -> bool:
        return self.selected is not None and cached_environment(self.selected) is None

    def select(self, identifier: str) -> None:
        """Selecting an absent map does not change current lighting or start I/O."""
        self._stop_job()
        self.selected = environment_catalog().get(identifier)
        self.setVisible(self.selected is not None)
        if self.selected is None:
            return
        asset = self.selected
        self.download.setText(f"Download & apply — {asset.size / 1024**2:.1f} MiB")
        self.download.setEnabled(True)
        present = cached_environment(asset) is not None
        self.download.setVisible(not present)
        self.status.setText(
            ("Installed" if present else "Not installed; current lighting is unchanged")
            + f' · 2K · CC0 · <a href="{asset.source_page}">Poly Haven</a>'
        )
        if present:
            self.ready.emit(asset.id)

    def start_download(self) -> None:
        if self.selected is None or self._cancel is not None or self._closed:
            return
        self._cancel = Event()
        self._messages = Queue()
        self.download.setEnabled(False)
        self.cancel.show()
        self.progress.setValue(0)
        self.progress.show()
        self.status.setText("Downloading sky; the current scene remains available…")
        Thread(
            target=_download, args=(self.selected, self._cancel, self._messages), daemon=True
        ).start()
        self._timer.start()

    def _poll(self) -> None:
        if self._messages is None or self._closed:
            return
        try:
            while True:
                kind, value = self._messages.get_nowait()
                if kind == "progress":
                    self.progress.setValue(value)
                    continue
                identifier = self.selected.id
                self._stop_job()
                if kind == "done":
                    self.select(identifier)
                else:
                    self.download.setEnabled(True)
                    self.status.setText(
                        "Download cancelled; current lighting is unchanged."
                        if kind == "cancelled"
                        else f"Download failed: {value}. Try Download again, or use Default offline."
                    )
                break
        except Empty:
            pass

    def _cancel_download(self) -> None:
        if self._cancel is not None:
            self._cancel.set()
            self.cancel.setEnabled(False)
            self.status.setText("Cancelling download…")

    def _stop_job(self) -> None:
        self._timer.stop()
        if self._cancel is not None:
            self._cancel.set()
        self._cancel = self._messages = None
        self.cancel.hide()
        self.cancel.setEnabled(True)
        self.progress.hide()

    def cleanup(self) -> None:
        self._closed = True
        self._stop_job()
