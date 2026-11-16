"""Bind one embedded coverage view to the panel's canonical metric selection.

This adapter owns no formulas or rendering code. Regular saved grids use the
volume controller; other recordings keep the existing height-plane controls.
"""

import numpy as np
from PySide6.QtCore import QSignalBlocker, QTimer
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..renderers.protocol import volume_rendering_unavailable_reason


class CoveragePanelView:
    """Lazily attach one source-bound view; release it when the source changes."""

    def __init__(self, panel):
        self.panel = panel
        self.visualizer = panel.parent
        self.widget = QWidget()
        self.layout = QVBoxLayout(self.widget)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.note = QLabel()
        self.note.setWordWrap(True)
        self.layout.addWidget(self.note)
        self.window = None
        self.failed_source = None
        self._syncing = False
        self._queued = False
        self.timer = QTimer(self.widget)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.sync)

    def queue_sync(self):
        """Coalesce startup/status changes into one settled UI update."""
        if not self._queued:
            self._queued = True
            self.timer.start(0)

    def availability(self):
        source = getattr(self.visualizer, "coverage_data", None)
        if not source or not source.get("coverage_file"):
            return "Height-plane view · this recording has no regular saved volume."
        reason = volume_rendering_unavailable_reason(getattr(self.visualizer, "renderer", None))
        if reason:
            return f"Height-plane view · {reason}"
        heights = np.asarray(source.get("heights", []), dtype=float)
        if heights.size < 2:
            return "Height-plane view · one recorded height."
        spacing = np.diff(heights)
        if (
            not np.isfinite(heights).all()
            or np.any(spacing <= 0)
            or not np.allclose(spacing, spacing[0])
        ):
            return "Height-plane view · recorded heights are not uniformly spaced."
        return ""

    def sync(self):
        self._queued = False
        if self._syncing:
            return
        self._syncing = True
        try:
            self._sync()
        finally:
            self._syncing = False

    def _sync(self):
        viz = self.visualizer
        source = getattr(viz, "coverage_data", None)
        toggle = self.panel.widgets["coverage_toggle"]
        with QSignalBlocker(toggle):
            toggle.setChecked(
                bool(getattr(getattr(viz, "app_state", None), "show_coverage", False))
            )
        if self.window is not None and (self.window.closed or source is not self.window.source):
            if not self.window.closed:
                self.window.close()
            self.window = None
        scripted = getattr(viz, "coverage_volume_window", None)
        if scripted is not None and not scripted.closed and not scripted.embedded:
            # The compatibility API is used by reproducible capture scripts.
            # Do not replace their explicitly requested legacy view on refresh.
            return
        reason = self.availability()
        if source is not None and self.failed_source is source:
            reason = self.note.text()
        if reason:
            if self.window is not None:
                self.window.close()
                self.window = None
            self.note.setText(reason)
            self.note.setVisible(bool(source))
            self.panel.set_volume_controls_active(False)
            return
        self.note.hide()
        self.panel.set_volume_controls_active(True)
        if self.window is None:
            if not bool(getattr(getattr(viz, "app_state", None), "show_coverage", False)):
                self.note.setText("Enable coverage to inspect slices, 3D regions and projections.")
                self.note.show()
                return
            # A scripted inspector shares the renderer's analysis layer. Retire
            # it before the main panel takes ownership, never render both.
            old = getattr(viz, "coverage_volume_window", None)
            if old is not None and not old.closed:
                old.close()
            sensing = getattr(viz, "sensing_volume_window", None)
            if sensing is not None and not sensing.closed:
                sensing.close()
            from .volume_window import CoverageVolumeWindow

            self.window = CoverageVolumeWindow(viz, embedded=True, on_failure=self._failed)
            viz.coverage_volume_window = self.window
            self.layout.addWidget(self.window)
            self.window.show()
        else:
            self.window.sync_from_panel()

    def _failed(self, reason):
        """Keep a usable height view after unsupported grids/read errors."""
        self.failed_source = self.window.source
        self.note.setText(f"Height-plane view · {reason}")
        self.queue_sync()
        viz = self.visualizer
        viz.coverage_volume_active = False
        viz.schedule_update()

    def reset(self):
        if self.window is not None and not self.window.closed:
            self.window.close()
        self.window = None
        self.failed_source = None
        self.visualizer.coverage_volume_active = False
