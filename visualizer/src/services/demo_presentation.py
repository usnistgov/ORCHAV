"""Temporary presentation chrome around the embedded scene viewport."""

from pathlib import Path

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtWidgets import (
    QDockWidget,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)
from shiboken6 import isValid


class DemoPresentation:
    """Hide interface chrome reversibly and explain the current Demo view."""

    def __init__(self, visualizer, service):
        self.viz = visualizer
        self.service = service
        self.state = None
        self.footer = None
        self._connected = False
        self.play_button = self.seek_slider = self.time_label = None
        self._playback_timer = None

    def enter(self, *, clean_view: bool, captions: bool, compact=False, playback=False) -> None:
        """Attach a footer without moving or reparenting the native canvas."""
        if self.state is not None:
            return
        workspace = getattr(self.viz, "_visualization_workspace", None)
        splitter = getattr(workspace, "splitter", None)
        host = getattr(workspace, "viewport_host", None)
        if (
            not isinstance(self.viz, QMainWindow)
            or splitter is None
            or splitter.count() < 2
            or not isinstance(host, QWidget)
            or not isValid(host)
            or splitter.indexOf(host) < 0
            or not isinstance(host.layout(), QVBoxLayout)
        ):
            raise ValueError("Demo presentation requires the embedded scene viewport")

        controls = splitter.widget(0)
        docks = self.viz.findChildren(QDockWidget)
        chrome = self.viz.findChildren(QToolBar)
        menu = self.viz.menuWidget()
        if menu is not None:
            chrome.append(menu)
        status = self.viz.findChild(QStatusBar, options=Qt.FindDirectChildrenOnly)
        if status is not None:
            chrome.append(status)
        self.state = {
            "clean_view": clean_view,
            "window": self.viz.saveState(),
            "splitter": splitter,
            "splitter_state": splitter.saveState(),
            "controls": controls,
            "controls_visible": not controls.isHidden(),
            "docks": [(dock, not dock.isHidden()) for dock in docks],
            "chrome": [(widget, not widget.isHidden()) for widget in chrome],
            "host": host,
            "host_spacing": host.layout().spacing(),
        }
        self._create_footer(host, captions=captions, compact=compact, playback=playback)
        host.layout().setSpacing(0)
        host.layout().addWidget(self.footer)
        if clean_view:
            controls.hide()
            for widget in (*docks, *chrome):
                widget.hide()
        self.service.changed.connect(self.refresh)
        self._connected = True
        self.refresh()
        self.footer.show()

    def _create_footer(
        self, host: QWidget, *, captions: bool, compact=False, playback=False
    ) -> None:
        footer = self.footer = QFrame(host)
        footer.setObjectName("demoPresentationFooter")
        footer.setAccessibleName("Demo presentation")
        footer.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        footer.setStyleSheet(
            "QFrame#demoPresentationFooter {background: #101c2b; "
            "border-top: 1px solid #355064;}"
            "QFrame#demoPresentationFooter QLabel {color: #e3edf4; background: transparent; "
            "font-size: 15px;}"
            "QFrame#demoPresentationFooter QLabel#demoPresentationTitle "
            "{color: #72ddd4; font-weight: 600; font-size: 18px;}"
            "QFrame#demoPresentationFooter QPushButton {background: #20364a; color: #f2f7fa; "
            "border: 1px solid #55728a; border-radius: 4px; padding: 5px 10px; font-size: 13px;}"
            "QFrame#demoPresentationFooter QPushButton:hover {background: #304d65;}"
            "QFrame#demoPresentationFooter QPushButton:focus {border: 2px solid #72ddd4;}"
        )
        layout = QVBoxLayout(footer)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(4)
        self.title = QLabel(footer)
        self.title.setObjectName("demoPresentationTitle")
        self.title.setTextFormat(Qt.PlainText)
        self.title.setWordWrap(True)
        layout.addWidget(self.title)
        self.title.setVisible(not compact)
        self.caption = QLabel(footer)
        self.caption.setObjectName("demoPresentationCaption")
        self.caption.setTextFormat(Qt.PlainText)
        self.caption.setWordWrap(True)
        self.caption.setVisible(captions)
        layout.addWidget(self.caption)
        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        if playback:
            self.play_button = QPushButton("Pause", footer)
            self.play_button.setToolTip(
                "Pause or resume recorded playback without ending the demo."
            )
            self.play_button.clicked.connect(self._toggle_playback)
            buttons.addWidget(self.play_button)
            self.seek_slider = QSlider(Qt.Horizontal, footer)
            self.seek_slider.setAccessibleName("Demo sensing result")
            self.seek_slider.setToolTip("Seek a saved sensing result.")
            self.seek_slider.setTracking(False)
            self.seek_slider.valueChanged.connect(self._seek)
            buttons.addWidget(self.seek_slider, 1)
            self.time_label = QLabel(footer)
            buttons.addWidget(self.time_label)
            self._playback_timer = QTimer(footer)
            self._playback_timer.setInterval(250)
            self._playback_timer.timeout.connect(self._refresh_playback)
            self._playback_timer.start()
            self._refresh_playback()
        else:
            buttons.addStretch()
        self.resume_button = QPushButton("Resume camera", footer)
        self.resume_button.setToolTip("Resume the automatic camera motion; Demo keeps running.")
        self.resume_button.clicked.connect(lambda: self.service.resume_camera())
        buttons.addWidget(self.resume_button)
        self.stop_button = QPushButton("Stop demo", footer)
        self.stop_button.setToolTip("Stop Demo and restore your controls and settings (Esc).")
        self.stop_button.clicked.connect(lambda: self.service.stop())
        buttons.addWidget(self.stop_button)
        layout.addLayout(buttons)
        if compact:
            layout.setContentsMargins(8, 5, 8, 5)
            self.caption.setStyleSheet("font-size: 12px;")

    def _toggle_playback(self):
        self.viz.animation_controller.toggle_animation()
        self._refresh_playback()

    def _seek(self, value):
        # Reuse the ordinary player's ordinal-to-source-frame mapping.
        slider = getattr(self.viz, "step_slider", None)
        if slider is not None:
            slider.setValue(value)
            self.viz.flush_pending_slider_scrub()
        self._refresh_playback()

    def _refresh_playback(self):
        if self.play_button is None or not isValid(self.play_button):
            return
        self.play_button.setText(
            "Pause" if getattr(self.viz, "animation_running", False) else "Play"
        )
        slider = getattr(self.viz, "step_slider", None)
        if slider is not None and not self.seek_slider.isSliderDown():
            with QSignalBlocker(self.seek_slider):
                self.seek_slider.setRange(slider.minimum(), slider.maximum())
                self.seek_slider.setValue(slider.value())
            position = f"{slider.value() + 1}/{slider.maximum() + 1}"
            playback = getattr(self.viz, "sensing_playback", None)
            timestamp = getattr(playback, "timestamps", {}).get(
                getattr(self.viz, "animation_step", None)
            )
            self.time_label.setText(
                f"{timestamp:.2f} s · {position}" if timestamp is not None else position
            )

    def refresh(self) -> None:
        """Update captions and the camera action from the Demo coordinator."""
        if self.footer is None or not isValid(self.footer):
            return
        scenario = getattr(self.viz, "scenario", None)
        name = getattr(scenario, "name", None)
        if not isinstance(name, str) or not name.strip():
            defaults = getattr(scenario, "view_defaults", None)
            tour = defaults.get("demo_tour") if isinstance(defaults, dict) else None
            name = tour.get("label") if isinstance(tour, dict) else None
        if not isinstance(name, str) or not name.strip():
            root = getattr(scenario, "root", None) or getattr(
                self.viz, "current_scenario_path", None
            )
            name = Path(root).name if root else "Scene"
            if name.endswith((".yaml", ".yml")):
                name = Path(root).parent.name
            name = name.replace("_", " ").replace("-", " ")
            name = name[:1].upper() + name[1:]
        self.title.setText(name.strip()[:100])
        self.caption.setText(self.service.caption_text())
        self.resume_button.setVisible(bool(self.service.camera_paused))

    def owns_widget(self, widget) -> bool:
        """Let footer controls handle their input without triggering takeover."""
        return (
            self.footer is not None
            and isValid(self.footer)
            and isinstance(widget, QWidget)
            and (widget is self.footer or self.footer.isAncestorOf(widget))
        )

    def restore(self, *, restore_docks: bool = True) -> None:
        """Remove only this footer and restore the captured interface visibility."""
        if self._connected:
            try:
                self.service.changed.disconnect(self.refresh)
            except (RuntimeError, TypeError):
                pass  # The service can already be disposed during window teardown.
            self._connected = False
        if self._playback_timer is not None and isValid(self._playback_timer):
            self._playback_timer.stop()
        self._playback_timer = None
        state, self.state = self.state, None
        footer, self.footer = self.footer, None
        if footer is not None and isValid(footer):
            footer.hide()
            if state is not None and isValid(state["host"]):
                state["host"].layout().removeWidget(footer)
                state["host"].layout().setSpacing(state["host_spacing"])
            footer.deleteLater()
        if state is None or not state["clean_view"]:
            return
        if restore_docks and isValid(self.viz):
            self.viz.restoreState(state["window"])
            for dock, visible in state["docks"]:
                if isValid(dock):
                    dock.setVisible(visible)
        for widget, visible in state["chrome"]:
            if isValid(widget):
                widget.setVisible(visible)
        if isValid(state["controls"]):
            state["controls"].setVisible(state["controls_visible"])
        if isValid(state["splitter"]):
            state["splitter"].restoreState(state["splitter_state"])
