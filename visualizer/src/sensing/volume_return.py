"""One-level native-evidence breadcrumb; no closed volume widgets are retained."""

from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget
from shiboken6 import isValid


class VolumeReturn(QWidget):
    def __init__(self, visualizer, context, native, *, arm="A", native_sources=None):
        super().__init__(visualizer)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle("Volume trial evidence")
        self.visualizer = visualizer
        self.context = context
        self.native = Path(native).resolve()
        self.native_sources = {Path(path).resolve() for path in native_sources or [native]}
        layout = QVBoxLayout(self)
        message = QLabel(
            f"Experiment → {context['trial_id']} → Recording ({arm})\nReport selection is preserved for return to the volume."
        )
        message.setWordWrap(True)
        layout.addWidget(message)
        button = QPushButton("Return to volume")
        button.clicked.connect(self.return_to_volume)
        layout.addWidget(button)
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._check_source)
        self.timer.start()

    def _check_source(self):
        current = getattr(self.visualizer, "current_scenario_path", None)
        if current:
            current = Path(current).resolve()
            if current.is_file():
                current = current.parent
            if current not in self.native_sources:
                self.close()

    def return_to_volume(self):
        from .volume_window import open_sensing_volume

        panel = self.visualizer.ui_manager.panels.get("sensing")
        host = getattr(panel, "dashboard_host", None)
        if host is not None:
            host.suspend()
        context = self.context
        self.close()
        open_sensing_volume(self.visualizer, context["volume"], context=context)

    def closeEvent(self, event):
        self.timer.stop()
        panel = self.visualizer.ui_manager.panels.get("sensing")
        if panel is not None:
            panel.return_host.hide()
        super().closeEvent(event)


def install_return(visualizer, context, native, *, arm="A", native_sources=None):
    old = getattr(visualizer, "volume_return", None)
    if old is not None and isValid(old):
        old.close()
    dialog = VolumeReturn(visualizer, context, native, arm=arm, native_sources=native_sources)
    visualizer.volume_return = dialog
    visualizer.ui_manager.ensure_panel("sensing")
    panel = visualizer.ui_manager.panels["sensing"]
    panel.return_layout.addWidget(dialog)
    panel.return_host.show()
    dialog.show()
    return dialog
