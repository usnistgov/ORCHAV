"""Qt application and window identity helpers for ORCHAV."""

from __future__ import annotations

from importlib.resources import files

from PySide6.QtGui import QIcon, QPixmap
from PySide6.QtWidgets import QApplication, QWidget

APPLICATION_NAME = "ORCHAV"
ORGANIZATION_NAME = "NIST"
WINDOW_TITLE = "ORCHAV - Radio Propagation Analysis & Visualization"
APPLICATION_ICON_SIZES = (16, 20, 24, 30, 32, 36, 40, 48, 60, 64, 72, 80, 96, 128, 256)


def build_orchav_icon() -> QIcon:
    """Load the multi-resolution ORCHAV application icon."""
    png_directory = files("visualizer").joinpath("resources", "branding", "png")
    icon = QIcon()
    for size in APPLICATION_ICON_SIZES:
        resource = png_directory.joinpath(f"orchav-app-icon-{size}.png")
        pixmap = QPixmap()
        if not pixmap.loadFromData(resource.read_bytes(), "PNG"):
            raise RuntimeError(f"Could not load ORCHAV icon resource: {resource}")
        icon.addPixmap(pixmap)
    return icon


def apply_application_identity(app: QApplication | None = None) -> QIcon:
    """Apply ORCHAV name, organization, and icon to the current Qt app."""
    app = app or QApplication.instance()
    icon = build_orchav_icon()
    if app is not None:
        app.setApplicationName(APPLICATION_NAME)
        app.setOrganizationName(ORGANIZATION_NAME)
        app.setWindowIcon(icon)
    return icon


def apply_window_identity(window: QWidget) -> None:
    """Apply ORCHAV title and icon to a top-level window."""
    window.setWindowTitle(WINDOW_TITLE)
    window.setWindowIcon(apply_application_identity())
